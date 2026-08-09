//! Runtime admission policy and reusable snapshot storage for learned providers.
//!
//! Dynamic loading is deliberately deferred. This crate establishes the policy and
//! in-memory contracts that any loader must obey.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use mnel_provider_api::{
    Digest32, ProviderStatusV1, ABI_VERSION_V1, AUTHORITY_DIAGNOSTIC_ONLY, OUTPUT_ANOMALY_SCORE,
    PROVIDER_STATUS_ABSTAINED, PROVIDER_STATUS_BUDGET_EXCEEDED, PROVIDER_STATUS_COMPLETED,
    PROVIDER_STATUS_INVALID_INPUT, PROVIDER_STATUS_OUT_OF_DISTRIBUTION,
    PROVIDER_STATUS_RUNTIME_ERROR, RESULT_FLAG_OUT_OF_DISTRIBUTION, RESULT_FLAG_TRUNCATED_PAYLOAD,
    VERDICT_SEMANTICS_NOT_A_VERDICT,
};
use mnel_provider_sdk::{
    DiagnosticResult, Invocation, InvocationIdentity, LearnedProvider, ProviderError,
    ResourceBudget, SnapshotRef,
};

pub mod placement;
use placement::{PlacementCapabilities, PlacementError, PlacementPolicy};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImplementationLanguage {
    Rust,
    C,
    Cpp,
    Zig,
    Wasm,
    Python,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionTier {
    NativeTrusted,
    WasmQuarantined,
    ExternalExperimental,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeLanguageException {
    pub rationale: String,
    pub benchmark_evidence_ids: Vec<String>,
    pub threat_review_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderManifest {
    pub provider_id: String,
    pub provider_version: String,
    pub declaration_identity: Digest32,
    pub artifact_identity: Digest32,
    pub language: ImplementationLanguage,
    pub tier: ExecutionTier,
    pub abi_version: u32,
    pub persistent_host: bool,
    pub language_exception: Option<NativeLanguageException>,
    pub placement_policy: PlacementPolicy,
    pub placement_capabilities: PlacementCapabilities,
}

impl ProviderManifest {
    pub fn validate(&self) -> Result<(), AdmissionError> {
        if self.provider_id.trim().is_empty() || self.provider_version.trim().is_empty() {
            return Err(AdmissionError::MissingIdentity);
        }
        if self.abi_version != ABI_VERSION_V1 {
            return Err(AdmissionError::UnsupportedAbi);
        }
        if !self.persistent_host {
            return Err(AdmissionError::ProcessPerInvocationForbidden);
        }
        if self.placement_policy.execution_device == placement::ExecutionDevice::Cpu
            && self.placement_policy.offload == placement::OffloadMode::SequentialCpu
        {
            return Err(AdmissionError::InvalidPlacement(
                PlacementError::InvalidPolicy,
            ));
        }
        if self.placement_capabilities.cpu_precisions.is_empty()
            || self.placement_capabilities.cuda_precisions.is_empty()
        {
            return Err(AdmissionError::InvalidPlacement(
                PlacementError::MissingCpuPrecision,
            ));
        }
        if self.placement_policy.max_vram_bytes == Some(0)
            || self.placement_policy.host_memory_budget_bytes == Some(0)
            || self
                .placement_policy
                .host_memory_budget_bytes
                .is_some_and(|budget| self.placement_policy.model_storage_bytes > budget)
        {
            return Err(AdmissionError::InvalidPlacement(
                PlacementError::InvalidPolicy,
            ));
        }
        match self.tier {
            ExecutionTier::NativeTrusted => {
                if self.language != ImplementationLanguage::Rust {
                    let exception = self
                        .language_exception
                        .as_ref()
                        .ok_or(AdmissionError::NonRustNativeRequiresException)?;
                    if exception.rationale.trim().is_empty()
                        || exception.benchmark_evidence_ids.is_empty()
                        || exception.threat_review_id.trim().is_empty()
                    {
                        return Err(AdmissionError::IncompleteLanguageException);
                    }
                }
            }
            ExecutionTier::WasmQuarantined => {
                if self.language != ImplementationLanguage::Wasm {
                    return Err(AdmissionError::TierLanguageMismatch);
                }
            }
            ExecutionTier::ExternalExperimental => {
                if self.language == ImplementationLanguage::Rust
                    && self.language_exception.is_some()
                {
                    return Err(AdmissionError::UnnecessaryLanguageException);
                }
            }
        }
        if self.language_exception.is_some()
            && !(self.tier == ExecutionTier::NativeTrusted
                && self.language != ImplementationLanguage::Rust)
        {
            return Err(AdmissionError::UnnecessaryLanguageException);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionError {
    MissingIdentity,
    UnsupportedAbi,
    ProcessPerInvocationForbidden,
    NonRustNativeRequiresException,
    IncompleteLanguageException,
    TierLanguageMismatch,
    UnnecessaryLanguageException,
    InvalidPlacement(PlacementError),
    DuplicateProvider,
}

#[derive(Clone, Debug)]
pub struct CachedSnapshot {
    pub identity: Digest32,
    pub feature_extractor_identity: Digest32,
    pub payload: Arc<[u8]>,
}

#[derive(Default)]
pub struct SnapshotCache {
    entries: BTreeMap<[u8; 32], CachedSnapshot>,
}

impl SnapshotCache {
    pub fn insert(&mut self, snapshot: CachedSnapshot) -> Arc<[u8]> {
        let key = snapshot.identity.bytes;
        if let Some(existing) = self.entries.get(&key) {
            return Arc::clone(&existing.payload);
        }
        let payload = Arc::clone(&snapshot.payload);
        self.entries.insert(key, snapshot);
        payload
    }

    pub fn get(&self, identity: &Digest32) -> Option<&CachedSnapshot> {
        self.entries.get(&identity.bytes)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Default)]
pub struct ProviderCatalog {
    manifests: BTreeMap<String, ProviderManifest>,
}

impl ProviderCatalog {
    pub fn admit(&mut self, manifest: ProviderManifest) -> Result<(), AdmissionError> {
        manifest.validate()?;
        if self.manifests.contains_key(&manifest.provider_id) {
            return Err(AdmissionError::DuplicateProvider);
        }
        self.manifests
            .insert(manifest.provider_id.clone(), manifest);
        Ok(())
    }

    pub fn get(&self, provider_id: &str) -> Option<&ProviderManifest> {
        self.manifests.get(provider_id)
    }

    pub fn remove(&mut self, provider_id: &str) -> Option<ProviderManifest> {
        self.manifests.remove(provider_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderState {
    Active,
    Quarantined {
        failures: u32,
        reason: ProviderError,
    },
}

struct HostedProvider {
    provider: Arc<dyn LearnedProvider>,
    state: ProviderState,
    failures: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvocationMeasurement {
    pub elapsed_ns: u64,
    pub bytes_copied: u64,
    pub snapshots_reused: u32,
    pub output_bytes: u64,
    pub execution_mode: String,
    pub precision: String,
    pub placement_reason: String,
    pub process_rss_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct HostedInvocation {
    pub status: ProviderStatusV1,
    pub output_kind: u32,
    pub scalar_value: f64,
    pub calibration_band: u32,
    pub flags: u64,
    pub payload: Vec<u8>,
    pub authority: u32,
    pub verdict_semantics: u32,
    pub state: ProviderState,
    pub measurement: InvocationMeasurement,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostError {
    Admission(AdmissionError),
    AcceleratorBackendUnavailable,
    UnknownProvider,
    ProviderQuarantined,
    MissingSnapshot,
    InvalidInvocation(ProviderError),
    SnapshotIdentityMismatch,
}

pub struct ProviderHost {
    catalog: ProviderCatalog,
    snapshots: SnapshotCache,
    providers: BTreeMap<String, HostedProvider>,
    max_failures: u32,
    max_output_bytes: usize,
}

impl ProviderHost {
    pub fn new(max_failures: u32, max_output_bytes: usize) -> Self {
        Self {
            catalog: ProviderCatalog::default(),
            snapshots: SnapshotCache::default(),
            providers: BTreeMap::new(),
            max_failures: max_failures.max(1),
            max_output_bytes: max_output_bytes.max(1),
        }
    }

    pub fn admit(
        &mut self,
        manifest: ProviderManifest,
        provider: Arc<dyn LearnedProvider>,
    ) -> Result<(), HostError> {
        if manifest.placement_policy.execution_device == placement::ExecutionDevice::Cuda
            || manifest.placement_policy.offload == placement::OffloadMode::SequentialCpu
        {
            return Err(HostError::AcceleratorBackendUnavailable);
        }
        let provider_id = manifest.provider_id.clone();
        self.catalog
            .admit(manifest.clone())
            .map_err(HostError::Admission)?;
        self.providers.insert(
            provider_id,
            HostedProvider {
                provider,
                state: ProviderState::Active,
                failures: 0,
            },
        );
        Ok(())
    }

    pub fn register_snapshot(&mut self, snapshot: CachedSnapshot) -> Arc<[u8]> {
        self.snapshots.insert(snapshot)
    }

    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    pub fn state(&self, provider_id: &str) -> Option<&ProviderState> {
        self.providers
            .get(provider_id)
            .map(|provider| &provider.state)
    }

    pub fn unload(&mut self, provider_id: &str) -> Result<(), HostError> {
        if self.providers.remove(provider_id).is_none() {
            return Err(HostError::UnknownProvider);
        }
        self.catalog.remove(provider_id);
        Ok(())
    }

    pub fn invoke(
        &mut self,
        provider_id: &str,
        identities: InvocationIdentity,
        snapshot_ids: &[Digest32],
        budget: ResourceBudget,
    ) -> Result<HostedInvocation, HostError> {
        let hosted = self
            .providers
            .get(provider_id)
            .ok_or(HostError::UnknownProvider)?;
        if hosted.state != ProviderState::Active {
            return Err(HostError::ProviderQuarantined);
        }
        let provider = Arc::clone(&hosted.provider);
        let mut payloads = Vec::with_capacity(snapshot_ids.len());
        for identity in snapshot_ids {
            let snapshot = self
                .snapshots
                .get(identity)
                .ok_or(HostError::MissingSnapshot)?;
            if snapshot.identity != *identity {
                return Err(HostError::SnapshotIdentityMismatch);
            }
            payloads.push(Arc::clone(&snapshot.payload));
        }
        let snapshot_refs = payloads
            .iter()
            .zip(snapshot_ids.iter())
            .map(|(payload, identity)| SnapshotRef {
                schema_version: 1,
                identity: *identity,
                feature_extractor_identity: Digest32::ZERO,
                payload,
            })
            .collect::<Vec<_>>();
        let invocation = Invocation::new(identities, budget, snapshot_refs)
            .map_err(HostError::InvalidInvocation)?;
        let started = Instant::now();
        let outcome = provider.infer(&invocation);
        let elapsed_ns = started.elapsed().as_nanos() as u64;
        let result = match outcome {
            Ok(result) => self.normalize_success(
                provider_id,
                result,
                budget,
                snapshot_ids.len() as u32,
                elapsed_ns,
            ),
            Err(error) => {
                self.normalize_error(provider_id, error, snapshot_ids.len() as u32, elapsed_ns)
            }
        };
        Ok(result)
    }

    fn normalize_success(
        &mut self,
        provider_id: &str,
        result: DiagnosticResult,
        budget: ResourceBudget,
        snapshots_reused: u32,
        elapsed_ns: u64,
    ) -> HostedInvocation {
        let output_limit = self.max_output_bytes.min(budget.memory_bytes as usize);
        let mut flags = 0;
        let mut payload = result.payload;
        let original_output_bytes = payload.len() as u64;
        let mut status = PROVIDER_STATUS_COMPLETED;
        if elapsed_ns > budget.wall_time_ns {
            status = PROVIDER_STATUS_BUDGET_EXCEEDED;
            self.record_failure(provider_id, ProviderError::BudgetExceeded);
        }
        if payload.len() > output_limit {
            payload.truncate(output_limit);
            flags |= RESULT_FLAG_TRUNCATED_PAYLOAD;
            status = PROVIDER_STATUS_BUDGET_EXCEEDED;
            self.record_failure(provider_id, ProviderError::BudgetExceeded);
        }
        if result.out_of_distribution {
            flags |= RESULT_FLAG_OUT_OF_DISTRIBUTION;
        }
        let state = self.provider_state(provider_id);
        HostedInvocation {
            status,
            output_kind: result.output_kind,
            scalar_value: result.value,
            calibration_band: result.calibration_band,
            flags,
            payload,
            authority: AUTHORITY_DIAGNOSTIC_ONLY,
            verdict_semantics: VERDICT_SEMANTICS_NOT_A_VERDICT,
            state,
            measurement: measurement(elapsed_ns, original_output_bytes, snapshots_reused),
        }
    }

    fn normalize_error(
        &mut self,
        provider_id: &str,
        error: ProviderError,
        snapshots_reused: u32,
        elapsed_ns: u64,
    ) -> HostedInvocation {
        self.record_failure(provider_id, error);
        let (status, flags) = match error {
            ProviderError::Abstained => (PROVIDER_STATUS_ABSTAINED, 0),
            ProviderError::InvalidBudget | ProviderError::EmptySnapshot => {
                (PROVIDER_STATUS_INVALID_INPUT, 0)
            }
            ProviderError::BudgetExceeded => (PROVIDER_STATUS_BUDGET_EXCEEDED, 0),
            ProviderError::OutOfDistribution => (
                PROVIDER_STATUS_OUT_OF_DISTRIBUTION,
                RESULT_FLAG_OUT_OF_DISTRIBUTION,
            ),
            ProviderError::MissingSnapshots
            | ProviderError::NonFiniteResult
            | ProviderError::RuntimeFailure => (PROVIDER_STATUS_RUNTIME_ERROR, 0),
        };
        HostedInvocation {
            status,
            output_kind: OUTPUT_ANOMALY_SCORE,
            scalar_value: 0.0,
            calibration_band: 0,
            flags,
            payload: Vec::new(),
            authority: AUTHORITY_DIAGNOSTIC_ONLY,
            verdict_semantics: VERDICT_SEMANTICS_NOT_A_VERDICT,
            state: self.provider_state(provider_id),
            measurement: measurement(elapsed_ns, 0, snapshots_reused),
        }
    }

    fn record_failure(&mut self, provider_id: &str, reason: ProviderError) {
        if let Some(provider) = self.providers.get_mut(provider_id) {
            provider.failures = provider.failures.saturating_add(1);
            if provider.failures >= self.max_failures {
                provider.state = ProviderState::Quarantined {
                    failures: provider.failures,
                    reason,
                };
            }
        }
    }

    fn provider_state(&self, provider_id: &str) -> ProviderState {
        self.providers.get(provider_id).map_or(
            ProviderState::Quarantined {
                failures: 0,
                reason: ProviderError::RuntimeFailure,
            },
            |provider| provider.state.clone(),
        )
    }
}

fn measurement(elapsed_ns: u64, output_bytes: u64, snapshots_reused: u32) -> InvocationMeasurement {
    InvocationMeasurement {
        elapsed_ns,
        bytes_copied: 0,
        snapshots_reused,
        output_bytes,
        execution_mode: "cpu".to_owned(),
        precision: "float32".to_owned(),
        placement_reason: "native Rust CPU backend; no accelerator adapter attached".to_owned(),
        process_rss_bytes: process_rss_bytes(),
    }
}

fn process_rss_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages = contents.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(resident_pages.saturating_mul(4096))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> Digest32 {
        Digest32 { bytes: [value; 32] }
    }

    fn ok<T, E: core::fmt::Debug>(result: Result<T, E>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => panic!("unexpected test error: {error:?}"),
        }
    }

    fn manifest(language: ImplementationLanguage, tier: ExecutionTier) -> ProviderManifest {
        ProviderManifest {
            provider_id: "state.hidden-markov-model".to_owned(),
            provider_version: "0.1.0".to_owned(),
            declaration_identity: digest(1),
            artifact_identity: digest(2),
            language,
            tier,
            abi_version: ABI_VERSION_V1,
            persistent_host: true,
            language_exception: None,
            placement_policy: PlacementPolicy::default(),
            placement_capabilities: PlacementCapabilities::default(),
        }
    }

    #[test]
    fn rust_is_the_native_default() {
        assert_eq!(
            manifest(ImplementationLanguage::Rust, ExecutionTier::NativeTrusted).validate(),
            Ok(())
        );
    }

    #[test]
    fn catalog_rejects_duplicate_admission() {
        let mut catalog = ProviderCatalog::default();
        let candidate = manifest(ImplementationLanguage::Rust, ExecutionTier::NativeTrusted);
        assert_eq!(catalog.admit(candidate.clone()), Ok(()));
        assert_eq!(
            catalog.admit(candidate),
            Err(AdmissionError::DuplicateProvider)
        );
    }

    #[test]
    fn non_rust_native_requires_evidence_backed_exception() {
        let mut candidate = manifest(ImplementationLanguage::Cpp, ExecutionTier::NativeTrusted);
        assert_eq!(
            candidate.validate(),
            Err(AdmissionError::NonRustNativeRequiresException)
        );
        candidate.language_exception = Some(NativeLanguageException {
            rationale: "Specialized GPU kernel unavailable in Rust toolchain".to_owned(),
            benchmark_evidence_ids: vec!["sha256:benchmark".to_owned()],
            threat_review_id: "sha256:threat-review".to_owned(),
        });
        assert_eq!(candidate.validate(), Ok(()));
    }

    #[test]
    fn python_cannot_enter_the_native_hot_path() {
        let candidate = manifest(ImplementationLanguage::Python, ExecutionTier::NativeTrusted);
        assert_eq!(
            candidate.validate(),
            Err(AdmissionError::NonRustNativeRequiresException)
        );
    }

    #[test]
    fn snapshot_payload_is_reused_by_arc() {
        let payload: Arc<[u8]> = Arc::from([1_u8, 2, 3, 4]);
        let mut cache = SnapshotCache::default();
        let returned = cache.insert(CachedSnapshot {
            identity: digest(7),
            feature_extractor_identity: digest(8),
            payload: Arc::clone(&payload),
        });
        assert!(Arc::ptr_eq(&payload, &returned));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn snapshot_identity_is_append_only_for_repeated_registration() {
        let mut cache = SnapshotCache::default();
        let identity = digest(15);
        cache.insert(CachedSnapshot {
            identity,
            feature_extractor_identity: digest(16),
            payload: Arc::from([1_u8]),
        });
        cache.insert(CachedSnapshot {
            identity,
            feature_extractor_identity: digest(17),
            payload: Arc::from([9_u8]),
        });
        assert_eq!(
            cache.get(&identity).map(|snapshot| &*snapshot.payload),
            Some(&[1_u8][..])
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn process_local_host_reuses_snapshot_and_normalizes_diagnostic_output() {
        use mnel_provider_classical::HiddenMarkovProvider;
        use mnel_provider_sdk::{InvocationIdentity, ResourceBudget};

        let provider_id = "state.hidden-markov-model";
        let mut host = ProviderHost::new(3, 128);
        ok(host.admit(
            manifest(ImplementationLanguage::Rust, ExecutionTier::NativeTrusted),
            Arc::new(HiddenMarkovProvider::new()),
        ));
        let snapshot_id = digest(9);
        host.register_snapshot(CachedSnapshot {
            identity: snapshot_id,
            feature_extractor_identity: digest(10),
            payload: Arc::from([0_u8, 0, 1, 1, 2]),
        });
        let result = ok(host.invoke(
            provider_id,
            InvocationIdentity {
                declaration: digest(1),
                model: digest(2),
                calibration: digest(3),
                query: digest(4),
            },
            &[snapshot_id],
            ResourceBudget {
                wall_time_ns: 1_000_000,
                operation_limit: 100,
                memory_bytes: 128,
            },
        ));
        assert_eq!(result.status, PROVIDER_STATUS_COMPLETED);
        assert_eq!(result.authority, AUTHORITY_DIAGNOSTIC_ONLY);
        assert_eq!(result.verdict_semantics, VERDICT_SEMANTICS_NOT_A_VERDICT);
        assert_eq!(result.measurement.bytes_copied, 0);
        assert_eq!(result.measurement.snapshots_reused, 1);
        assert_eq!(host.snapshot_count(), 1);
        assert_eq!(host.unload(provider_id), Ok(()));
        assert_eq!(host.state(provider_id), None);
    }

    #[test]
    fn cpu_host_rejects_explicit_accelerator_mode_instead_of_silently_falling_back() {
        let mut candidate = manifest(ImplementationLanguage::Rust, ExecutionTier::NativeTrusted);
        candidate.placement_policy.execution_device = placement::ExecutionDevice::Cuda;
        let mut host = ProviderHost::new(3, 128);
        assert_eq!(
            host.admit(candidate, Arc::new(VerboseProvider)),
            Err(HostError::AcceleratorBackendUnavailable)
        );
    }

    struct VerboseProvider;

    impl LearnedProvider for VerboseProvider {
        fn infer(&self, _invocation: &Invocation<'_>) -> Result<DiagnosticResult, ProviderError> {
            Ok(DiagnosticResult {
                output_kind: OUTPUT_ANOMALY_SCORE,
                value: 0.5,
                calibration_band: 0,
                out_of_distribution: false,
                payload: vec![7; 32],
            })
        }
    }

    #[test]
    fn host_owned_result_buffer_is_bounded() {
        let mut host = ProviderHost::new(3, 8);
        let provider_id = "state.hidden-markov-model";
        ok(host.admit(
            manifest(ImplementationLanguage::Rust, ExecutionTier::NativeTrusted),
            Arc::new(VerboseProvider),
        ));
        let snapshot_id = digest(11);
        host.register_snapshot(CachedSnapshot {
            identity: snapshot_id,
            feature_extractor_identity: digest(12),
            payload: Arc::from([0_u8]),
        });
        let result = ok(host.invoke(
            provider_id,
            InvocationIdentity {
                declaration: digest(1),
                model: digest(2),
                calibration: digest(3),
                query: digest(4),
            },
            &[snapshot_id],
            ResourceBudget {
                wall_time_ns: 1_000_000,
                operation_limit: 10,
                memory_bytes: 8,
            },
        ));
        assert_eq!(result.status, PROVIDER_STATUS_BUDGET_EXCEEDED);
        assert_eq!(result.payload.len(), 8);
        assert_eq!(result.flags, RESULT_FLAG_TRUNCATED_PAYLOAD);
    }

    struct FailingProvider;

    impl LearnedProvider for FailingProvider {
        fn infer(&self, _invocation: &Invocation<'_>) -> Result<DiagnosticResult, ProviderError> {
            Err(ProviderError::RuntimeFailure)
        }
    }

    #[test]
    fn repeated_provider_failures_quarantine_without_affecting_other_catalog_state() {
        let mut host = ProviderHost::new(2, 128);
        let provider_id = "state.hidden-markov-model";
        ok(host.admit(
            manifest(ImplementationLanguage::Rust, ExecutionTier::NativeTrusted),
            Arc::new(FailingProvider),
        ));
        let snapshot_id = digest(13);
        host.register_snapshot(CachedSnapshot {
            identity: snapshot_id,
            feature_extractor_identity: digest(14),
            payload: Arc::from([0_u8]),
        });
        let identities = InvocationIdentity {
            declaration: digest(1),
            model: digest(2),
            calibration: digest(3),
            query: digest(4),
        };
        let budget = ResourceBudget {
            wall_time_ns: 1_000_000,
            operation_limit: 10,
            memory_bytes: 128,
        };
        assert_eq!(
            ok(host.invoke(provider_id, identities, &[snapshot_id], budget)).status,
            PROVIDER_STATUS_RUNTIME_ERROR
        );
        assert_eq!(
            ok(host.invoke(provider_id, identities, &[snapshot_id], budget)).status,
            PROVIDER_STATUS_RUNTIME_ERROR
        );
        assert_eq!(
            host.state(provider_id),
            Some(&ProviderState::Quarantined {
                failures: 2,
                reason: ProviderError::RuntimeFailure,
            })
        );
        match host.invoke(provider_id, identities, &[snapshot_id], budget) {
            Ok(_) => panic!("quarantined providers must not be invoked"),
            Err(error) => assert_eq!(error, HostError::ProviderQuarantined),
        }
    }
}
