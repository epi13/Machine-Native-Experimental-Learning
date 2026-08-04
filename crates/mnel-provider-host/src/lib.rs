//! Runtime admission policy and reusable snapshot storage for learned providers.
//!
//! Dynamic loading is deliberately deferred. This crate establishes the policy and
//! in-memory contracts that any loader must obey.

use std::collections::BTreeMap;
use std::sync::Arc;

use mnel_provider_api::{Digest32, ABI_VERSION_V1};

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
        self.manifests.insert(manifest.provider_id.clone(), manifest);
        Ok(())
    }

    pub fn get(&self, provider_id: &str) -> Option<&ProviderManifest> {
        self.manifests.get(provider_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> Digest32 {
        Digest32 { bytes: [value; 32] }
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
}
