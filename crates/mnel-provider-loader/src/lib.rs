//! Small reviewed unsafe boundary for the versioned MNEL provider shared-library ABI.
//!
//! The loader is for native-trusted providers. It validates the descriptor and all
//! host-owned result metadata, copies output before returning, serializes calls, and
//! keeps the dynamic library alive for the entire provider handle lifetime. A malformed
//! native provider can still crash its own process while executing arbitrary code; a
//! process sandbox remains a separate future boundary.

#![deny(unsafe_code)]

use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use std::slice;
use std::sync::Mutex;

use libloading::{Library, Symbol};
use mnel_provider_api::{
    ByteView, Digest32, ProviderDescriptorV1, ProviderInferV1, ProviderQueryV1, ProviderResultV1,
    ProviderStatusV1, ABI_VERSION_V1, AUTHORITY_DIAGNOSTIC_ONLY, OUTPUT_ANOMALY_SCORE,
    PROVIDER_STATUS_ABSTAINED, PROVIDER_STATUS_BUDGET_EXCEEDED, PROVIDER_STATUS_COMPLETED,
    PROVIDER_STATUS_INVALID_INPUT, PROVIDER_STATUS_OUT_OF_DISTRIBUTION,
    PROVIDER_STATUS_RUNTIME_ERROR, RESULT_FLAG_OUT_OF_DISTRIBUTION,
    VERDICT_SEMANTICS_NOT_A_VERDICT,
};
use mnel_provider_sdk::{DiagnosticResult, Invocation, LearnedProvider, ProviderError};
use sha2::{Digest as Sha2Digest, Sha256};

const MAX_IDENTIFIER_BYTES: usize = 256;
const MAX_SNAPSHOTS: usize = 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderExpectation {
    pub provider_id: String,
    pub provider_version: String,
    pub declaration_identity: Digest32,
    pub artifact_identity: Digest32,
}

impl ProviderExpectation {
    pub fn validate(&self) -> Result<(), LoaderError> {
        if self.provider_id.trim().is_empty() || self.provider_version.trim().is_empty() {
            return Err(LoaderError::InvalidExpectation(
                "missing provider identity".to_owned(),
            ));
        }
        if self.artifact_identity == Digest32::ZERO {
            return Err(LoaderError::InvalidExpectation(
                "artifact identity must be non-zero".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum LoaderError {
    InvalidExpectation(String),
    Library(String),
    MissingEntrySymbol,
    NullDescriptor,
    UnsupportedAbi(u32),
    MalformedDescriptor(String),
    ProviderIdentityMismatch,
    DeclarationIdentityMismatch,
    ArtifactIdentityMismatch,
    InvalidQuery(String),
    InvalidResult(String),
    ProviderReturnCode(i32),
    ProviderStatus(ProviderStatusV1),
    CallLockPoisoned,
    Io(String),
}

impl Display for LoaderError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidExpectation(reason) => {
                write!(formatter, "invalid provider expectation: {reason}")
            }
            Self::Library(reason) => write!(formatter, "provider library error: {reason}"),
            Self::MissingEntrySymbol => write!(formatter, "missing mnel_provider_entry_v1 symbol"),
            Self::NullDescriptor => write!(formatter, "provider entry returned a null descriptor"),
            Self::UnsupportedAbi(version) => {
                write!(formatter, "unsupported provider ABI: {version}")
            }
            Self::MalformedDescriptor(reason) => {
                write!(formatter, "malformed provider descriptor: {reason}")
            }
            Self::ProviderIdentityMismatch => {
                write!(formatter, "provider descriptor identity mismatch")
            }
            Self::DeclarationIdentityMismatch => {
                write!(formatter, "provider declaration identity mismatch")
            }
            Self::ArtifactIdentityMismatch => {
                write!(formatter, "provider artifact identity mismatch")
            }
            Self::InvalidQuery(reason) => write!(formatter, "invalid provider query: {reason}"),
            Self::InvalidResult(reason) => write!(formatter, "invalid provider result: {reason}"),
            Self::ProviderReturnCode(code) => {
                write!(formatter, "provider returned error code {code}")
            }
            Self::ProviderStatus(status) => write!(formatter, "invalid provider status: {status}"),
            Self::CallLockPoisoned => write!(formatter, "provider call lock is poisoned"),
            Self::Io(reason) => write!(formatter, "provider artifact I/O error: {reason}"),
        }
    }
}

impl std::error::Error for LoaderError {}

pub fn artifact_identity(path: &Path) -> Result<Digest32, LoaderError> {
    let mut file = File::open(path).map_err(|error| LoaderError::Io(error.to_string()))?;
    let length = file
        .seek(std::io::SeekFrom::End(0))
        .map_err(|error| LoaderError::Io(error.to_string()))?;
    if length == 0 {
        return Err(LoaderError::Io("empty provider artifact".to_owned()));
    }
    file.rewind()
        .map_err(|error| LoaderError::Io(error.to_string()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| LoaderError::Io(error.to_string()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(Digest32 {
        bytes: hasher.finalize().into(),
    })
}

pub struct LoadedProvider {
    _library: Library,
    provider_id: String,
    provider_version: String,
    declaration_identity: Digest32,
    artifact_identity: Digest32,
    implementation_context: *mut c_void,
    infer: ProviderInferV1,
    call_lock: Mutex<()>,
    max_output_bytes: usize,
    max_snapshot_bytes: usize,
}

// The ABI contract forbids borrowed provider memory from escaping an invocation.
// Calls are serialized, so a provider context is never accessed concurrently through
// this adapter. The library field keeps all descriptor/function/context addresses valid.
#[allow(unsafe_code)]
unsafe impl Send for LoadedProvider {}
#[allow(unsafe_code)]
unsafe impl Sync for LoadedProvider {}

impl LoadedProvider {
    #[allow(unsafe_code)]
    pub fn load(
        path: &Path,
        expectation: &ProviderExpectation,
        max_output_bytes: usize,
        max_snapshot_bytes: usize,
    ) -> Result<Self, LoaderError> {
        expectation.validate()?;
        if max_output_bytes == 0 || max_snapshot_bytes == 0 {
            return Err(LoaderError::InvalidExpectation(
                "provider byte limits must be positive".to_owned(),
            ));
        }
        let canonical = path
            .canonicalize()
            .map_err(|error| LoaderError::Library(error.to_string()))?;
        let actual_artifact = artifact_identity(&canonical)?;
        if actual_artifact != expectation.artifact_identity {
            return Err(LoaderError::ArtifactIdentityMismatch);
        }
        // All operations involving the foreign library and raw pointers are confined to
        // this constructor and the `infer` method below.
        let library = unsafe { Library::new(&canonical) }
            .map_err(|error| LoaderError::Library(error.to_string()))?;
        let (provider_id, provider_version, declaration_identity, context, infer) = unsafe {
            let entry: Symbol<'_, mnel_provider_api::ProviderEntryV1> = library
                .get(mnel_provider_api::ENTRY_SYMBOL_V1.as_bytes())
                .map_err(|_| LoaderError::MissingEntrySymbol)?;
            let descriptor = entry();
            validate_descriptor(descriptor, expectation)?
        };
        Ok(Self {
            _library: library,
            provider_id,
            provider_version,
            declaration_identity,
            artifact_identity: actual_artifact,
            implementation_context: context,
            infer,
            call_lock: Mutex::new(()),
            max_output_bytes,
            max_snapshot_bytes,
        })
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn provider_version(&self) -> &str {
        &self.provider_version
    }

    pub fn declaration_identity(&self) -> Digest32 {
        self.declaration_identity
    }

    pub fn artifact_identity(&self) -> Digest32 {
        self.artifact_identity
    }

    fn invoke_abi(&self, invocation: &Invocation<'_>) -> Result<DiagnosticResult, LoaderError> {
        let _guard = self
            .call_lock
            .lock()
            .map_err(|_| LoaderError::CallLockPoisoned)?;
        let query = invocation.as_raw();
        validate_query(&query, self.max_snapshot_bytes)?;
        let mut result = HostResultBuffer::new(self.max_output_bytes);
        let return_code = (self.infer)(self.implementation_context, &query, result.as_raw_mut());
        result.finish(return_code)
    }
}

impl LearnedProvider for LoadedProvider {
    fn infer(&self, invocation: &Invocation<'_>) -> Result<DiagnosticResult, ProviderError> {
        self.invoke_abi(invocation).map_err(|error| match error {
            LoaderError::ProviderStatus(PROVIDER_STATUS_ABSTAINED) => ProviderError::Abstained,
            LoaderError::ProviderStatus(PROVIDER_STATUS_INVALID_INPUT) => {
                ProviderError::InvalidBudget
            }
            LoaderError::ProviderStatus(PROVIDER_STATUS_BUDGET_EXCEEDED) => {
                ProviderError::BudgetExceeded
            }
            LoaderError::ProviderStatus(PROVIDER_STATUS_OUT_OF_DISTRIBUTION) => {
                ProviderError::OutOfDistribution
            }
            LoaderError::ProviderStatus(PROVIDER_STATUS_RUNTIME_ERROR)
            | LoaderError::ProviderReturnCode(_)
            | LoaderError::ProviderStatus(_)
            | LoaderError::InvalidResult(_)
            | LoaderError::InvalidQuery(_)
            | LoaderError::CallLockPoisoned
            | LoaderError::Library(_)
            | LoaderError::MissingEntrySymbol
            | LoaderError::NullDescriptor
            | LoaderError::UnsupportedAbi(_)
            | LoaderError::MalformedDescriptor(_)
            | LoaderError::ProviderIdentityMismatch
            | LoaderError::DeclarationIdentityMismatch
            | LoaderError::ArtifactIdentityMismatch
            | LoaderError::InvalidExpectation(_)
            | LoaderError::Io(_) => ProviderError::RuntimeFailure,
        })
    }
}

struct HostResultBuffer {
    storage: Vec<u8>,
    result: ProviderResultV1,
}

impl HostResultBuffer {
    fn new(max_output_bytes: usize) -> Self {
        let mut storage = vec![0_u8; max_output_bytes];
        let result = ProviderResultV1 {
            abi_version: ABI_VERSION_V1,
            status: PROVIDER_STATUS_RUNTIME_ERROR,
            output_kind: OUTPUT_ANOMALY_SCORE,
            calibration_band: 0,
            scalar_value: 0.0,
            flags: 0,
            observation_payload: mnel_provider_api::MutableByteBuffer {
                data: storage.as_mut_ptr(),
                capacity: storage.len(),
                len: 0,
            },
            authority: AUTHORITY_DIAGNOSTIC_ONLY,
            verdict_semantics: VERDICT_SEMANTICS_NOT_A_VERDICT,
        };
        Self { storage, result }
    }

    fn as_raw_mut(&mut self) -> &mut ProviderResultV1 {
        &mut self.result
    }

    fn finish(self, return_code: i32) -> Result<DiagnosticResult, LoaderError> {
        if return_code != 0 {
            return Err(LoaderError::ProviderReturnCode(return_code));
        }
        if self.result.abi_version != ABI_VERSION_V1
            || self.result.authority != AUTHORITY_DIAGNOSTIC_ONLY
            || self.result.verdict_semantics != VERDICT_SEMANTICS_NOT_A_VERDICT
        {
            return Err(LoaderError::InvalidResult(
                "provider changed ABI or diagnostic authority fields".to_owned(),
            ));
        }
        let buffer = self.result.observation_payload;
        if !std::ptr::eq(buffer.data.cast_const(), self.storage.as_ptr())
            || buffer.capacity != self.storage.len()
            || buffer.len > buffer.capacity
        {
            return Err(LoaderError::InvalidResult(
                "provider returned an invalid host buffer pointer or length".to_owned(),
            ));
        }
        let payload = self.storage[..buffer.len].to_vec();
        if !(1..=7).contains(&self.result.output_kind) {
            return Err(LoaderError::InvalidResult(
                "provider returned an unknown output kind".to_owned(),
            ));
        }
        let result = match self.result.status {
            PROVIDER_STATUS_COMPLETED => DiagnosticResult {
                output_kind: self.result.output_kind,
                value: self.result.scalar_value,
                calibration_band: self.result.calibration_band,
                out_of_distribution: self.result.flags & RESULT_FLAG_OUT_OF_DISTRIBUTION != 0,
                payload,
            },
            status => return Err(LoaderError::ProviderStatus(status)),
        };
        result.validate().map_err(|error| match error {
            ProviderError::NonFiniteResult => {
                LoaderError::InvalidResult("non-finite scalar".to_owned())
            }
            _ => LoaderError::InvalidResult("provider result validation failed".to_owned()),
        })
    }
}

#[allow(unsafe_code)]
unsafe fn validate_descriptor(
    pointer: *const ProviderDescriptorV1,
    expectation: &ProviderExpectation,
) -> Result<(String, String, Digest32, *mut c_void, ProviderInferV1), LoaderError> {
    if pointer.is_null() {
        return Err(LoaderError::NullDescriptor);
    }
    let descriptor = &*pointer;
    if descriptor.abi_version != ABI_VERSION_V1 {
        return Err(LoaderError::UnsupportedAbi(descriptor.abi_version));
    }
    if descriptor.reserved != 0 {
        return Err(LoaderError::MalformedDescriptor(
            "reserved descriptor field is non-zero".to_owned(),
        ));
    }
    let provider_id = view_to_string(descriptor.provider_id, "provider_id")?;
    let provider_version = view_to_string(descriptor.provider_version, "provider_version")?;
    if provider_id != expectation.provider_id || provider_version != expectation.provider_version {
        return Err(LoaderError::ProviderIdentityMismatch);
    }
    if descriptor.declaration_identity != expectation.declaration_identity {
        return Err(LoaderError::DeclarationIdentityMismatch);
    }
    let infer = descriptor
        .infer
        .ok_or_else(|| LoaderError::MalformedDescriptor("missing infer function".to_owned()))?;
    Ok((
        provider_id,
        provider_version,
        descriptor.declaration_identity,
        descriptor.implementation_context,
        infer,
    ))
}

#[allow(unsafe_code)]
unsafe fn view_to_string(view: ByteView, label: &str) -> Result<String, LoaderError> {
    if view.len == 0 || view.len > MAX_IDENTIFIER_BYTES || view.data.is_null() {
        return Err(LoaderError::MalformedDescriptor(format!(
            "{label} has an invalid pointer or length"
        )));
    }
    let bytes = slice::from_raw_parts(view.data, view.len);
    let value = std::str::from_utf8(bytes)
        .map_err(|_| LoaderError::MalformedDescriptor(format!("{label} is not UTF-8")))?;
    if value.trim().is_empty() {
        return Err(LoaderError::MalformedDescriptor(format!(
            "{label} is empty"
        )));
    }
    Ok(value.to_owned())
}

#[allow(unsafe_code)]
fn validate_query(query: &ProviderQueryV1, max_snapshot_bytes: usize) -> Result<(), LoaderError> {
    if query.abi_version != ABI_VERSION_V1 {
        return Err(LoaderError::InvalidQuery(
            "unsupported ABI version".to_owned(),
        ));
    }
    if query.reserved != 0 {
        return Err(LoaderError::InvalidQuery(
            "reserved query field is non-zero".to_owned(),
        ));
    }
    if query.budget.wall_time_ns == 0
        || query.budget.operation_limit == 0
        || query.budget.memory_bytes == 0
    {
        return Err(LoaderError::InvalidQuery("zero resource budget".to_owned()));
    }
    if query.snapshot_count == 0
        || query.snapshot_count > MAX_SNAPSHOTS
        || query.snapshots.is_null()
    {
        return Err(LoaderError::InvalidQuery(
            "invalid snapshot pointer or count".to_owned(),
        ));
    }
    let snapshots = unsafe { slice::from_raw_parts(query.snapshots, query.snapshot_count) };
    for snapshot in snapshots {
        if snapshot.reserved != 0 {
            return Err(LoaderError::InvalidQuery(
                "reserved snapshot field is non-zero".to_owned(),
            ));
        }
        if snapshot.payload.len == 0
            || snapshot.payload.len > max_snapshot_bytes
            || snapshot.payload.data.is_null()
        {
            return Err(LoaderError::InvalidQuery(
                "invalid snapshot payload view".to_owned(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(unsafe_code)]
    fn null_descriptor_is_rejected_without_dereference() {
        let expectation = ProviderExpectation {
            provider_id: "fixture".to_owned(),
            provider_version: "0.1".to_owned(),
            declaration_identity: Digest32 { bytes: [1; 32] },
            artifact_identity: Digest32 { bytes: [2; 32] },
        };
        let result = unsafe { validate_descriptor(std::ptr::null(), &expectation) };
        assert!(matches!(result, Err(LoaderError::NullDescriptor)));
    }

    #[test]
    fn invalid_result_pointer_is_rejected_before_copy() {
        let mut buffer = HostResultBuffer::new(8);
        buffer.result.observation_payload.data = 1 as *mut u8;
        assert!(matches!(
            buffer.finish(0),
            Err(LoaderError::InvalidResult(_))
        ));
    }
}
