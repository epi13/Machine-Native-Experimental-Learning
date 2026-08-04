//! Stable, allocation-neutral ABI vocabulary for diagnostic-only learned providers.
//!
//! The ABI intentionally contains no evaluator verdict, conformance, promotion, or
//! acceptance field. Provider output is diagnostic context only.

use core::ffi::c_void;

pub const ABI_VERSION_V1: u32 = 1;
pub const ENTRY_SYMBOL_V1: &str = "mnel_provider_entry_v1";
pub const AUTHORITY_DIAGNOSTIC_ONLY: u32 = 1;
pub const VERDICT_SEMANTICS_NOT_A_VERDICT: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Digest32 {
    pub bytes: [u8; 32],
}

impl Digest32 {
    pub const ZERO: Self = Self { bytes: [0; 32] };
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ByteView {
    pub data: *const u8,
    pub len: usize,
}

impl ByteView {
    pub const EMPTY: Self = Self {
        data: core::ptr::null(),
        len: 0,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MutableByteBuffer {
    pub data: *mut u8,
    pub capacity: usize,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudgetV1 {
    pub wall_time_ns: u64,
    pub operation_limit: u64,
    pub memory_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SnapshotViewV1 {
    pub schema_version: u32,
    pub reserved: u32,
    pub snapshot_identity: Digest32,
    pub feature_extractor_identity: Digest32,
    pub payload: ByteView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProviderQueryV1 {
    pub abi_version: u32,
    pub reserved: u32,
    pub declaration_identity: Digest32,
    pub model_identity: Digest32,
    pub calibration_identity: Digest32,
    pub query_identity: Digest32,
    pub snapshots: *const SnapshotViewV1,
    pub snapshot_count: usize,
    pub budget: ResourceBudgetV1,
}

pub type ProviderStatusV1 = u32;
pub const PROVIDER_STATUS_COMPLETED: ProviderStatusV1 = 0;
pub const PROVIDER_STATUS_ABSTAINED: ProviderStatusV1 = 1;
pub const PROVIDER_STATUS_INVALID_INPUT: ProviderStatusV1 = 2;
pub const PROVIDER_STATUS_BUDGET_EXCEEDED: ProviderStatusV1 = 3;
pub const PROVIDER_STATUS_OUT_OF_DISTRIBUTION: ProviderStatusV1 = 4;
pub const PROVIDER_STATUS_RUNTIME_ERROR: ProviderStatusV1 = 5;

pub type OutputKindV1 = u32;
pub const OUTPUT_LATENT_DISCREPANCY: OutputKindV1 = 1;
pub const OUTPUT_STRUCTURAL_DISCREPANCY: OutputKindV1 = 2;
pub const OUTPUT_ANOMALY_SCORE: OutputKindV1 = 3;
pub const OUTPUT_PAIR_SIMILARITY: OutputKindV1 = 4;
pub const OUTPUT_NEXT_STATE_DISTRIBUTION: OutputKindV1 = 5;
pub const OUTPUT_FEATURE_CONTRIBUTIONS: OutputKindV1 = 6;
pub const OUTPUT_CANDIDATE_RANKING: OutputKindV1 = 7;

pub const RESULT_FLAG_OUT_OF_DISTRIBUTION: u64 = 1 << 0;
pub const RESULT_FLAG_CALIBRATION_REQUIRED: u64 = 1 << 1;
pub const RESULT_FLAG_TRUNCATED_PAYLOAD: u64 = 1 << 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProviderResultV1 {
    pub abi_version: u32,
    pub status: ProviderStatusV1,
    pub output_kind: OutputKindV1,
    pub calibration_band: u32,
    pub scalar_value: f64,
    pub flags: u64,
    pub observation_payload: MutableByteBuffer,
    pub authority: u32,
    pub verdict_semantics: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProviderDescriptorV1 {
    pub abi_version: u32,
    pub reserved: u32,
    pub provider_id: ByteView,
    pub provider_version: ByteView,
    pub declaration_identity: Digest32,
    pub implementation_context: *mut c_void,
    pub infer: Option<ProviderInferV1>,
}

pub type ProviderInferV1 = extern "C" fn(
    context: *mut c_void,
    query: *const ProviderQueryV1,
    result: *mut ProviderResultV1,
) -> i32;

pub type ProviderEntryV1 = extern "C" fn() -> *const ProviderDescriptorV1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_authority_is_fixed_and_distinct() {
        assert_ne!(AUTHORITY_DIAGNOSTIC_ONLY, 0);
        assert_ne!(VERDICT_SEMANTICS_NOT_A_VERDICT, 0);
    }

    #[test]
    fn abi_types_are_c_compatible_and_nonzero_sized() {
        assert!(core::mem::size_of::<ProviderQueryV1>() > 0);
        assert!(core::mem::size_of::<ProviderResultV1>() > 0);
        assert!(core::mem::align_of::<ProviderQueryV1>() >= core::mem::align_of::<u64>());
    }

    #[test]
    fn entry_symbol_is_versioned() {
        assert_eq!(ENTRY_SYMBOL_V1, "mnel_provider_entry_v1");
    }
}
