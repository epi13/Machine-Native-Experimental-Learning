//! Deterministic cdylib fixture. Query identity byte zero selects bounded test behavior:
//! 1 success, 2 oversized length, 3 provider runtime error, and 4 invalid output pointer.

#![allow(unsafe_code)]

use core::ptr;

use mnel_provider_api::{
    ByteView, Digest32, ProviderDescriptorV1, ProviderQueryV1, ProviderResultV1, ABI_VERSION_V1,
    AUTHORITY_DIAGNOSTIC_ONLY, OUTPUT_ANOMALY_SCORE, PROVIDER_STATUS_COMPLETED,
    PROVIDER_STATUS_RUNTIME_ERROR, VERDICT_SEMANTICS_NOT_A_VERDICT,
};

static PROVIDER_ID: &[u8] = b"fixture.dynamic-provider";
static PROVIDER_VERSION: &[u8] = b"0.1.0";
static DECLARATION: Digest32 = Digest32 { bytes: [9; 32] };

static mut DESCRIPTOR: ProviderDescriptorV1 = ProviderDescriptorV1 {
    abi_version: ABI_VERSION_V1,
    reserved: 0,
    provider_id: ByteView {
        data: PROVIDER_ID.as_ptr(),
        len: PROVIDER_ID.len(),
    },
    provider_version: ByteView {
        data: PROVIDER_VERSION.as_ptr(),
        len: PROVIDER_VERSION.len(),
    },
    declaration_identity: DECLARATION,
    implementation_context: ptr::null_mut(),
    infer: Some(infer),
};

#[no_mangle]
pub extern "C" fn mnel_provider_entry_v1() -> *const ProviderDescriptorV1 {
    &raw const DESCRIPTOR
}

extern "C" fn infer(
    _context: *mut core::ffi::c_void,
    query: *const ProviderQueryV1,
    result: *mut ProviderResultV1,
) -> i32 {
    // This fixture is intentionally a trusted native test artifact. The loader tests
    // malformed output metadata without asking the fixture to dereference invalid memory.
    unsafe {
        if query.is_null() || result.is_null() {
            return -1;
        }
        let mode = (*query).query_identity.bytes[0];
        (*result).abi_version = ABI_VERSION_V1;
        (*result).authority = AUTHORITY_DIAGNOSTIC_ONLY;
        (*result).verdict_semantics = VERDICT_SEMANTICS_NOT_A_VERDICT;
        (*result).output_kind = OUTPUT_ANOMALY_SCORE;
        (*result).scalar_value = 0.75;
        (*result).calibration_band = 1;
        (*result).flags = 0;
        if mode == 3 {
            (*result).status = PROVIDER_STATUS_RUNTIME_ERROR;
            return -7;
        }
        if mode == 4 {
            (*result).status = PROVIDER_STATUS_COMPLETED;
            (*result).observation_payload.data = 1 as *mut u8;
            (*result).observation_payload.len = 1;
            return 0;
        }
        (*result).status = PROVIDER_STATUS_COMPLETED;
        if mode == 2 {
            (*result).observation_payload.len = (*result).observation_payload.capacity + 1;
            return 0;
        }
        let payload = b"fixture-observation";
        if (*result).observation_payload.capacity < payload.len() {
            (*result).status = PROVIDER_STATUS_RUNTIME_ERROR;
            return -2;
        }
        ptr::copy_nonoverlapping(
            payload.as_ptr(),
            (*result).observation_payload.data,
            payload.len(),
        );
        (*result).observation_payload.len = payload.len();
        0
    }
}
