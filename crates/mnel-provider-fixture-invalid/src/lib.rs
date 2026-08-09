//! A deterministic descriptor that must be rejected before invocation.

#![allow(unsafe_code)]

use mnel_provider_api::{ByteView, Digest32, ProviderDescriptorV1, ABI_VERSION_V1};

static ID: &[u8] = b"fixture.invalid";
static VERSION: &[u8] = b"0.1.0";
static mut DESCRIPTOR: ProviderDescriptorV1 = ProviderDescriptorV1 {
    abi_version: ABI_VERSION_V1 + 1,
    reserved: 0,
    provider_id: ByteView {
        data: ID.as_ptr(),
        len: ID.len(),
    },
    provider_version: ByteView {
        data: VERSION.as_ptr(),
        len: VERSION.len(),
    },
    declaration_identity: Digest32 { bytes: [8; 32] },
    implementation_context: core::ptr::null_mut(),
    infer: None,
};

#[no_mangle]
pub extern "C" fn mnel_provider_entry_v1() -> *const ProviderDescriptorV1 {
    &raw const DESCRIPTOR
}
