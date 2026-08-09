//! A shared library without the versioned MNEL entry symbol.

#![allow(unsafe_code)]

#[no_mangle]
pub extern "C" fn unrelated_fixture_symbol() {}
