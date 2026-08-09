use std::path::PathBuf;
use std::sync::Arc;

use mnel_provider_api::Digest32;
use mnel_provider_host::{ExecutionTier, ImplementationLanguage, ProviderHost, ProviderManifest};
use mnel_provider_loader::{artifact_identity, LoadedProvider, ProviderExpectation};
use mnel_provider_sdk::{
    Invocation, InvocationIdentity, LearnedProvider, ResourceBudget, SnapshotRef,
};

fn fixture_path(package: &str) -> PathBuf {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    let stem = package.replace('-', "_");
    let filename = if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    };
    let direct = target.join("debug").join(&filename);
    if direct.is_file() {
        direct
    } else {
        target.join("debug").join("deps").join(filename)
    }
}

fn expectation(path: &std::path::Path, provider_id: &str, declaration: u8) -> ProviderExpectation {
    ProviderExpectation {
        provider_id: provider_id.to_owned(),
        provider_version: "0.1.0".to_owned(),
        declaration_identity: Digest32 {
            bytes: [declaration; 32],
        },
        artifact_identity: ok(artifact_identity(path)),
    }
}

fn ok<T, E: std::fmt::Debug>(result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("unexpected fixture test error: {error:?}"),
    }
}

fn invocation(mode: u8) -> Invocation<'static> {
    let payload: &'static [u8] = Box::leak(vec![1_u8, 2, 3].into_boxed_slice());
    ok(Invocation::new(
        InvocationIdentity {
            declaration: Digest32 { bytes: [9; 32] },
            model: Digest32 { bytes: [2; 32] },
            calibration: Digest32 { bytes: [3; 32] },
            query: Digest32 { bytes: [mode; 32] },
        },
        ResourceBudget {
            wall_time_ns: 1_000_000,
            operation_limit: 100,
            memory_bytes: 1_024,
        },
        vec![SnapshotRef {
            schema_version: 1,
            identity: Digest32 { bytes: [4; 32] },
            feature_extractor_identity: Digest32::ZERO,
            payload,
        }],
    ))
}

#[test]
fn native_fixture_loads_invokes_repeatedly_and_unloads_cleanly() {
    let path = fixture_path("mnel-provider-fixture");
    let provider = ok(LoadedProvider::load(
        &path,
        &expectation(&path, "fixture.dynamic-provider", 9),
        64,
        64,
    ));
    for _ in 0..2 {
        let result = ok(provider.infer(&invocation(1)));
        assert_eq!(result.payload, b"fixture-observation");
    }
    drop(provider);
    assert!(path.is_file());
}

#[test]
fn malformed_abi_and_missing_symbol_are_rejected() {
    let invalid = fixture_path("mnel-provider-fixture-invalid");
    assert!(matches!(
        LoadedProvider::load(
            &invalid,
            &expectation(&invalid, "fixture.invalid", 8),
            64,
            64,
        ),
        Err(mnel_provider_loader::LoaderError::UnsupportedAbi(_))
    ));
    let no_entry = fixture_path("mnel-provider-fixture-no-entry");
    assert!(matches!(
        LoadedProvider::load(
            &no_entry,
            &ProviderExpectation {
                provider_id: "missing".to_owned(),
                provider_version: "0.1.0".to_owned(),
                declaration_identity: Digest32 { bytes: [1; 32] },
                artifact_identity: ok(artifact_identity(&no_entry)),
            },
            64,
            64,
        ),
        Err(mnel_provider_loader::LoaderError::MissingEntrySymbol)
    ));
}

#[test]
fn oversized_and_invalid_output_are_bounded_errors() {
    let path = fixture_path("mnel-provider-fixture");
    let provider = ok(LoadedProvider::load(
        &path,
        &expectation(&path, "fixture.dynamic-provider", 9),
        8,
        64,
    ));
    assert_eq!(
        provider.infer(&invocation(2)),
        Err(mnel_provider_sdk::ProviderError::RuntimeFailure)
    );
    assert_eq!(
        provider.infer(&invocation(4)),
        Err(mnel_provider_sdk::ProviderError::RuntimeFailure)
    );
}

#[test]
fn runtime_error_quarantines_through_the_existing_host() {
    let path = fixture_path("mnel-provider-fixture");
    let provider = ok(LoadedProvider::load(
        &path,
        &expectation(&path, "fixture.dynamic-provider", 9),
        64,
        64,
    ));
    let mut host = ProviderHost::new(1, 64);
    ok(host.admit(
        ProviderManifest {
            provider_id: "fixture.dynamic-provider".to_owned(),
            provider_version: "0.1.0".to_owned(),
            declaration_identity: Digest32 { bytes: [9; 32] },
            artifact_identity: ok(artifact_identity(&path)),
            language: ImplementationLanguage::Rust,
            tier: ExecutionTier::NativeTrusted,
            abi_version: 1,
            persistent_host: true,
            language_exception: None,
            placement_policy: Default::default(),
            placement_capabilities: Default::default(),
        },
        Arc::new(provider),
    ));
    host.register_snapshot(mnel_provider_host::CachedSnapshot {
        identity: Digest32 { bytes: [4; 32] },
        feature_extractor_identity: Digest32::ZERO,
        payload: Arc::from([1_u8, 2, 3]),
    });
    let result = ok(host.invoke(
        "fixture.dynamic-provider",
        InvocationIdentity {
            declaration: Digest32 { bytes: [9; 32] },
            model: Digest32 { bytes: [2; 32] },
            calibration: Digest32 { bytes: [3; 32] },
            query: Digest32 { bytes: [3; 32] },
        },
        &[Digest32 { bytes: [4; 32] }],
        ResourceBudget {
            wall_time_ns: 1_000_000,
            operation_limit: 100,
            memory_bytes: 1_024,
        },
    ));
    assert_eq!(
        result.status,
        mnel_provider_api::PROVIDER_STATUS_RUNTIME_ERROR
    );
    assert!(matches!(
        host.state("fixture.dynamic-provider"),
        Some(mnel_provider_host::ProviderState::Quarantined { .. })
    ));
}
