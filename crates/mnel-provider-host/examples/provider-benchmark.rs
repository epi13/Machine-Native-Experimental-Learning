use std::sync::Arc;
use std::time::Instant;

use mnel_provider_api::{Digest32, ABI_VERSION_V1};
use mnel_provider_classical::HiddenMarkovProvider;
use mnel_provider_host::placement::{PlacementCapabilities, PlacementPolicy};
use mnel_provider_host::{
    CachedSnapshot, ExecutionTier, ImplementationLanguage, ProviderHost, ProviderManifest,
};
use mnel_provider_sdk::{InvocationIdentity, ResourceBudget};

fn digest(value: u8) -> Digest32 {
    Digest32 { bytes: [value; 32] }
}

fn main() {
    let manifest = ProviderManifest {
        provider_id: "state.hidden-markov-model".to_owned(),
        provider_version: "0.1.0".to_owned(),
        declaration_identity: digest(1),
        artifact_identity: digest(2),
        language: ImplementationLanguage::Rust,
        tier: ExecutionTier::NativeTrusted,
        abi_version: ABI_VERSION_V1,
        persistent_host: true,
        language_exception: None,
        placement_policy: PlacementPolicy::default(),
        placement_capabilities: PlacementCapabilities::default(),
    };
    let mut host = ProviderHost::new(3, 1024);
    let cold_started = Instant::now();
    if let Err(error) = host.admit(manifest, Arc::new(HiddenMarkovProvider::new())) {
        panic!("benchmark provider admission should succeed: {error:?}");
    }
    let cold_admission_ns = cold_started.elapsed().as_nanos();
    let snapshot_id = digest(9);
    host.register_snapshot(CachedSnapshot {
        identity: snapshot_id,
        feature_extractor_identity: digest(10),
        payload: Arc::from([0_u8, 0, 1, 1, 2, 2, 3, 3]),
    });
    let identities = InvocationIdentity {
        declaration: digest(1),
        model: digest(2),
        calibration: digest(3),
        query: digest(4),
    };
    let budget = ResourceBudget {
        wall_time_ns: 1_000_000,
        operation_limit: 100,
        memory_bytes: 1024,
    };
    let mut samples = Vec::new();
    let mut process_rss_bytes = None;
    for _ in 0..32 {
        let result = match host.invoke(
            "state.hidden-markov-model",
            identities,
            &[snapshot_id],
            budget,
        ) {
            Ok(result) => result,
            Err(error) => panic!("benchmark invocation should succeed: {error:?}"),
        };
        samples.push(result.measurement.elapsed_ns);
        process_rss_bytes = result.measurement.process_rss_bytes;
    }
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
    let p99 = samples[(samples.len() * 99 / 100).min(samples.len() - 1)];
    println!(
        "{{\"provider\":\"state.hidden-markov-model\",\"cold_admission_ns\":{cold_admission_ns},\"warm_p50_ns\":{p50},\"warm_p95_ns\":{p95},\"warm_p99_ns\":{p99},\"samples\":{},\"snapshot_reuse\":true,\"bytes_copied_per_invocation\":0,\"process_rss_bytes\":{rss},\"execution_mode\":\"cpu\",\"precision\":\"float32\",\"placement_reason\":\"native Rust CPU backend; no accelerator adapter attached\"}}",
        samples.len(),
        rss = process_rss_bytes.map_or_else(|| "null".to_owned(), |value| value.to_string())
    );
}
