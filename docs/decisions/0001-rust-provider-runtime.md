# ADR 0001: Rust-first learned micro-provider runtime

- **Status:** Accepted
- **Date:** 2026-08-04
- **Scope:** MNEL learned micro-providers and the MNCS project-family runtime boundary

## Context

Learned micro-providers are intended to be small, frequently invoked diagnostic models.
Their usefulness depends on low warm latency, predictable resource use, cheap reuse of
Forge-produced snapshots, and strict preservation of the `diagnostic-only` authority
boundary. A provider invocation also has to bind its declaration, model artifact,
feature extractor, calibration, query, snapshots, and resource budget.

Python is the correct research and orchestration environment for training, calibration,
data preparation, and architecture experiments. It is not the default per-invocation
hot path because interpreter startup, serialization, garbage collection, process
management, and repeated model loading can dominate the arithmetic of a micro-provider.

C and C++ can provide maximum control but make memory safety, concurrency, dependency
isolation, and provider lifecycle enforcement more difficult. A Rust-only ABI would tie
the project to compiler-specific Rust ABI details and make specialized kernels or future
third-party providers unnecessarily difficult to integrate.

## Decision

MNEL adopts the following implementation policy:

1. **Rust is the reference and default production language** for the persistent provider
   host, provider SDK, runtime policy, snapshot cache, dispatch, budget enforcement, and
   CPU-first learned micro-providers.
2. **Python remains the research, training, calibration, export, admission-study, and
   high-level MNEL orchestration language.** Python is not admitted into the trusted
   native per-invocation hot path.
3. **The stable cross-language boundary is a versioned C ABI**, beginning with
   `mnel-provider-c-abi/1` and the symbol `mnel_provider_entry_v1`.
4. **Native providers are persistent and weight-resident.** Starting a process or loading
   weights for each invocation is forbidden for admitted providers.
5. **Hot-path snapshots use identity-bound compact binary views.** Canonical JSON remains
   appropriate for durable records, manifests, and ledgers, but not as the required
   internal inference representation.
6. **Rust is required for the `native-trusted` tier by default.** C, C++, Zig, or another
   native language requires an explicit exception containing benchmark evidence and a
   threat-review identity. Specialized GPU or vendor kernels may qualify.
7. **WASM is the quarantine and portability tier** for experimental, third-party, or
   less-trusted providers when its isolation benefit exceeds its overhead.
8. **Learned provider output remains diagnostic-only.** The ABI and manifest vocabulary
   contain no evaluator verdict, conformance, acceptance, or promotion authority.

The concise policy is:

> Train and study in Python; host, dispatch, and implement the default hot path in Rust;
> interoperate through a versioned C ABI; admit exceptions only through evidence.

## Enforcement

This decision is enforced by repository artifacts rather than prose alone:

- `Cargo.toml` defines the Rust provider workspace.
- `mnel-provider-api` defines allocation-neutral ABI vocabulary.
- `mnel-provider-sdk` provides a safe Rust authoring surface.
- `mnel-provider-host` encodes native-language admission policy and snapshot reuse.
- `mnel-provider-host::placement` mirrors the backend-neutral CPU/CUDA/offload policy;
  physical accelerator adapters remain outside the trusted ABI boundary.
- `include/mnel_provider_v1.h` is the language-neutral ABI header.
- `mnel-provider-loader` is the reviewed native-trusted dynamic-library boundary. It
  hashes the admitted artifact, validates v1 descriptors and pointer/length metadata,
  copies results into host-owned memory, serializes calls, and preserves diagnostic-only
  semantics before handing a provider to the existing host.
- `ProviderRuntimeManifest` mirrors the admission contract in the Python control plane.
- `learned-provider-runtime-manifest.schema.json` makes the durable manifest testable.
- CI runs Rust formatting, linting, and tests alongside the Python suite.

The first executable native baseline is a deterministic Rust HMM diagnostic provider. Its
host integration demonstrates persistent admission, identity-bound snapshot reuse,
bounded output normalization, timing measurements, and quarantine without changing the
v1 ABI. Sequential CPU offload is an optional external/backend capability, not a reason to
replace the Rust host with a Python daemon.

A loader may not weaken these requirements. It must reject unsupported ABI
versions, missing identities, unbounded queries, process-per-invocation providers,
invalid tier/language combinations, or attempts to grant learned output evaluator
semantics.

## Performance requirements

Provider studies must report at least:

- snapshot construction time separately from inference time;
- warm p50, p95, and p99 invocation latency;
- cold admission and weight-load time;
- bytes copied per invocation;
- peak and resident memory;
- batching behavior and queue delay;
- useful confirmed Forge probes per unit of compute;
- abstention and out-of-distribution behavior; and
- comparison against deterministic or classical baselines under equal budgets.

A faster model that requires materially more snapshot construction or copying is not
presumed to be the faster provider system.

## Consequences

### Positive

- Native performance with stronger memory and concurrency safety than a C/C++ default.
- One provider contract across Rust, specialized native kernels, WASM, and future hosts.
- Lower startup and serialization overhead through a persistent resident runtime.
- A testable exception process rather than informal language drift.
- Alignment with the existing Rust direction in MNCS language and validator projects.

### Costs

- The repository becomes a mixed Python/Rust project.
- Training exports and runtime artifacts need explicit compatibility testing.
- FFI versioning and buffer ownership require careful discipline.
- Some model runtimes may still require C++ or vendor libraries behind the ABI.

## Non-goals

This decision does not select a neural inference engine, serialization library, GPU
stack, dynamic loader, or model format. It does not claim that every Rust implementation
is faster than every C++ implementation. It establishes the default ownership and
contract boundary so those choices can be measured without changing MNEL authority.

## Reconsideration

The decision may be revisited only with repository-recorded evidence showing that a
replacement preserves the ABI and authority guarantees while materially improving
cost-per-useful-probe across representative providers and hardware. Convenience alone
is not sufficient.
