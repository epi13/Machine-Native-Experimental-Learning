# Learned micro-provider runtime

This document defines the implementation and execution contract for the learned
micro-provider registry. The architecture decision is recorded in
[ADR 0001](decisions/0001-rust-provider-runtime.md).

## Runtime planes

```text
Python research and training
        |
        | export identified model, calibration, and feature contracts
        v
Python MNEL control plane
        |
        | validate declaration + runtime manifest + budget
        v
persistent Rust provider host
        |
        | identity-bound borrowed snapshot views
        v
Rust provider / native kernel / quarantined WASM provider
        |
        | diagnostic value, calibration, OOD state, bounded payload
        v
MNEL diagnostic observation
        |
        v
investigator proposes a bounded Forge question
```

The provider host does not evaluate MNEL gates and cannot promote an observation into a
verdict. Its job is admission, dispatch, bounded execution, normalization, and
measurement.

## Placement is separate from provider lifetime

The runtime manifest carries two different facts:

1. `weight_residency: resident-on-admission` means an admitted provider remains loaded
   and reusable; weights are not reloaded for each invocation.
2. `placement` describes where execution and physical weight storage should occur.

The policy vocabulary is:

| Field | Values |
|---|---|
| execution device | `auto`, `cpu`, `cuda` |
| offload | `auto`, `none`, `sequential-cpu` |
| precision | `auto`, `float32`, `float16`, `bfloat16` |

Sequential CPU offload is therefore compatible with persistent admission. Weights remain
in system RAM and supported modules are temporarily moved to CUDA for execution. This
reduces persistent VRAM at the cost of host memory and transfer time; it is not a
process-per-invocation reload strategy.

The Python control-plane implementation is in `mnel.placement`. The Rust host mirrors the
policy vocabulary in `mnel-provider-host::placement`. Both are backend-neutral. The
optional `mnel.torch_runtime` adapter performs actual Torch probes and applies a decision
through Accelerate when sequential offload is selected.

AUTO placement requires a real accelerator execution probe, measures currently free VRAM,
subtracts GPU reserve, applies an optional cap, and includes model plus workspace estimates.
It selects full CUDA when it fits, sequential offload when supported, and CPU otherwise.
Only AUTO may recover from a bounded CUDA OOM sequence; explicit operator choices fail
instead of silently changing execution mode.

## Language and execution tiers

| Tier | Default language | Purpose | Admission rule |
|---|---|---|---|
| `native-trusted` | Rust | Warm production hot path | Non-Rust requires benchmark and threat-review exception |
| `wasm-quarantined` | WASM | Portable or less-trusted provider | Capability-limited and measured against native overhead |
| `external-experimental` | Python or other | Training, prototyping, admission studies | Never treated as the production latency baseline |

A provider's training language does not determine its runtime tier. A model may be
trained in Python and exported to a Rust-hosted native runtime.

## Versioned C ABI

The v1 contract is defined twice from the same conceptual vocabulary:

- Rust: `crates/mnel-provider-api/src/lib.rs`
- C: `include/mnel_provider_v1.h`

The public entry symbol is:

```c
const mnel_provider_descriptor_v1 *mnel_provider_entry_v1(void);
```

The descriptor remains valid for the lifetime of the loaded provider. The host owns
query memory and the result payload buffer. Providers must not retain borrowed snapshot
pointers after returning.

The ABI binds every invocation to:

- provider declaration identity;
- model or weight identity;
- calibration identity;
- query identity;
- one or more snapshot identities;
- feature-extractor identities; and
- wall-time, operation, and memory budgets.

## Result semantics

A provider may complete, abstain, report invalid input, exceed a budget, identify an
out-of-distribution input, or fail at runtime. Those are runtime statuses, not evaluator
verdicts.

The normalized result contains:

- one declared output kind;
- a finite scalar when applicable;
- a calibration-band identifier;
- flags such as out-of-distribution or truncated payload; and
- an optional bounded host-owned payload.

Every result is stamped with `diagnostic-only` authority and `not-a-verdict` semantics.
No ABI field may represent PASS, FAIL, conformance, causal truth, acceptance, or
promotion.

## Persistent host requirements

An admitted host must:

1. validate the runtime manifest against the canonical provider declaration;
2. verify all material artifact identities before admission;
3. load weights once and keep them resident while admitted;
4. reuse immutable snapshot payloads across compatible providers;
5. avoid JSON parsing and process startup in the normal invocation path;
6. enforce time, operation, memory, and output-payload limits;
7. preserve provider-specific observations rather than voting them into one answer;
8. record warm and cold performance separately; and
9. unload or quarantine a provider after integrity, calibration, or budget failures.

The Rust host now provides a process-local lifecycle for admitted `LearnedProvider` trait
objects, reusable snapshot storage, bounded result normalization, timing/copy measurements,
clean unload, and deterministic failure quarantine. `mnel-provider-loader` adds the
native-trusted dynamic-library increment: it hashes the artifact, resolves
`mnel_provider_entry_v1`, validates descriptor identities and ABI version, checks query and
result pointer/length metadata, copies output into host-owned memory, serializes calls, and
unloads the library with the provider handle. Native code is not sandboxed; OS isolation,
and a production accelerator backend remain future work. The C ABI v1 remains unchanged.

## Snapshot transport

The initial Python snapshot producers construct bounded transition, pair, or tabular
snapshots once. Each immutable payload is binary-friendly and carries producer, source,
dependency, feature-extractor, schema, and payload identities. Compatible deterministic
probes and learned providers can consume the same payload boundary; changing a material
dependency changes the content identity and prevents silent reuse. Forge or another
identified producer can extend this vocabulary to AST, graph, trace, or composite views.

The durable ledger may describe the snapshot with canonical JSON, but the hot path uses
compact binary bytes with explicit schema and feature-extractor identities. Any material
change to source, dependency, extractor, normalization, toolchain, or environment
invalidates reuse unless the dependency envelope proves the snapshot unaffected.

## Native-language exceptions

A non-Rust provider may enter `native-trusted` only when its manifest includes:

- a concrete technical rationale;
- identities for equal-budget benchmark evidence; and
- an identity for the relevant threat and ownership review.

Typical candidates are vendor inference engines, CUDA kernels, or specialized libraries
without an adequate Rust implementation. The exception applies to one identified
artifact and does not establish a general language preference.

## Delivery sequence

1. Freeze and test the v1 manifest and ABI vocabulary.
2. Use the executable Rust HMM reference provider as the classical baseline.
3. Keep dynamic loading and host-owned ABI output enforcement behind the reviewed
   `mnel-provider-loader` boundary and its native cdylib fixtures.
4. Add Forge snapshot producers and reuse measurements.
5. Export one Python-trained neural provider and compare it with the baseline.
6. Add WASM quarantine only after native measurements establish the overhead budget.
7. Integrate Fabric placement after single-host identity and replay behavior is stable.

Each stage must preserve the current diagnostic authority boundary and may terminate in
`UNKNOWN` rather than silently widening capability.
