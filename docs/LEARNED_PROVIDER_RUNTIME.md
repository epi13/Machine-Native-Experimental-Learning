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

The initial Rust host crate establishes admission policy and reusable snapshot storage.
Dynamic loading, operating-system sandboxing, and model-runtime selection remain future
implementation work.

## Snapshot transport

Forge or another identified producer should construct an AST, graph, trace, transition,
pair, tabular, or composite snapshot once. Compatible providers consume borrowed views
of that immutable payload.

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
2. Implement a process-local Rust reference provider for a classical baseline.
3. Add the persistent loader and host-owned output buffer enforcement.
4. Add Forge snapshot producers and reuse measurements.
5. Export one Python-trained neural provider and compare it with the baseline.
6. Add WASM quarantine only after native measurements establish the overhead budget.
7. Integrate Fabric placement after single-host identity and replay behavior is stable.

Each stage must preserve the current diagnostic authority boundary and may terminate in
`UNKNOWN` rather than silently widening capability.
