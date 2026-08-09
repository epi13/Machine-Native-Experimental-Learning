# Threat model

## Assets

- evaluator, gate, partition, and resource-policy identities;
- experiment and candidate lineage;
- hidden transfer and future-final material;
- raw observations and artifact identities;
- failed, rejected, neutral, abstaining, and `UNKNOWN` experience;
- accepted RAVEL candidate identity and rollback target.

## Primary threats

### Self-poisoning

An incorrect attribution becomes a principle and biases future retrieval and experiment
selection. Preserve competing explanations, counterexamples, transfer gates, and replay
of foundational evidence.

### Verifier monoculture

Investigators and probes share the same mistaken abstraction. Use independently written
probes, mutation tests, brute-force oracles where possible, alternate providers, and
adversarial fixtures.

### Experimental leakage

Selection, transfer, or future-final records enter same-candidate repair context. Enforce
visibility labels at retrieval and dispatch time, not only by directory convention.

### Productive-looking stagnation

Workers generate many near-duplicate candidates. Measure duplicate rate, new failure
coverage, accepted improvement per experiment, hypothesis diversity, and time since the
last new attribution. Halt on stagnation.

### Evaluator gaming

A candidate optimizes a proxy or learns test-specific artifacts. Use held-out providers,
shuffled mappings, mutation tests, and complete-field identity checks.

### Concurrency and lineage corruption

Parallel workers race or overwrite a shared parent. Require append-only results,
content-addressed parent and snapshot identities, unique transactions, and atomic
selection.

### Runtime drift

Model, quantization, runtime, prompt, tool schema, compiler, or context-packing changes
silently change behavior. Bind all of them in experiment records.

### False offload claims

An accelerator may be discoverable while kernels or a requested dtype fail, and a runtime
may accept an offload option without moving weights as intended. Require a real execution
probe before CUDA placement, keep explicit choices fail-closed, record reserve/cap/workspace
math, and mark sequential offload verified only from completed inference plus observed hooks
and parameter residency.

Snapshot reuse is also identity-gated. The current snapshot producers include source,
dependency, extractor, producer, schema, and payload identities in the content identity;
material dependency changes therefore invalidate reuse rather than silently transferring
stale diagnostic context.

The 0.3 diagnostic lifecycle fails closed on malformed snapshot bytes, incompatible
verifiers, unavailable preconditions, malformed verifier output, budget exhaustion, and
repeated verifier errors. A verifier may be quarantined for runtime reliability without
being treated as truthful. Mutation operators are a fixed registered set; arbitrary
callbacks and in-place authoritative snapshot mutation are not accepted. Learned
provider observations and verifier witnesses remain distinct diagnostic records, and
neither can authorize conformance or promotion.

### Apparent independence

Multiple local machines run the same operator-controlled stack. This is replication,
not independent evaluation or protected custody.

## Current residual risks

The foundation now validates and loads identified native provider libraries through a
small Rust boundary, but it does not provide process isolation for that native code.
Network enforcement, cgroups, hardware attestation, authenticated Fabric transport,
protected custody, and immutable remote verifier nodes remain roadmap requirements before
unattended operation on untrusted workloads.
