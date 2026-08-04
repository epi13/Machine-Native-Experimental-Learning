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

### Apparent independence

Multiple local machines run the same operator-controlled stack. This is replication,
not independent evaluation or protected custody.

## Current residual risks

The foundation does not provide process isolation, network enforcement, cgroups,
hardware attestation, authenticated Fabric transport, protected custody, or immutable
remote verifier nodes. Those remain roadmap requirements before unattended operation on
untrusted workloads.
