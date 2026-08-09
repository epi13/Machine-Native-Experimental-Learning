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

The 0.4 study layer adds an explicit visibility gate: development access rejects hidden
transfer and future-final records, clustering/training/retrieval operate on an eligible
view, and transfer prediction is frozen before a separate transfer-evaluator view can
read the held-out result. Same-candidate repair from that result is rejected. Shuffled
attribution and memory ablations create new study identities while preserving their
source records. Deterministic feature grouping is bounded and explicitly labeled as a
heuristic limitation, so it cannot masquerade as semantic understanding. Distillation,
retrieval, calibration, and learned-provider observations remain proposal or diagnostic
records; none includes evaluator verdict authority.

The provider portfolio adds further guards: provider training and calibration accept only
explicit development views; the held-out transfer fixture is exposed through a separate
transfer-evaluator identity after prediction freeze; provider artifacts are hash-bound and
reload-tested; and lifecycle transitions require evidence identities and a policy identity.
Random and diversity routing cannot expand snapshot compatibility, and pairwise
disagreement is not treated as error without an identified reference outcome. Optional
energy readings are recorded as unavailable when no trusted source exists rather than
estimated from latency. A small tabular provider is intentionally bounded and CPU-only;
native export remains outside this iteration rather than weakening the ABI boundary.

### Apparent independence

Multiple local machines run the same operator-controlled stack. This is replication,
not independent evaluation or protected custody.

### MNCS-family boundaries

The family integration layer pins public sibling shapes and fails closed on unsupported
protocol drift. A sibling checkout, CLI, or binary is not trusted merely because it is
available: live commit identity is reported, provider identity and artifact identities are
bound, and external records remain opaque observations. Fabric duplicate/replay behavior is
preserved rather than converted into success. Receipt `claim_boundary` fields remain
unasserted. Commons records are inert and unpublished; RAVEL material is proposal context
only. Forge configuration and local sibling checkouts remain operator-controlled inputs and
must be reviewed for substitution or drift.

## Current residual risks

The foundation now validates and loads identified native provider libraries through a
small Rust boundary, but it does not provide process isolation for that native code.
Network enforcement, cgroups, hardware attestation, authenticated Fabric transport,
protected custody, and immutable remote verifier nodes remain roadmap requirements before
unattended operation on untrusted workloads.
