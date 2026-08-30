# MNCS-native training pipeline

## Status

Architectural direction for MNEL and the MNCS family.

## Objective

MNEL should move toward a training pipeline whose semantics are owned by MNCS rather
than by an external host language. The goal is not to reimplement every numerical
kernel, accelerator driver, or mature ML library immediately. The goal is to make MNCS
the source of truth for the computation that turns evidence into a trained model.

Today, a practical learned-provider path can still look like:

```text
MNCS evidence
    -> dataset / snapshot
    -> Python training code
    -> PyTorch / other runtime
    -> model artifact
    -> MNCS evaluation and lineage
```

That is useful as a transition path, but it creates an evidence discontinuity. MNCS can
identify the inputs and resulting artifact while the most consequential transformation
is still described and controlled elsewhere.

The intended long-term architecture is:

```text
MNCS evidence
    -> MNCS dataset construction
    -> MNCS transforms / sampling / batching
    -> MNCS model and training specification
    -> MNCS-controlled optimization and checkpoint policy
    -> backend / accelerator execution
    -> MNCS evaluation and promotion policy
    -> MNCS model artifact
    -> MNCS lineage
```

External runtimes may remain execution targets where they are useful. They should not be
the semantic authority for the training run.

## Why this matters

A trained model is not explained only by its final weights. Its behavior depends on the
full transformation history that produced those weights, including:

- source observations and dataset identity;
- inclusion, exclusion, filtering, labeling, and balancing decisions;
- preprocessing and feature extraction;
- sampling, shuffling, batching, and random seeds;
- model architecture and initialization;
- objective and loss definitions;
- optimizer and learning-rate policy;
- numerical precision and device placement;
- checkpoint selection and stopping rules;
- evaluation suites and thresholds;
- quantization, pruning, or distillation;
- promotion, quarantine, rollback, and retirement decisions.

If those semantics live primarily in an external script, MNEL can attach provenance to
the result but cannot naturally represent the complete training computation in the same
machine-native evidence system used elsewhere in the MNCS family.

For MNEL micro-models this is especially important. Small, task-specific providers are
intended to be narrow, inspectable, reproducible components. A family of tiny models
should not become a family of tiny opaque training islands.

## Core principle

> Training is an evidence-producing computation, not an opaque artifact-generation
> step.

A training run should eventually be a first-class MNCS computation that produces both a
model artifact and the evidence needed to explain and reproduce that artifact.

Conceptually:

```text
model fabric.patch-evaluator@7

sources:
    dataset forge.patch-history@<digest>
    labels forge.verdicts@<digest>

transform:
    tokenize(...)
    normalize(...)
    balance(...)

architecture:
    micro-transformer(...)

train:
    optimizer adamw(...)
    seed ...
    schedule ...
    stopping-rule ...

evaluate:
    regression-suite ...
    false-positive-rate ...
    calibration ...

promote-if:
    hard gates ...

produces:
    model artifact
    training trace
    evaluation evidence
    lineage edges
```

The syntax above is illustrative only. This document establishes semantics and ownership,
not a final language syntax.

## Relationship with `mncs-lineage`

MNEL should design training records so `mncs-lineage` can traverse a deployed learned
component all the way back to the evidence that taught it.

The desired chain is approximately:

```text
application or service
    -> selected micro-model
    -> quantized / packaged artifact
    -> parent model artifact
    -> training run
    -> MNCS training graph
    -> dataset snapshot
    -> source observations
    -> original execution evidence
```

Model training should therefore become a first-class lineage event rather than a JSON
manifest attached after training completes.

Lineage should preserve at minimum:

- exact input dataset and observation identities;
- transform identities and parameters;
- training-graph identity;
- model architecture identity;
- optimizer and schedule identity;
- deterministic seeds where applicable;
- backend/runtime and accelerator identity;
- parent-model identity for fine-tuning or distillation;
- checkpoint ancestry;
- quantization/pruning/distillation transformations;
- evaluation evidence used for admission;
- final artifact digest and packaging identity.

This does not make a learned model an evaluator authority. Existing MNEL rules around
diagnostic-only learned providers, hard gates, quarantine, and independent evaluation
remain intact.

## Standard-library opportunity

A major reason to pursue this work is that training stresses capabilities that are useful
far beyond AI/ML. Specialized MNEL needs should be promoted into the MNCS language or
standard library whenever the primitive is independently useful.

Likely reusable areas include:

### Numeric and array computation

- vectors and matrices;
- dense and sparse arrays;
- tensor-like multidimensional storage;
- reductions and scans;
- broadcasting;
- linear algebra;
- statistics;
- SIMD-friendly operations.

These are also useful for simulation, geometry, graphics, signal processing, control
systems, scientific computing, optimization, and finance.

### Data pipelines

- iterators and streams;
- lazy transforms;
- filtering and mapping;
- batching and windowing;
- deterministic shuffling and sampling;
- joins and grouping;
- partitioning and sharding;
- bounded streaming IO.

These primitives also serve ETL, telemetry, logs, databases, compilers, build systems,
and distributed workflows.

### Deterministic randomness

- seeded random streams;
- reproducible sampling;
- distributions;
- independent stream derivation;
- recorded stochastic decisions.

The same machinery is useful for simulation, randomized testing, procedural generation,
and Monte Carlo methods.

### Parallel and heterogeneous execution

- task graphs;
- worker pools;
- parallel map/reduce;
- deterministic reductions where required;
- synchronization primitives;
- device placement;
- host/device transfers;
- execution queues;
- accelerator resource accounting.

Training becomes one demanding consumer of a general MNCS compute model rather than a
special-purpose island.

### Artifact, checkpoint, and state handling

- content-addressed artifacts;
- snapshots;
- versioned schemas;
- transactional writes;
- checkpoint/resume;
- rollback;
- resumable long-running computations.

These belong naturally with the broader evidence and execution architecture.

### Graphs and incremental computation

Training graphs create pressure for reusable DAG primitives, dependency tracking, cached
nodes, invalidation, and incremental recomputation. Those capabilities can also serve the
compiler, Forge, Fabric, build systems, workflows, and reactive applications.

### Metrics and observability

- counters;
- histograms;
- traces;
- resource measurements;
- structured diagnostic events;
- reproducible benchmark records.

These should be general system capabilities rather than private MNEL implementations.

## Layering principle

When MNEL needs a primitive that has obvious value outside learning, prefer promoting it
into the language or standard library rather than hiding it behind an MNEL-specific API.

A possible eventual layering is:

```text
MNCS standard library
    collections
    io
    math
    numeric
    stats
    random
    graph
    stream
    parallel
    compute
    artifact
    evidence
    checkpoint
        |
        v
MNEL
    dataset
    tensor / differentiable numeric surface where still ML-specific
    model
    optimizer
    train
    evaluate
    distill
```

Some components may move downward over time. For example, tensor operations may prove
sufficiently general to belong in the standard library rather than MNEL.

The first one-step distilled specialist implementation intentionally remains in MNEL's
Python/reference-runtime layer. It uses a dependency-free bounded numerical trainer and
identity-bound JSON artifacts while MNCS-native training representation and semantic
closure remain future work. No new MNCS syntax, compiler behavior, backend lowering, or
standard-library primitive is required by this reference slice.

## Backend boundary

MNCS-native training does **not** require MNCS to own every low-level implementation on
day one.

The intended separation is:

```text
MNCS training semantics
        |
        v
MNCS compiler / runtime
        |
        +-> native CPU / SIMD
        +-> CUDA
        +-> ROCm
        +-> LLVM / Cranelift lowering
        +-> C ABI numerical libraries
        +-> external ML runtime adapter, where justified
```

An external runtime is acceptable as a backend implementation detail when MNCS still
owns and records the semantic training graph and can validate the resulting execution.

This is analogous to a compiler targeting an external assembler or system linker: using
an external implementation does not require surrendering ownership of program semantics.

## Micro-model implications

MNEL should prioritize this architecture first on small and task-specific models where a
full conventional ML stack may be unnecessary overhead.

Examples include narrow classifiers, routers, anomaly detectors, discrepancy models,
small sequence models, compact tabular models, and learned heuristics.

A sufficiently small model may eventually need only a compact subset such as:

```text
std.numeric
std.graph
std.compute
std.artifact
std.evidence
mnel.model
mnel.train
```

That creates an opportunity for resident micro-models whose training and execution are
both lightweight, reproducible, and deeply integrated with MNCS evidence.

## Language stress-test value

Building this pipeline should be treated as a deliberate stress test for `mncs-language`
and its standard library.

Training will expose weaknesses in areas such as:

- ownership and lifetimes for large buffers;
- zero-copy and bounded-copy data movement;
- generic numeric types;
- memory layout and alignment;
- SIMD and accelerator execution;
- asynchronous pipelines;
- deterministic concurrency;
- graph execution;
- binary serialization;
- long-running task recovery;
- resource accounting;
- reproducible randomness;
- incremental state;
- error propagation across backend boundaries.

When the MNCS implementation encounters a missing language or standard-library
capability, the preferred response is to improve the shared MNCS foundation rather than
quietly route around the limitation in another language.

## Phased implementation

### Phase 1 — MNCS-owned training manifests and lineage

Keep existing Python/native reference trainers where necessary, but move the canonical
training specification into an MNCS-consumable schema with complete identities for data,
transforms, model configuration, optimization, evaluation, and artifacts.

The external trainer becomes an adapter executing an MNCS-described run.

### Phase 2 — MNCS-native dataset construction and evaluation

Move dataset composition, filtering, transformations, partitioning, evaluation, and hard
promotion criteria into MNCS-facing implementations. These are high-value provenance
surfaces and provide broadly reusable standard-library pressure.

### Phase 3 — MNCS training primitives

Introduce the minimum primitives necessary to express model execution and training:

- numeric/tensor operations;
- model parameter state;
- objective computation;
- gradient or model-specific update mechanisms;
- optimizer state;
- batching;
- checkpointing;
- metric emission.

Do not assume every model requires automatic differentiation. Classical and compact
models should use the smallest suitable mechanism.

### Phase 4 — backend lowering

Lower MNCS training graphs to available CPU, accelerator, and external numerical
backends while preserving semantic identity and execution evidence.

### Phase 5 — native micro-model training

Move selected MNEL micro-models completely off Python where doing so reduces dependency,
context, memory, startup, or provenance costs without weakening correctness.

Use these migrations to discover missing language and standard-library capabilities.

### Phase 6 — full lineage closure

Integrate training-run, checkpoint, transformation, admission, quantization, distillation,
and deployment identities with `mncs-lineage` so deployed model behavior can be traced to
its eligible source evidence.

## Non-goals

This direction does not require the project to:

- rewrite CUDA, ROCm, BLAS, or mature numerical kernels immediately;
- create a general replacement for PyTorch before MNEL can make progress;
- move every existing reference trainer to MNCS in one release;
- give learned models evaluator authority;
- hide nondeterminism or backend differences behind a false reproducibility claim;
- require automatic differentiation for model families that do not need it;
- duplicate general-purpose primitives privately inside MNEL.

## Initial acceptance criteria

The first implementation milestone should be considered successful when:

1. A training run has a canonical MNCS-owned specification independent of the host
   trainer implementation.
2. Dataset, transform, architecture, optimization, backend, checkpoint, evaluation, and
   output-artifact identities are preserved as evidence.
3. Replaying an unchanged deterministic reference run produces either the same artifact
   identity or an explicit machine-readable explanation of permitted nondeterminism.
4. `mncs-lineage` has a stable conceptual contract for consuming training lineage even
   if full cross-repository integration lands separately.
5. At least one reusable primitive discovered during this work is implemented or
   proposed at the MNCS standard-library layer rather than kept MNEL-private.
6. Existing hard-gate and diagnostic-only authority boundaries remain unchanged.

## Architectural rule

MNEL may use external implementations, but it should avoid external semantic ownership.

The desired endpoint is not "all ML code was rewritten in MNCS." The desired endpoint is
that MNCS can explain, reproduce, inspect, and govern the computation that created a
learned component without encountering an avoidable language boundary in the middle of
the evidence chain.
