# ADR 0003: One-step distilled specialists for bounded micro-providers

- **Status:** Accepted
- **Date:** 2026-08-30
- **Scope:** MNEL learned micro-providers and the MNCS-native training/lineage boundary

## Context

MNEL already treats small, task-specific learned providers as bounded cognitive
components. Some useful provider tasks are currently performed by an expensive
iterative procedure: a larger model may reason over a structured snapshot, an analyzer
may make several passes, or an optimizer may search a bounded space. Repeating that
procedure for every similar query can dominate latency and resource use even when the
input representation and constraints strongly narrow the answer.

The one-step idea is to use the iterative procedure as a teacher, retain its candidate
traces and independent evidence, and distill a compact student that emits the bounded
result in one learned forward pass. The student should expose a structured output,
calibrated confidence, and explicit abstention so it can escalate when the one-pass
approximation is not justified.

The key placement question is whether this belongs in MNEL, MNCS, Forge, or Fabric.
It crosses those boundaries, but it is not a new execution authority or conformance
mechanism.

## Decision

MNEL adopts one-step distilled specialists as a documented provider pattern.

1. **MNEL owns the operational provider surface.** It declares the task, input snapshot,
   output contract, teacher/procedure identity, distillation transform, calibration,
   abstention, fallback, study controls, diagnostic observation, and routing/escalation
   behavior.
2. **The student is a learned micro-provider, not a new catalog family.** The pattern can
   apply to transition, graph, sequence, pair, anomaly, tabular, or routing providers.
   Catalog selection remains deterministic-first and capability-bound.
3. **One step means one learned forward pass after representation construction.**
   Snapshot construction, feature extraction, normalization, schema/policy checks, Forge
   probes, and escalation are outside that bound.
4. **The teacher provides candidate behavior, not truth.** Teacher traces are training
   evidence or candidate targets. Independent witnesses, evaluators, and hard gates
   retain authority over claims and outcomes.
5. **Abstention is mandatory.** Novel, ambiguous, unsupported, long-horizon, or
   out-of-distribution inputs must escalate to the teacher, an ordinary model, a
   deterministic path, or an explicit `UNKNOWN` result.
6. **Admission is evidence-based.** A student must be compared with its iterative teacher,
   a deterministic/classical baseline, and a no-distillation control under equal budgets.
   Admission studies must measure end-to-end snapshot cost, cold/warm latency, useful
   downstream probe yield, calibration, false accepts, abstention, and hidden transfer.
7. **MNCS owns the eventual training semantics and lineage.** The canonical training
   graph should record source evidence, teacher/procedure, target construction, student
   configuration, optimization, checkpointing, evaluation, packaging, and deployment
   identities. External trainers may serve as execution backends while that semantic
   contract is being implemented.
8. **Forge, Fabric, Harness, and Control retain their existing boundaries.** Forge verifies
   bounded claims and operating-envelope failures; Fabric executes and reports factual
   placement/residency; Harness/Control govern use and escalation. None treats student
   output as conformance, evaluator, permission, or promotion authority.

## Current bounded implementation

The Python teacher/student study remains the independent reference and no-distillation
control. A first MNCS-native fixed-point slice now exercises the same architectural
boundary: it retains six accepted observations plus rejected and `UNKNOWN` observations,
records the teacher/model/dataset/distillation/training/checkpoint/artifact lineage, and
executes stateful training followed by gated inference. The native source never invokes
the teacher, emits `Authority::DIAGNOSTIC_ONLY`, and sets `fallback_required` on OOD,
schema, and generation failures. Its four-case corpus agrees across research bytecode,
portable WASM, LLVM, C11, and Cranelift for the declared decision codes.

This evidence is intentionally bounded. The five compiler results retain `UNKNOWN`
status for unresolved checked-arithmetic obligations, and the execution agreement is
not a conformance proof, production admission, wall-clock speed result, or permission to
replace the teacher. Native `u64` lineage fields in this slice are bounded fingerprints,
not cryptographic identities.

## Consequences

### Positive

- Repeated bounded reasoning can move to a low-latency, resident specialist.
- The iterative teacher remains available as a correctness and novelty fallback.
- The architecture makes the speed claim measurable at system level instead of only at
  model forward-pass level.
- Teacher, student, distillation, and lineage identities remain auditable.
- The pattern can be applied across heterogeneous provider architectures without making
  them interchangeable.

### Costs and limitations

- Distillation can memorize teacher errors or reduce coverage on rare cases.
- A one-pass student may hide useful intermediate signals needed for attribution.
- Snapshot construction and verification may dominate the actual inference savings.
- Teacher traces, independent checks, calibration, transfer studies, and controls add
  data and operational cost.
- The pattern is not appropriate for open-ended generation, weakly constrained inputs,
  or tasks whose correctness requires unbounded interaction.

## Implementation and review requirements

Future implementations must:

- preserve the `diagnostic-only` authority and existing `PASS`/`FAIL`/`UNKNOWN` separation;
- retain teacher traces, source evidence, counterexamples, abstentions, and rejected
  students rather than collapsing them into a score;
- bind model, feature, snapshot, distillation, calibration, runtime, and artifact identities;
- keep the teacher/fallback path available during shadow and governed deployment;
- evaluate with hidden transfer and a no-distillation control before direct routing; and
- document any claim of one-step with the exact measured boundary and excluded costs.

This ADR does not add a runtime dependency, change the v1 ABI, or claim that a one-step
student currently improves MNEL. The bounded native slice is diagnostic pressure and
evidence for the migration path, not implementation or promotion closure.

See [MNEL architecture](../ARCHITECTURE.md), [learned micro-provider registry](../LEARNED_MICRO_PROVIDERS.md),
and [MNCS-native training pipeline](../MNCS_NATIVE_TRAINING_PIPELINE.md).
