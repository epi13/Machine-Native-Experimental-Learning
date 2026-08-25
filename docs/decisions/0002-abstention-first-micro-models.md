# ADR 0002: Abstention-first micro-models as bounded cognitive components

- **Status:** Accepted
- **Date:** 2026-08-25

## Context

MNEL was created in part to explore very small learned providers that can replace repeated general-model reasoning with narrow, evidence-governed operations. The repository already supports learned micro-providers, but this intent should be explicit enough to guide future runtime, training, and family-integration work.

A micro-model should not be judged by whether it behaves like a miniature general assistant. It should be judged by whether it performs one bounded operation reliably, identifies when that operation is outside its calibrated envelope, and escalates without acquiring authority.

## Decision

MNEL will treat task-specific micro-models as first-class bounded cognitive components.

The default design order is the **smallest sufficient mechanism**:

1. deterministic mechanism;
2. tiny learned provider;
3. small specialist model;
4. ordinary local model;
5. larger general model.

Every production learned micro-provider must have an explicit abstention/escalation path. Learned outputs remain proposals or diagnostic observations unless a separate governed consumer gives them bounded operational effect after validation.

MNEL work on micro-models should optimize for false-accept reduction, calibrated coverage, latency, resource use, and larger-model/token avoidance—not raw completion rate alone.

For structured tool calling, constrained generation from authoritative schemas is preferred so structural invalidity is removed from the learned problem wherever possible.

## Consequences

- Training and evaluation pipelines should preserve the provider's declared operating envelope and escalation target.
- Shadow deployment and adversarial evaluation should precede direct routing.
- Verified successful traces from larger models are eligible training evidence when their lineage and authority are preserved.
- Failures, abstentions, operator corrections, and schema changes are first-class negative evidence.
- MNEL should expose provider declarations and observations that Harness/Control can consume without transferring authority to the provider.
- Fabric may carry factual residency/capability and execute selected providers, but semantic selection remains outside Fabric.
- Forge should be able to evaluate operating-envelope violations and abstention failures.

See `docs/MICRO_MODEL_OPERATING_MODEL.md` for the detailed operating model.