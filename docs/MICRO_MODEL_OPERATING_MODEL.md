# Micro-model operating model

MNEL treats very small, task-specific learned providers as bounded cognitive components rather than miniature general assistants. The objective is to collapse repeated, well-defined reasoning work into the smallest mechanism that can perform it reliably, while preserving abstention, evidence, and authority boundaries.

## Principle: smallest sufficient mechanism

For a recurring task, prefer the least complex mechanism that can satisfy the declared operating envelope:

1. deterministic code or grammar;
2. tiny learned provider;
3. small specialist model;
4. ordinary local model;
5. larger general model.

Escalation is expected behavior. A micro-model is successful when it handles its admitted domain precisely and abstains outside it; it is not required to solve every request.

## Intended roles

Micro-models may be trained for narrow operations such as:

- tool-family selection;
- tool selection from a bounded candidate set;
- argument extraction under a compiled schema or grammar;
- task and intent classification;
- model or worker routing proposals;
- context and source relevance ranking;
- error-family classification;
- verifier or probe selection;
- anomaly and omission detection;
- risk or escalation prediction;
- compact state summarization.

These outputs remain proposals or diagnostics. They do not grant authority.

## Authority boundary

A learned provider may propose what should happen next, but it must not decide that it is authorized to happen. Permissions, trust, worker identity, destructive-action approval, evidence validity, custody, promotion, and conformance remain outside the learned provider.

Where a provider emits a tool call or structured action, deterministic policy and schema validation still gate execution.

## Abstention-first contract

Every production micro-model must define an explicit abstention path. The provider contract should distinguish at least:

- `ACCEPTED`: result is within the declared operating envelope;
- `ABSTAIN`: uncertainty, novelty, ambiguity, or missing context requires escalation;
- `REJECTED`: input violates the provider contract or policy.

Confidence is evidence for routing, not authority by itself. Thresholds must be calibrated on held-out and adversarial data rather than chosen from training performance.

## Operating envelope

A micro-model declaration should identify:

- task family and exact output contract;
- admissible input representation;
- schema or grammar identity when output is constrained;
- training/evaluation lineage;
- calibration method and thresholds;
- known failure modes and counterexamples;
- resource profile and expected latency;
- escalation target;
- version, artifact identity, and rollback predecessor.

MNEL should preserve these declarations alongside observations so that deployment decisions can be reproduced and audited.

## Verified distillation path

A preferred lifecycle for repeated work is:

```text
verified successful general-model decisions
  -> task-family clustering
  -> bounded dataset construction
  -> micro-model training
  -> offline evaluation
  -> shadow deployment
  -> Forge adversarial evaluation
  -> calibrated operating envelope
  -> governed Harness/Control use
  -> continuous observation and rollback eligibility
```

Promotion requires evidence that the micro-model improves the intended objective inside its envelope without silently expanding its authority or domain.

## Token-efficiency objective

Micro-models should be evaluated not only on task accuracy but on how much larger-model context and invocation they eliminate. Useful metrics include:

- larger-model calls avoided;
- prompt/schema tokens avoided;
- p50/p95 routing latency;
- abstention rate;
- false-accept rate;
- schema-valid output rate;
- escalation correctness;
- cost and energy per accepted task;
- reliability under catalog/schema changes.

A provider that abstains frequently but nearly eliminates false accepts can still be valuable if escalation remains cheap.

## Structured tool calling

Tool-calling micro-models should prefer constrained generation. When a tool contract can be expressed as a typed schema, enum, grammar, or MNCS-language contract, the decoder should be restricted so that invalid tool names, argument keys, and structurally invalid values cannot be produced.

The recommended pipeline is:

```text
request
  -> candidate-family routing
  -> bounded tool candidate set
  -> constrained argument generation
  -> schema validation
  -> deterministic policy gate
  -> execution substrate
```

This keeps semantic prediction separate from authorization and execution.

## Relationship to MNCS family

- **MNEL** discovers, trains, calibrates, evaluates, and records candidate micro-models.
- **MNCS Harness / Control** decide when a micro-model may be used and when to escalate.
- **MNCS Fabric** may report factual provider residency/capability and execute the selected provider on an exact target; it does not own semantic routing.
- **MNCS Forge** stress-tests the provider's operating envelope, abstention behavior, and regressions.
- **MNCS Language** can supply typed contracts and compiled constrained-decoding grammars.

The key design rule is that learned components reduce repeated cognitive work while deterministic systems retain authority.