# MNEL architecture

## Objective

MNEL coordinates a bounded scientific-development loop around RAVEL and the MNCS
project family. The architecture is designed so that a capable local model can spend
long periods generating useful experiments without gaining authority over the facts it
produces.

```text
eligible experience snapshot
          |
          v
investigator portfolio
  investigator / skeptic / replicator / synthesizer / auditor
          |
          v
preregistered hypotheses, predictions, interventions, and budgets
          |
          v
Forge-compatible bounded probes
          |
          v
single-host or MNCS Fabric execution
          |
          v
raw observations and artifact identities
          |
          v
immutable hard-gate evaluator
          |
          v
causal attribution with alternatives and credit classes
          |
          v
Verified Experience Distillation proposals
          |
          v
transactional RAVEL knowledge or policy candidate
```

## Planes

### Investigator plane

Local models inspect only records eligible for their role and visibility. They may
propose hypotheses, probes, interventions, principles, or strategies. Their output is
untrusted and never becomes a verdict merely because it is coherent.

### Probe plane

Forge supplies small, identity-bearing questions and witnesses. The stable interface is
the bounded question, inputs, outputs, resource budget, and verifier identity—not a
specific analyzer brand.

### Execution plane

MNCS Fabric is the intended distributed execution and reconciliation substrate. MNEL
also supports a deterministic single-host reference path. Execution produces raw facts,
not semantic acceptance.

### Evaluation plane

An immutable evaluator derives every hard gate. Failed gates cannot be compensated by a
weighted aggregate. Missing or incomparable evidence remains `UNKNOWN`.

### Experience plane

The append-only ledger retains successful, erroneous, neutral, abstaining, rejected,
and inconclusive experience. Records are content-identified and chained in order.

### Distillation plane

VED proposes compact principles and strategies while preserving source lineage,
counterexamples, scope, falsifiers, transfer status, and known failure modes. It never
deletes the underlying episodes.

### Governor plane

The recursion governor checks evaluator, threshold, partition, resource-policy,
visibility, budget, lineage, stopping, and rollback invariants. It is not an optimizer.

## Concurrency

Parallel workers must never mutate a shared active candidate in place. Each job binds:

- parent candidate digest;
- eligible experience snapshot digest;
- investigator and model/runtime identity;
- probe, executor, evaluator, and governor identities;
- preregistered predictions and hard gates;
- operation and wall-time budget; and
- unique transaction identity.

Workers append results. A separate selector may accept one child transaction; rejected
children remain available as negative memory.

## Model independence

MNEL is designed so Gemma, Qwen, a multimodal 4M worker, or another model can act as an
investigator without becoming the durable knowledge store. Persistent learning resides
in identified episodes, attributions, principles, strategies, and RAVEL candidate
lineage.
