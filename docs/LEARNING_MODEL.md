# Learning model

## Machine-Native Experimental Learning

MNEL is a process in which machine agents improve a system by generating and testing
bounded explanations rather than only adjusting a monolithic parameter vector.

```text
observation
  -> hypothesis
  -> prediction
  -> intervention
  -> experiment
  -> immutable evaluation
  -> causal attribution
  -> transfer test
  -> reusable principle or strategy
```

The method can eventually produce neural training data, adapters, expert updates,
routing policies, transition memory, verifier improvements, or source-code candidates,
but none of those representations defines MNEL by itself.

## Knowledge classes

| Class | Purpose |
|---|---|
| Episodic | Exact events, routes, actions, outcomes, identities, and costs |
| Causal | Competing explanations, probes, interventions, effects, alternatives, and falsifiers |
| Semantic | Generalized principles supported by attributed experience |
| Procedural | Reusable adaptation strategies with triggers, preconditions, and failure modes |
| Negative | Rejected hypotheses, failed repairs, regressions, counterexamples, and prohibited contexts |

The classes may share storage but must not collapse into one score or embedding.

## Distillation is not deletion

Verified Experience Distillation compresses retrieval and reuse, not history. Every
principle and strategy retains source record identities. Raw records remain immutable.

## What can improve without changing model weights

A fixed local model can become more effective through:

- better retrieval of relevant experience;
- stronger negative memory;
- improved diagnostic and counterfactual probes;
- supported procedural strategies;
- calibrated investigator routing;
- richer RAVEL experts and transition structures; and
- improved experiment selection under fixed authority.

Later studies may test whether verified traces should also train a small proposer model,
but weight distillation is optional and subordinate to the evidence lifecycle.

## Required controls

A credible MNEL study should include:

- aggregate-only feedback;
- shuffled attribution;
- random proposal;
- success-memory ablation;
- negative-memory ablation;
- fixed-policy control;
- equal-budget investigator comparison; and
- hidden transfer evaluation.

A better final candidate alone does not prove the learning process caused the gain.

## Executable 0.4 foundation

The current deterministic study path preserves raw development episodes and builds
identity-bound feature groups, attribution-linked provisional strategies, and explicit
negative memory. Retrieval keeps episodic, causal, semantic, procedural, and negative
records as separate classes; a negative-memory conflict can demote a strategy without
deleting its positive lineage. Transfer predictions are frozen before hidden evaluation,
and hidden results cannot repair the same candidate.

`mnel distill-reference` exercises A0, A4, and A5 declarations plus aggregate-only,
shuffled-attribution, random, success-memory, negative-memory, fixed-policy,
equal-budget, and hidden-transfer controls. It computes retrieval and calibration
measurements and trains a tiny transition-frequency diagnostic provider that serializes
and reloads deterministically. These are measurements from a bounded synthetic fixture,
not evidence of general learning improvement or evaluator authority.

The companion `mnel provider-study-reference` path extends this into a small controlled
portfolio: a transition-frequency provider and a structurally different tabular
nearest-centroid provider are trained from development-visible records, calibrated from
separate identified examples, serialized/reloaded, and evaluated on a held-out fixture
through random, heuristic, single-provider, and diversity routing policies. It records
pairwise disagreement, bounded error correlation, abstention/OOD behavior, cold-load,
first-inference, warm-latency, model/artifact size, and optional energy measurements.
Provider admission, quarantine, retirement, and rollback are explicit diagnostic
lifecycle records. The synthetic labels and transfer policy are study evidence, not
MNEL/MNCS verdicts; the Rust HMM is retained as a separate native baseline because it is
not input-compatible with the mixed compact fixture.
