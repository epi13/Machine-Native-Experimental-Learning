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
learned micro-provider registry
  discrepancy / anomaly / similarity / disagreement
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

### Learned diagnostic plane

The learned micro-provider registry exposes small, heterogeneous models through typed,
identity-bearing declarations. Providers locate surprise, disagreement, similarity,
likely omitted questions, and candidate regions worth probing. Deterministic matching
filters providers by uncertainty class, artifact and snapshot compatibility, output
kind, cost, and declared limitations before any future runtime executes them.

Learned-provider observations are development context with `diagnostic-only` authority.
They have no `PASS`, `FAIL`, or evaluator-eligibility semantics. Agreement among learned
providers remains agreement among proposals; disagreement is preserved as a potentially
useful diagnostic event. An investigator or skeptic must translate an observation into
a falsifiable hypothesis, and Forge must answer bounded claims through its normal
verifier boundary.

Provider lifetime and physical placement are separate. The persistent Rust host keeps an
admitted provider reusable, while its placement policy may select CPU, full CUDA, or
sequential CPU offload. The latter keeps weights in system RAM and temporarily executes
modules on CUDA; it does not reload a provider for each query. Placement decisions are
resource-accounted and diagnostic, never evaluator decisions.

### Bounded Forge-oriented diagnostic lifecycle

The executable local lifecycle is deliberately split into independent records:

```text
identified snapshot -> compatible verifier -> precondition report -> bounded probe
       -> diagnostic witness -> optional registered mutation -> independent comparison
       -> health/coverage -> proposal-only omitted-question candidate
```

`src/mnel/snapshots.py` provides compact immutable views shared by deterministic
verifiers and learned providers. `src/mnel/forge_lifecycle.py` provides the explicit
registry and reference execution surface. Witnesses characterize observations; they
contain no evaluator verdict field. Health describes execution reliability, not truth,
and comparisons preserve disagreement rather than voting it away.

### Probe plane

Forge supplies small, identity-bearing questions and witnesses. The stable interface is
the bounded question, inputs, outputs, resource budget, and verifier identity—not a
specific analyzer brand. Learned-provider output may prioritize probes but cannot
replace their witnesses or verdicts.

### Execution plane

MNCS Fabric is the intended distributed execution and reconciliation substrate. MNEL
also supports a deterministic single-host reference path. Execution produces raw facts,
not semantic acceptance.

### Evaluation plane

An immutable evaluator derives every hard gate. Failed gates cannot be compensated by a
weighted aggregate. Missing or incomparable evidence remains `UNKNOWN`. Learned models
are not evaluator providers in this architecture.

### Experience plane

The append-only ledger retains successful, erroneous, neutral, abstaining, rejected,
and inconclusive experience. Records are content-identified and chained in order.
Learned observations retain declaration, weight, feature, query, snapshot, calibration,
and limitation identities before they are eligible for later study.

The 0.4 provider-study plane adds a small explicit portfolio contract on top of that
ledger. `mnel.provider_study` binds provider artifacts to training/calibration dataset
identities, supported snapshot families, artifact/model identities, and diagnostic-only
metadata. It runs a transition-frequency provider beside a structurally distinct tabular
nearest-centroid provider and explicit random/heuristic controls. Routing is recorded,
not learned authority; every observation remains separate. Cold load, first inference,
warm samples, artifact/model bytes, bounded Python allocation, calibration, disagreement,
OOD, abstention, and optional energy availability are measurements.

### Verified distillation and study plane

The 0.4 data plane makes reuse explicit without replacing evidence:

```text
development-visible episodes
  -> attribution-linked source-preserving group
  -> provisional principle/strategy with counterexamples
  -> negative-memory-aware retrieval
  -> frozen transfer prediction
  -> hidden transfer evidence
  -> retained/quarantined/rejected strategy lineage
```

`mnel.distillation` exposes explicit development and transfer access views. Development
study code cannot retrieve `TRANSFER_HIDDEN` or `FUTURE_FINAL` records, and a strategy
cannot be repaired from its own hidden-transfer result. Groups, principles, strategies,
and negative memory retain source identities; heuristic grouping is labeled as a
reference feature baseline rather than semantic truth. Retrieval returns record classes
and match reasons separately, while calibration and transfer status remain measurements.
Study arms and ablations carry equal-budget declarations and deterministic identities.
The `distill-reference` command is a synthetic, no-network study surface, not an
evaluator or a substitute for Forge, Fabric, MNCS, MNCDS, or RAVEL.

Provider lifecycle records separately capture candidate, development admission,
transfer-pending, transfer admission, quarantine, retirement, and rollback. Admission
criteria identify artifact/calibration/resource/OOD/authority checks; a provider cannot
transition on its own score, and quarantine preserves its artifact and evidence. The
portfolio reference command is a bounded local study, not a production provider
orchestrator and not a native ABI export.

### Distillation plane

VED proposes compact principles and strategies while preserving source lineage,
counterexamples, scope, falsifiers, transfer status, and known failure modes. It never
deletes the underlying episodes.

### Governor plane

The recursion governor checks evaluator, threshold, partition, resource-policy,
visibility, budget, lineage, stopping, and rollback invariants. It is not an optimizer.
A learned router may eventually rank already-compatible providers, but it cannot expand
capability, authority, disclosure, partition access, or cost ceilings.

## Concurrency

Parallel workers must never mutate a shared active candidate in place. Each job binds:

- parent candidate digest;
- eligible experience snapshot digest;
- investigator and model/runtime identity;
- learned-provider declaration, weights, feature extractor, calibration, and snapshot
  identities when used;
- probe, executor, evaluator, and governor identities;
- preregistered predictions and hard gates;
- operation and wall-time budget; and
- unique transaction identity.

Workers append results. A separate selector may accept one child transaction; rejected
children remain available as negative memory.

The local investigator harness now packs eligible records under explicit visibility and
byte ceilings, separates read-only from proposal workspaces, binds runtime/model/tool
identities, and creates proposal-only candidate transaction identities. These contracts
do not grant filesystem or network authority and do not implement unattended execution.

## Model independence

MNEL is designed so Gemma, Qwen, a multimodal worker, a JEPA-derived predictor, a graph
network, a state-space model, a classical learned baseline, or another model can perform
its declared role without becoming the durable knowledge store. Persistent learning
resides in identified episodes, observations, attributions, principles, strategies, and
RAVEL candidate lineage.

See [Learned micro-provider registry](LEARNED_MICRO_PROVIDERS.md) for the initial
architecture portfolio, matching contract, training admission requirements, and current
implementation boundary.
