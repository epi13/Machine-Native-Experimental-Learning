# Learned micro-provider registry

## Purpose

MNEL uses learned micro-providers to locate surprise, disagreement, similarity, and
likely omitted questions inside bounded experimental evidence. They complement Gemma or
another language investigator and Forge micro-verifiers; they do not replace either.

A learned micro-provider may answer questions such as:

- Which candidate transition differs most from learned expectations?
- Which bounded graph region is structurally unusual?
- Does a runtime event sequence contain an unfamiliar phase transition?
- Which candidate/reference pair deserves a deterministic equivalence probe?
- Which verifier and metric combination is outside the calibrated experience envelope?

It may not answer:

- Does this candidate pass MNEL evaluation?
- Is this implementation conformant with MNCS or MNCDS?
- Is the proposed explanation causally true?
- Should a candidate be promoted?

Those decisions remain outside the learned-provider surface.

## System position

```text
normalized event, experiment record, or diagnostic snapshot
                         |
                         v
       deterministic learned-provider capability matching
                         |
                         v
       one or more diagnostic-only learned micro-providers
                         |
             discrepancy / anomaly / similarity
                         |
                         v
       investigator or skeptic states falsifiable hypotheses
                         |
                         v
           Forge runs bounded deterministic probes
                         |
                         v
           verifier result: PASS / FAIL / UNKNOWN
                         |
                         v
        MNEL evaluation and attribution remain separate
```

A learned result is a diagnostic observation. Agreement among several models does not
become a verifier verdict. Disagreement is preserved because the pattern of disagreement
may be more useful than a vote.

## Registry contract

Every declaration identifies:

- provider ID and version;
- architecture and objective families;
- purpose and architectural rationale;
- training objective;
- accepted input views and artifact types;
- uncertainty classes it can help investigate;
- output kinds;
- required diagnostic snapshot types;
- advantages and limitations;
- approximate learned-scalar capacity range;
- runtime cost and CPU suitability;
- calibration requirement; and
- the immutable `diagnostic-only` authority boundary.

The declaration receives a canonical identity. A model invocation must additionally bind
its exact weight identity, feature-extractor identity, query identity, snapshot
identities, calibration interpretation, and limitations.

The registry performs deterministic compatibility matching. It does not execute a model,
download weights, infer undeclared capability, or permit a learned router to cross policy
boundaries.

## Initial heterogeneous portfolio

The first catalog intentionally includes neural and classical learned architectures.
The objective is complementary failure surfaces, not architectural novelty for its own
sake.

| Provider | Architecture | Primary input | Why it is included | Important limitation |
|---|---|---|---|---|
| `latent.transition-jepa` | Transition JEPA | Parent state, intervention, resulting state | Directly models the MNEL episode and identifies effects that diverge from expected transition semantics | Can absorb unseen failures into its learned notion of normal |
| `latent.graph-jepa` | Graph JEPA | Masked AST, CFG, use-def, or dependency subgraphs | Predicts semantic graph regions rather than source tokens and can surface missing structural relations | Graph extraction and snapshot identity can dominate cost |
| `graph.message-passing-gnn` | Typed-edge GNN | Ownership, dependency, contract, and data-flow graphs | Explicit relation propagation is useful for ownership, reachability, and contract questions | Bounded neighborhoods can miss long paths and deep stacks can oversmooth |
| `sequence.selective-state-space` | Selective state-space model | Long ordered diagnostic and runtime streams | Retains relevant temporal state with bounded memory and near-linear sequence scaling | Internal state is hard to attribute and training is less mature than simpler baselines |
| `sequence.dilated-temporal-conv` | Dilated temporal convolution | Runtime windows, compiler events, metric series | Very low startup cost and multi-scale temporal pattern detection make it suitable for CPU-first operation | The receptive field is fixed and sparse long-range effects may be missed |
| `pair.contrastive-siamese` | Siamese metric learner | Candidate/reference, before/after, and mutation pairs | Natural differential representation for ranking likely semantic drift and duplicate mechanisms | Similarity is not equivalence and pair sampling defines much of the behavior |
| `anomaly.deep-svdd` | One-class representation model | Verifier vectors and experiment summaries | Provides a compact anomaly detector when verified failures are scarce | Contaminated normal data and weak localization can make scores misleading |
| `anomaly.denoising-autoencoder` | Denoising autoencoder | Multi-view structured feature vectors | Cross-view reconstruction residuals can identify missing evidence and observability loss | Reconstruction quality is not semantic correctness and identity shortcuts are possible |
| `interaction.tiny-transformer` | Small attention encoder | Bounded heterogeneous evidence sets | Handles sparse nonlocal interactions among evidence items that graphs or fixed windows may omit | Highest initial cost, easy to overfit, and attention is not explanation |
| `tabular.gradient-boosted-trees` | Boosted shallow trees | Structured verifier, metric, and routing features | Strong small-data, CPU-friendly, interpretable baseline that prevents neural models from receiving automatic credit | Cannot directly consume raw graphs or long sequences |
| `state.hidden-markov-model` | Hidden Markov model | Lifecycle and protocol event sequences | Explicit transition probabilities are appropriate for skipped phases, stale states, and illegal transitions | Markov assumptions and state-count selection are restrictive |
| `sequence.reservoir-computer` | Echo-state reservoir | Runtime traces, event sequences, metric series | Only the readout is trained, providing a very cheap temporal control on local hardware | Reservoir hyperparameters are brittle and task adaptation is limited |

The catalog is a starting portfolio, not a commitment to retain every architecture.
Providers must earn continued inclusion through hidden-transfer utility, calibration,
complementary error, and cost-per-useful-probe measurements.

## Why this is not a conventional mixture of experts

A normal MoE routes tokens through interchangeable neural experts and combines their
outputs. MNEL instead routes a declared diagnostic question to heterogeneous providers
whose outputs remain separate records.

```text
conventional MoE                 MNEL learned micro-provider layer
------------------------------   ------------------------------------------
learned token router             deterministic capability filter first
mostly interchangeable experts   different architecture and evidence views
combined hidden output           preserved individual observations
single model authority surface   no learned evaluator authority
optimize task loss               optimize useful-question discovery under cost
```

A learned ranking router may later order already-compatible declarations, but it may not
expand authority, access hidden partitions, bypass cost ceilings, or invoke an
undeclared provider.

## Deterministic matching and diversity selection

A query declares one or more uncertainty classes and may constrain artifact types,
available snapshot types, output kinds, architecture preferences, exclusions, and a
maximum cost.

The registry:

1. filters by declared compatibility and cost;
2. rejects providers whose required snapshots are known to be unavailable;
3. scores matching uncertainty, artifact, and output coverage;
4. returns stable results ordered by score, cost, and provider ID; and
5. optionally selects a diverse subset by architecture family, objective family, and
   input view.

Diversity selection is not voting. It is a deterministic way to avoid spending a budget
on several highly correlated models when a smaller, more heterogeneous panel can expose
more failure classes.

## Snapshot reuse

Model startup should be secondary to representation construction. Forge or another
provider may build an identity-bound AST, IR, graph, trace, metric, or composite
snapshot once and allow several learned micro-providers to consume bounded views of it.

Reuse must bind at least:

- candidate and epoch;
- source, generated input, and dependency identities;
- feature extractor and normalization identities;
- compiler, analyzer, toolchain, and environment identities;
- snapshot construction method and version;
- provider declaration and weight identities; and
- calibration snapshot identity.

A changed material identity invalidates reuse unless a complete dependency envelope
proves the snapshot unaffected. Missing impact information remains unresolved.

## Observation semantics

A learned-provider observation records a finite value and one declared output kind, for
example latent discrepancy, anomaly score, pair similarity, next-state distribution,
feature contributions, or candidate ranking.

It also records:

- calibration band;
- out-of-distribution state;
- suggested uncertainty classes;
- bounded candidate locations when available; and
- explicit limitations.

The record carries `authority: diagnostic-only` and `verdict_semantics: not-a-verdict`.
There is deliberately no `PASS`, `FAIL`, or evaluator-eligibility field in the
observation.

## Training and admission requirements

A provider should not enter active routing merely because it can be trained. Admission
should require:

1. a deterministic or classical baseline for the same input view;
2. immutable training, calibration, and hidden-transfer partitions;
3. exact data, feature, architecture, weight, and runtime identities;
4. calibration and abstention behavior;
5. unsupported-construct and out-of-distribution reporting;
6. ablations against random routing and heuristic discrepancy ranking;
7. correlated-error analysis against the existing registry;
8. cost per useful downstream Forge probe;
9. proof that it discovers confirmed failure families missed by cheaper alternatives; and
10. a quarantine and rollback path.

A provider that is accurate but redundant may be removed. A less accurate provider may
remain valuable when it discovers a distinct class of omitted questions.

## Current implementation boundary

This foundation implements declarations, canonical identities, deterministic matching,
diversity selection, a diagnostic observation record, CLI inspection, schema, examples,
and tests.

It does not yet:

- train or execute any model;
- download third-party weights;
- add PyTorch, ONNX, or another runtime dependency;
- build Forge diagnostic snapshots;
- create an automatic learned router;
- convert observations into hypotheses without an investigator;
- grant evaluator-mode access; or
- claim that any listed architecture improves MNEL.

Those are empirical follow-on tasks. The first implementation should collect identified
MNEL episodes and Forge diagnostic snapshots, then compare each provider against cheap
baselines under equal budgets.
