# Roadmap

## 0.1 — executable method foundation

- canonical identities and append-only ledger;
- investigator contracts;
- experiment lifecycle and governor;
- independent hard-gate evaluator;
- causal attribution and VED proposals;
- provider-neutral integration records;
- diagnostic-only learned micro-provider declaration registry;
- deterministic compatibility matching and diversity selection;
- learned-provider schema, architecture catalog, CLI inspection, and negative tests;
- accepted Rust-first provider-runtime architecture decision;
- versioned provider C ABI, safe Rust SDK, host admission policy, manifest schema, and
  Python control-plane contract;
- schema, reference study, tests, CI, and documentation.

## 0.2 — local investigator harness and provider runtime

- **Implemented:** eligible-context packing with visibility and byte ceilings;
- **Implemented:** explicit read-only/proposal workspace models and proposal-only candidate
  transactions;
- **Implemented:** model, quantization, runtime, prompt, and tool-schema identity envelope;
- **Implemented:** deterministic morning-report and quarantine queue records;
- **Implemented:** process-local persistent Rust provider host with reusable provider state;
- **Implemented:** host-owned bounded result normalization, failure quarantine, and clean unload;
- **Implemented:** first native Rust HMM classical provider baseline;
- **Implemented:** warm/cold timing, copied-byte, output, placement, and snapshot-reuse
  measurement harness;
- **Implemented:** bounded adapter to the local JSON-line harness protocol, including
  timeout/output limits, machine-readable response validation, authority-expansion
  rejection, deterministic diagnostic observations, and explicit proposal-only records;
- **Implemented:** Git-validated detached proposal worktrees with source commit identity,
  root confinement, reproducible metadata, immutable authoritative checkout behavior, and
  explicit preservation or cleanup;
- **Implemented:** ABI v1 dynamic-library loading and validation in a small Rust unsafe
  boundary, including artifact hashing, descriptor identity checks, pointer/length checks,
  host-owned output copying, clean unload, and quarantine integration. ABI v1 is unchanged.

## 0.3 — Forge experiment lifecycle

- **Implemented:** explicit micro-verifier registry and identity-bound declarations;
- **Implemented:** probe preconditions, proposal-bound requests, and diagnostic-only witness schemas;
- **Implemented:** bounded counterfactual and registered mutation probe support;
- **Implemented:** independent-probe comparison preserving agreement, disagreement, and incomplete coverage;
- **Implemented:** verifier health, quarantine, and coverage records;
- **Implemented:** deterministic skeptic-driven omitted-question candidate discovery with
  bounded, deduplicated, lineage-checked candidates for coverage holes, disagreements,
  abstentions, missing mutation counterparts, and verifier health gaps;
- **Implemented:** identity-bound transition, tabular, pair, trace, graph, and composite diagnostic snapshots with
  immutable compact binary payloads, producer/source/dependency/extractor identities, and
  deterministic content identities suitable for deterministic probes and learned
  micro-providers;
- **Implemented:** compact binary snapshot views shared across compatible providers;
- **Implemented:** learned observations normalized as diagnostic events without verifier status.

## 0.4 — verified distillation and learned-provider studies

- **Implemented:** source-preserving deterministic reference grouping with explicit
  extractor/method identities and limitations;
- **Implemented:** attribution-linked provisional principles and strategies, negative
  memory, frozen transfer predictions, held-out transfer evidence, and same-candidate
  hidden-repair rejection;
- **Implemented:** success- and negative-memory, shuffled-attribution, aggregate-only,
  random, fixed-policy, equal-budget, and hidden-transfer control specifications;
- **Implemented:** explicit class-preserving retrieval with negative-memory demotion and
  precision/recall/hit-rate/reuse/diversity metrics plus bounded calibration metrics;
- **Implemented:** deterministic `mnel distill-reference` study and a tiny reloadable CPU
  transition-frequency learned provider with OOD abstention and diagnostic-only output;
- **Implemented:** a deterministic heterogeneous reference portfolio with a structurally
  distinct tabular nearest-centroid provider, explicit calibration records, deterministic
  serialization/reload, routing controls, disagreement/correlation/OOD/abstention/resource
  measurements, and diagnostic-only evidence ledgers;
- **Implemented:** provider candidate/admission, transfer-pending, quarantine, retirement,
  and rollback records with explicit evidence checklists;
- **Implemented:** first executable one-step distilled specialist reference for the
  Forge evidence-relevance role, including retained recurrent teacher observations,
  independently checked target status, a tiny reloadable affine/tanh student,
  no-distillation control, calibrated/OOD abstention, explicit teacher fallback, and
  measured reference comparisons. This remains diagnostic-only and experimental.
- **Started:** broader provider portfolios and native export of Python-trained artifacts;
- **Implemented:** a bounded Rust parser/reference inference surface for the existing
  transition-frequency artifact, with checked-in Python/Rust identity and score-equivalence
  fixtures; ABI v1 host initialization remains open;
- export Python-trained providers into the versioned native runtime boundary;
- **Implemented:** compare the reference providers against seeded-random and explicit
  heuristic controls; the Rust HMM remains a separate not-applicable native baseline for
  this compact mixed snapshot fixture;
  the reference transition study records deterministic and heuristic controls; the
  existing Rust HMM baseline is not input-compatible with this compact fixture and is
  retained as a separate native runtime baseline;
- **Implemented:** random, heuristic, single-provider, and diversity-routed controls;
- **Implemented:** correlated-error, disagreement, abstention, and out-of-distribution
  measurements, plus cold-load/first-inference/warm-latency and bounded Python allocation
  measurements;
- **Started:** energy measurements; an optional reader records trusted joules when supplied,
  otherwise the study records unavailable rather than estimating;
- non-Rust native exception studies with benchmark and threat-review identities;
- hidden-transfer admission, quarantine, retirement, and rollback workflows;
- optional small proposer-model distillation from verified traces, including one-step
  students measured against iterative teachers, deterministic baselines, and explicit
  abstention/escalation paths.

## 0.5 — MNCS Fabric execution

- **Implemented (local reference):** content-addressed identified workloads and bounded
  acyclic workload graphs;
- **Implemented (local reference):** capability-aware dispatch through Fabric's public
  `LocalController`/`LocalWorker` boundary with provider-artifact identity binding;
- **Implemented (local reference):** typed Fabric execution/receipt observations retained
  as MNEL evidence without evaluator authority;
- **Implemented (local reference):** replicated expert runs and deterministic reconciliation
  observations across distinct logical worker identities;
- **Implemented (local reference):** transition-frequency and nearest-centroid sharded
  sufficient-statistic training with duplicate/overlap/missing/visibility validation;
- **Started:** optional authenticated remote dispatch through `NetworkController` and
  `TLSNetworkTransport`; current network mode requires operator-pre-staged bundles;
- **Started:** node-loss/UNKNOWN handling, locality hints, and scaling measurements;
- **Started:** authenticated worker enrollment consumption and provider-artifact admission
  across physical nodes; live remote execution remains operator-only and unverified here.

## Cross-cutting — MNCS-native training and lineage closure

See [MNCS-native training pipeline](MNCS_NATIVE_TRAINING_PIPELINE.md).

- make MNCS the semantic source of truth for dataset construction, transforms, training,
  checkpointing, evaluation, promotion, and model-artifact creation;
- retain external numerical and accelerator runtimes as implementation backends where
  useful without letting them own the canonical training semantics;
- treat every training run as an evidence-producing computation with stable identities
  for data, transforms, model architecture, optimization policy, backend/runtime,
  checkpoints, evaluation, and resulting artifacts;
- define training-lineage records that `mncs-lineage` can consume so deployed models can
  be traced through quantization/distillation/checkpoint ancestry to source observations;
- use MNEL training as a deliberate stress test for `mncs-language`, its compiler, and
  the standard library;
- promote broadly useful primitives discovered by training work into shared MNCS layers
  instead of hiding numeric, graph, streaming, parallel, checkpoint, artifact, or
  observability capabilities inside MNEL;
- begin with MNCS-owned training manifests and lineage over existing trainers, then move
  dataset/evaluation semantics, training primitives, backend lowering, and selected
  micro-model training into MNCS in phases;
- preserve existing diagnostic-only learned-provider authority, hard-gate evaluation,
  quarantine, rollback, and explicit UNKNOWN semantics throughout the migration.

## 0.6 — RAVEL integration study

- RAVEL episode ingestion and candidate proposal API;
- transactional knowledge and policy candidates;
- retention, transition, planning, resource, and rollback gates;
- lineage-aware delayed credit;
- hidden transfer provider;
- equal-budget investigator and recursion controls;
- learned-provider disagreement and utility as candidate-generation context only.

## Later assurance

- hardened Fedora runner with network denial and cgroup limits;
- immutable or attested verifier nodes;
- signed ledgers and transparency log;
- external transfer and final custody;
- heterogeneous cluster studies;
- energy and compute-efficiency evidence;
- formal MNCS/MNCDS integration after eligible external evidence exists.
