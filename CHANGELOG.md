# Changelog

## 0.3.0a0 — unreleased

- Add bounded binary views and an identity-keyed shared store for transition, pair,
  tabular, trace, graph, and composite diagnostic snapshots.
- Add an explicit MNEL-side Forge lifecycle surface: verifier registry and declarations,
  preconditions, bounded probe requests, diagnostic-only witnesses, reference verifiers,
  registered mutations, independent comparison, health/quarantine, coverage, learned
  observation events, and omitted-question candidates.
- Add a deterministic `mnel forge-reference` study path and lifecycle schema. This is a
  local diagnostic reference surface, not an implementation of external Forge authority.

## 0.2.0a0 — unreleased

- Add backend-neutral CPU, full-CUDA, and sequential CPU offload placement policy with
  reserve/cap/workspace math, real accelerator probe requirements, precision checks, and
  bounded AUTO OOM recovery.
- Extend runtime manifests with backward-compatible placement policy and capability
  metadata without changing C ABI v1.
- Add optional Torch/Accelerate adapter with observed sequential-offload verification.
- Add process-local Rust host lifecycle, identity-bound snapshot reuse, bounded host-owned
  result buffers, normalized diagnostic statuses, measurements, unload, and quarantine.
- Add executable deterministic Rust HMM provider baseline and provider-host benchmark.
- Add bounded investigator context packing, explicit workspace access, candidate transaction,
  identity envelope, quarantine, and morning-report contracts.
- Add executable local-harness request/observation normalization with timeout and output
  bounds, authority-expansion rejection, and explicit proposal-only semantics.
- Add Git-validated detached proposal worktree materialization, source commit identities,
  reproducible metadata, and explicit preserve-or-cleanup behavior.
- Add the reviewed Rust `mnel-provider-loader` unsafe boundary, SHA-256 artifact admission,
  v1 descriptor/query/result validation, host-owned output copying, native cdylib fixtures,
  and existing-host quarantine integration tests. ABI v1 is unchanged.

## 0.1.0a0 — unreleased

- Define Machine-Native Experimental Learning and Verified Experience Distillation.
- Add investigator role contracts and explicit authority boundaries.
- Add a canonical, append-only, hash-chained evidence ledger.
- Add experiment lifecycle, resource governor, hard-gate evaluator, causal attribution,
  distillation proposals, and negative-memory records.
- Add local-harness, Forge, MNCS Fabric, and RAVEL adapter contracts.
- Add a diagnostic-only learned micro-provider registry with canonical declarations,
  observations that cannot carry evaluator verdicts, deterministic capability matching,
  cost filtering, and diversity-aware selection.
- Add an initial heterogeneous portfolio spanning transition and graph JEPA, GNN,
  state-space, temporal convolution, Siamese, one-class, reconstruction, attention,
  boosted-tree, hidden-state, and reservoir architectures.
- Add learned-provider CLI inspection, JSON schema, catalog example, architecture and
  roadmap documentation, and negative tests.
- Establish Rust as the default production language for learned micro-provider hosting,
  dispatch, SDKs, and CPU-first provider implementations while retaining Python for
  training, calibration, experimentation, and high-level orchestration.
- Add accepted ADR 0001, the learned-provider runtime specification, and an explicit
  evidence-backed exception path for specialized non-Rust native implementations.
- Add a versioned allocation-neutral C ABI, matching public header, safe Rust SDK,
  persistent-host admission policy, reusable snapshot cache, and runtime manifest schema.
- Add Python runtime-manifest validation, a checked-in example, negative tests, and Rust
  formatting, Clippy, and test jobs in CI.
- Add schema, deterministic reference workflow, tests, CI, documentation, and roadmap.
