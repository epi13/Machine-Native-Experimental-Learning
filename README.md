# Machine-Native Experimental Learning

Machine-Native Experimental Learning (MNEL) is an evidence-governed experimental
learning framework in which autonomous **investigators** propose hypotheses, design
bounded interventions, invoke deterministic tools and verifiers, and submit observed
outcomes for causal attribution and verified-experience distillation.

MNEL does not treat model output as knowledge or authority.

```text
investigators propose
        |
        v
learned micro-providers locate surprise and omitted-question candidates
        |
        v
Forge-compatible probes interrogate bounded claims
        |
        v
MNCS Fabric-compatible executors run identified experiments
        |
        v
immutable evaluators derive PASS / FAIL / UNKNOWN
        |
        v
causal attribution separates result from explanation
        |
        v
Verified Experience Distillation proposes principles and strategies
        |
        v
RAVEL receives separately governed candidate knowledge
```

The project explores a learning process built from persistent machine-readable
experience, negative memory, causal attribution, transfer-gated principles, reusable
strategies, and append-only candidate lineage rather than relying exclusively on
conventional neural-weight training.

> **Current status:** functional `0.2.0a0` iteration. The repository now includes a
> backend-neutral accelerator placement policy, optional Torch/Accelerate adapter,
> process-local persistent Rust host, reusable identity-bound snapshots, bounded and
> normalized diagnostic results, failure quarantine, an executable Rust HMM baseline,
> deterministic runtime measurements, and bounded investigator context/workspace
> contracts, an executable local-harness/worktree path, and a validated Rust v1 dynamic
> provider loader. It still does not provide process isolation,
> unattended model execution, distributed scheduling, protected final custody, formal
> MNCS/MNCDS conformance, or automatic RAVEL promotion.

## Core rule

**Investigators and learned providers may propose knowledge. They may not declare it true.**

The model or agent that creates a hypothesis must not also become the authority that
accepts the resulting lesson. Evaluator identity, hard gates, partitions, resource
ceilings, custody, and promotion remain outside the investigator and learned-provider
surfaces.

## Relationship to the project family

| Project | MNEL role |
|---|---|
| `MNEL-local-harness` | Routes work across local models and exposes bounded tools to investigator roles |
| MNCS Forge | Supplies provider-neutral micro-verifiers, diagnostic probes, snapshots, and counterfactual witnesses |
| MNCS Fabric | Executes and reconciles content-addressed experiments across machines |
| RAVEL | Supplies the adaptive mechanism and consumes governed experience or candidate proposals |
| MNCDS | Records candidate generation, evidence eligibility, feedback boundaries, and lineage |
| MNCS | Evaluates frozen bounded claims; MNEL cannot issue conformance by itself |

MNEL integrates with those systems through explicit records and adapters. It does not
copy their authority or silently create substitute implementations.

## Implemented foundation

- canonical JSON and SHA-256 identities;
- append-only hash-chained JSONL evidence ledger;
- explicit experiment lifecycle and state transitions;
- investigator, skeptic, replicator, synthesizer, and auditor role contracts;
- immutable authority and resource-policy checks;
- development, selection-observed, transfer-hidden, and future-final visibility labels;
- independent hard-gate evaluation with `PASS`, `FAIL`, and `UNKNOWN`;
- causal-attribution records distinct from evaluator verdicts;
- Verified Experience Distillation (VED) principle and strategy proposals;
- transfer gating, source lineage, falsifiers, counterexamples, and negative memory;
- provider-neutral adapters for the local harness, Forge, MNCS Fabric, and RAVEL;
- typed learned micro-provider declarations and diagnostic observations;
- deterministic capability matching, cost filtering, and diversity-aware selection;
- an initial 12-family architecture catalog with declared advantages and limitations;
- accepted Rust-first runtime architecture decision and versioned C ABI;
- safe Rust provider SDK, host admission policy, reusable snapshot cache, and runtime
  manifest validation;
- backend-neutral CPU/full-CUDA/sequential-CPU-offload policy with reserve/cap/workspace
  accounting, real-probe requirements, precision checks, and bounded AUTO OOM recovery;
- explicit distinction between persistent provider lifetime and physical weight placement;
- process-local Rust host lifecycle, snapshot reuse, output-buffer limits, normalized
  status handling, measurement collection, and deterministic quarantine;
- executable `mnel-provider-classical` HMM diagnostic provider and host integration tests;
- eligible-context packing, read-only/proposal workspace models, identity envelopes,
  candidate transactions, quarantine queues, and deterministic morning-report records;
- bounded local-harness JSON-line execution with timeout/output ceilings, authority
  rejection, deterministic observations, and detached Git proposal worktrees;
- SHA-256 artifact-bound Rust dynamic loading for `mnel_provider_entry_v1`, descriptor and
  pointer/length validation, host-owned output copying, clean unload, and quarantine
  integration through the existing provider host;
- initial immutable transition, tabular, and pair diagnostic snapshot producers with
  compact binary payloads and dependency-bound content identities;
- deterministic reference workflow, JSON schemas, mutation-oriented tests, and CI.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

MNEL currently requires Python 3.11 or newer and has no Python runtime dependencies
outside the standard library. Building the provider runtime contracts additionally
requires Rust 1.79 or newer.

## Quick start

Inspect the environment and built-in role contracts:

```bash
mnel doctor
mnel investigator list
```

Inspect the learned micro-provider catalog:

```bash
mnel learned-provider list
mnel learned-provider describe latent.transition-jepa
mnel learned-provider match \
  --uncertainty unexpected-transition \
  --artifact candidate-transition \
  --snapshot transition-feature-snapshot \
  --max-cost low
```

Request a heterogeneous panel rather than the highest-scoring correlated providers:

```bash
mnel learned-provider match \
  --uncertainty temporal-anomaly \
  --uncertainty unexpected-transition \
  --uncertainty metric-inconsistency \
  --diverse \
  --limit 4
```

The catalog commands do not download or train models. The Rust reference provider can be
measured locally with:

```bash
cargo run -p mnel-provider-host --example provider-benchmark
```

The benchmark reports one local observation of host admission and warm invocation timing;
hardware-independent placement policy tests use fake accelerator diagnostics.

Run the deterministic reference lifecycle:

```bash
mnel demo --workspace build/demo
```

The demo preregisters a bounded experiment, records an observation, evaluates hard
gates, attributes the intervention, and creates a provisional principle proposal. It
does not call a model or modify RAVEL.

Verify and summarize the resulting ledger:

```bash
mnel ledger verify build/demo/evidence.jsonl
mnel ledger summarize build/demo/evidence.jsonl
```

## Experiment lifecycle

```text
draft
  -> preregistered
  -> running
  -> observed
  -> evaluated
  -> attributed
  -> distilled proposal

Any stage may terminate as rejected or UNKNOWN. A successful task metric alone does
not establish causal understanding, transfer, conformance, independence, or promotion.
```

Each material record binds its source identities, visibility, experiment identity,
parent or candidate identity where applicable, and the authority that produced it.

## Learned micro-providers

The registry starts with transition and graph JEPA variants, a message-passing GNN,
selective state-space and temporal convolution sequence models, a contrastive Siamese
encoder, Deep SVDD, a denoising autoencoder, a tiny transformer, gradient-boosted trees,
a hidden Markov model, and reservoir computing.

The portfolio is intentionally heterogeneous. Each architecture has a declared input
view, purpose, objective, cost, size range, strengths, and limitations. Learned outputs
remain `diagnostic-only`, carry `verdict_semantics: not-a-verdict`, and can only help an
investigator decide which bounded Forge question to ask next.

See [Learned micro-provider registry](docs/LEARNED_MICRO_PROVIDERS.md).

## Provider runtime implementation policy

Rust is the reference and default production language for the persistent provider host,
provider SDK, dispatch, budget enforcement, snapshot reuse, and CPU-first provider
implementations. Python remains the training, calibration, experimentation, export, and
high-level orchestration language.

The stable cross-language boundary is `mnel-provider-c-abi/1`. Native-trusted providers
must be Rust unless an identified benchmark and threat review justify a specialized
non-Rust implementation. WASM is reserved as a quarantine and portability tier.
Admitted providers are persistent, weight-resident, and consume identity-bound compact
binary snapshot views; process startup and JSON parsing are not part of the normal hot
path.

See [ADR 0001](docs/decisions/0001-rust-provider-runtime.md) and the
[learned-provider runtime contract](docs/LEARNED_PROVIDER_RUNTIME.md).

### Placement and residency

Provider admission is persistent, but GPU residency is a separate policy decision.
`resident-on-admission` means the provider is loaded and reusable; it does not mean all
weights must remain permanently on a GPU. With sequential CPU offload, weights remain in
system RAM while individual modules execute temporarily on CUDA, trading VRAM for host
memory and transfer overhead. Explicit CPU/CUDA/offload choices are honored or rejected;
only `auto` may use bounded full-CUDA → sequential-offload → CPU recovery.

## Investigator roles

- **Investigator** — proposes falsifiable hypotheses and bounded interventions.
- **Skeptic** — searches for alternative explanations, omitted assumptions, and
  verifier gaps.
- **Replicator** — repeats frozen experiments across seeds, nodes, or providers.
- **Synthesizer** — proposes principles and reusable strategies from eligible
  attributions.
- **Auditor** — checks lineage, contamination, budgets, identity, and authority
  boundaries.

These are contracts, not trusted personalities. A role label does not grant execution
or acceptance authority.

## Verified Experience Distillation

VED is the consolidation stage inside MNEL:

```text
verified episodes
  -> evaluator-derived effects
  -> causal attribution
  -> provisional principle
  -> transfer tests
  -> supported strategy
  -> separately evaluated RAVEL candidate proposal
```

Distillation never deletes the raw evidence. A compact lesson retains references to
supporting attributions, known counterexamples, falsifiers, declared scope, transfer
state, and failure modes.

## Repository map

```text
src/mnel/                          Python control plane and executable foundation
crates/mnel-provider-api/          versioned provider ABI vocabulary
crates/mnel-provider-sdk/          safe Rust provider authoring surface
crates/mnel-provider-host/         persistent process-local host, policy, and snapshots
crates/mnel-provider-classical/    executable deterministic HMM diagnostic baseline
include/                           language-neutral provider ABI header
schemas/                           machine-readable record and runtime vocabulary
docs/                              architecture, decisions, method, boundaries, roadmap
examples/reference-study/          deterministic lifecycle example
examples/learned-providers/        architecture catalog and runtime manifest example
tests/                             lifecycle, integrity, registry, runtime, negative tests
.github/workflows/                 Python and Rust continuous verification
```

## Run the checks

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m mnel learned-provider list
python -m mnel demo --workspace build/demo
python -m mnel ledger verify build/demo/evidence.jsonl
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Safety and claim boundary

The current command adapters use argument arrays with `shell=False`, but MNEL is not an
operating-system sandbox. Untrusted experiment execution belongs in a hardened runner
with network restrictions, resource controls, immutable verifiers, and disposable
workspaces.

The provider runtime includes a small native-trusted dynamic loader with explicit ABI,
artifact, pointer/length, output, and diagnostic-authority checks. It is not an
operating-system sandbox: malformed native code can still crash the host process, so
untrusted experiment execution belongs in a hardened runner with network restrictions,
resource controls, immutable verifiers, and disposable workspaces.

A local MNEL result or learned-provider observation can describe bounded development
context. It cannot by itself establish independent evaluation, protected custody,
real-world safety, general recursive self-improvement, formal MNCS/MNCDS status,
certification, or promotion.

See [Architecture](docs/ARCHITECTURE.md), [Learning model](docs/LEARNING_MODEL.md),
[Learned micro-providers](docs/LEARNED_MICRO_PROVIDERS.md),
[learned-provider runtime](docs/LEARNED_PROVIDER_RUNTIME.md),
[Threat model](docs/THREAT_MODEL.md), and [Roadmap](docs/ROADMAP.md).
