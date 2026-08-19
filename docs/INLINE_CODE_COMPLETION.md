# Forge-callable inline code completion providers

## Purpose

MNEL should support a family of very small, frequently invoked code-generation providers that Forge can call for inline completion, infill, repair, proof completion, refactoring, optimization, and other local construction tasks.

These providers are not a single autocomplete model and are not assumed to be transformers. The capability is intentionally heterogeneous: different languages, representations, operations, and model architectures may justify different specialists. A provider may be a tiny transformer, a selective state-space model, a graph/HIR model, a recurrent or reservoir model, a retrieval-conditioned model, a RAVEL-derived experimental architecture, or another learned mechanism that satisfies the declared contract.

The architectural objective is:

```text
Forge construction request
        |
        v
MNEL completion capability boundary
        |
        v
deterministic compatibility filter
        |
        +--> language / representation
        +--> operation
        +--> evidence requirements
        +--> latency / memory / output budgets
        +--> declared limitations
        |
        v
optional evidence-trained ranking among eligible providers
        |
        v
one or more proposal-only completion providers
        |
        v
bounded candidate completions
        |
        v
Forge validation / compiler / types / tests / proofs / static analysis
        |
        v
accepted, rejected, or revised candidate evidence
        |
        v
MNEL experience records and later RAVEL/MNEL study
```

Forge owns the construction workflow and verification boundary. MNEL owns provider declaration, compatibility matching, runtime admission, placement, invocation evidence, and learned selection among already-eligible providers.

## Why this is a separate provider class

The existing learned micro-provider registry is deliberately `diagnostic-only`. Its providers locate surprise, disagreement, anomaly, similarity, and likely omitted questions. Inline code completion is different because it emits a candidate artifact that may be inserted into a working program.

MNEL must therefore not smuggle code generation through the diagnostic observation contract or reinterpret a diagnostic scalar/payload as an accepted edit.

Completion providers use a separate authority class:

- `authority: proposal-only`
- `verdict_semantics: not-a-verdict`
- no evaluator, conformance, acceptance, merge, promotion, or proof authority

A completion may become part of a candidate only through the external Forge workflow and its normal validation/evidence rules.

The existing provider runtime can be reused for residency, placement, resource accounting, artifact identity, quarantine, and measurement, but the completion request/response surface is versioned separately from the diagnostic ABI.

## Specialization dimensions

The first implementation may use one or more models per source language, but language is only one specialization axis.

### Language and representation

Initial source-language specialists may include:

- C/C++
- Rust
- Python
- Haskell
- shell/configuration languages
- MNCS-language as it becomes usable

Long term, MNCS-language and its semantic/HIR representation should reduce dependence on source-language-specific predictors. A preferred future path is:

```text
source language or MNCS-language
        |
        v
identified semantic/HIR context
        |
        v
operation-specialized MNEL provider
        |
        v
semantic/HIR candidate
        |
        v
language-specific lowering when required
```

The registry must therefore declare accepted context representations independently from source language. A provider may consume source tokens, AST, typed AST, CFG, SSA/use-def graph, proof state, MNCS HIR, or a bounded composite snapshot.

### Operation

Representative operation classes are:

- `inline-completion`
- `infill`
- `repair`
- `refactor`
- `proof-completion`
- `api-completion`
- `optimization`
- `control-flow-construction`
- `branch-mask-construction`
- `contract-satisfaction`
- `lowering`

MNEL should prefer operation specialists when evidence shows that they outperform a generic local code model under the same latency and verification budget.

### Architecture

Provider architecture is declared rather than assumed. Candidate families include:

- tiny decoder transformer;
- fill-in-the-middle transformer;
- selective state-space model;
- graph neural or graph-JEPA model over AST/CFG/HIR;
- recurrent/reservoir sequence model;
- retrieval-conditioned predictor;
- contrastive/ranking model paired with a deterministic enumerator;
- RAVEL-derived or other experimental machine-native architecture.

Architectural diversity is useful when it produces complementary verified success surfaces, not merely because an architecture is novel.

### Evidence profile

A completion provider may declare that it is designed to operate with specific external evidence channels, for example:

- type information;
- compiler diagnostics;
- proof goals;
- static-analysis facts;
- API signatures;
- execution traces;
- contracts and pre/postconditions;
- prior verified local edits.

The provider does not become an authority over those channels. They are bounded inputs or post-generation validators.

## Forge request contract

Forge should call an MNEL completion capability, not a hard-coded model name.

Conceptually:

```text
completion_request
  request_identity
  artifact_identity
  source_identity
  cursor_or_span
  source_language
  context_representation
  operation
  semantic_context_identity
  constraints
  required_evidence
  latency_budget_ms
  memory_budget_bytes
  output_budget_bytes
  max_candidates
```

A caller may optionally express architecture preferences or exclusions, but it must not be required to know which model is currently best.

MNEL performs deterministic eligibility filtering first. A future learned router may rank only the providers that passed that filter. Learned routing may not expand language support, representation access, authority, hidden-partition access, or resource ceilings.

## Provider declaration

A completion provider declaration should bind at least:

- provider ID and declaration version;
- exact architecture and objective family;
- supported operations;
- supported source languages;
- accepted context representations;
- emitted candidate representations;
- required snapshot/evidence types;
- maximum context and output sizes;
- warm-latency and memory class;
- runtime tier and placement compatibility;
- model/weight identity requirements;
- calibration or abstention contract when applicable;
- known limitations and unsupported constructs;
- proposal-only authority boundary.

Runtime invocation additionally binds exact weight/model, tokenizer or feature extractor, query, source/context snapshot, toolchain, calibration, and runtime identities.

## Candidate response contract

A completion response is a bounded proposal set, not a verdict. Each candidate records:

- candidate identity;
- provider and model identities;
- emitted representation kind;
- bounded candidate bytes or an identified artifact reference;
- finish/abstention/truncation state;
- optional calibrated ranking score;
- runtime measurements;
- limitations and OOD state;
- `authority: proposal-only`;
- `verdict_semantics: not-a-verdict`.

The response must not contain fields equivalent to `PASS`, `verified`, `conformant`, `safe`, `merge`, or `promote`.

## Verification loop

Inline completion is especially useful to MNEL because it creates a dense stream of relatively cheap feedback:

```text
identified context
  -> provider candidate
  -> user/agent acceptance or rejection
  -> parse
  -> compile/type check
  -> static analysis
  -> unit/property tests
  -> proof obligations where available
  -> subsequent edits or rollback
```

Human/agent acceptance is useful behavioral evidence but is not the only target. Compiler, type, test, proof, static-analysis, and downstream-edit outcomes provide stronger labels for many tasks.

MNEL should retain both successful and unsuccessful completions. Useful study metrics include:

- verified acceptance rate;
- first-candidate verified success;
- characters/tokens kept after subsequent edits;
- compile/type/proof success by operation and language;
- verifier failure family;
- abstention quality;
- p50/p95/p99 warm latency;
- queue delay;
- bytes copied and snapshot construction cost;
- resident memory and accelerator pressure;
- useful verified completions per joule or unit compute when measurable;
- correlation and disagreement between provider families.

This evidence can train or select later providers without granting them evaluator authority.

## Routing policy

Routing is a two-stage contract.

### Stage 1: deterministic capability gate

Reject any provider that does not satisfy the request's declared language, representation, operation, snapshot, authority, budget, or runtime constraints.

### Stage 2: evidence-based ranking

Among eligible providers, MNEL may rank using historical verified utility conditioned on features such as:

- operation;
- source language;
- semantic/HIR construct class;
- context size;
- available evidence views;
- latency budget;
- current device pressure;
- provider warm/cold state;
- recent calibration/OOD behavior.

The ranking model is advisory over an already-safe candidate set. It cannot make an ineligible provider eligible.

For ambiguous or high-value requests, MNEL may return a small diverse candidate set rather than a single provider output. Diversity selection should favor complementary architecture/representation families when the extra verification cost is justified.

## Runtime placement

Keystroke-level completion should not place Fabric network dispatch in the critical path by default. The intended fast path is a persistent local MNEL provider host with resident admitted models and reusable Forge/MNCS snapshots.

Fabric remains useful for:

- training and distillation;
- evaluation sweeps;
- calibration;
- batch completion studies;
- larger non-interactive generation;
- prewarming or distributing providers to appropriate nodes;
- collecting heterogeneous hardware evidence.

Interactive remote inference may be allowed only when measured latency and trust requirements make it worthwhile.

## Relationship to MNCS-language

MNCS-language is strategically important because the strongest long-term specialization may be by semantic operation rather than by human source language.

A mature portfolio could contain providers such as:

- `mncs.next-expression`
- `mncs.proof-complete`
- `mncs.control-flow`
- `mncs.memory-layout`
- `mncs.repair`
- `mncs.contract-satisfier`
- `mncs.optimize`
- `mncs.branch-mask`

Those providers may operate directly over proof-carrying semantic/HIR artifacts and emit machine-native structures that are subsequently checked and lowered. Source-language-specific models remain useful as bootstrap providers and compatibility edges.

MNEL must not assume this transition has already happened. Source-language and semantic/HIR providers should coexist and compete under evidence.

## Relationship to RAVEL

RAVEL is not required to serve every completion request. Its role is more valuable as an adaptive source of provider architectures, routing strategies, retained local patterns, or learned mechanisms that can be admitted into the same proposal-only contract.

A RAVEL-derived provider receives no special trust. It must satisfy the same identity, budget, calibration, transfer, rollback, and verification requirements as any other provider.

## Initial delivery sequence

1. Freeze a durable `mnel-inline-completion/0.1` request/response schema and proposal-only authority semantics.
2. Add completion-provider declarations and deterministic compatibility matching without learned routing.
3. Reuse the persistent provider host's residency, placement, identity, measurement, and quarantine mechanisms while keeping completion protocol semantics separate from diagnostic ABI v1.
4. Implement one small source-language baseline and one structurally different baseline for the same bounded task.
5. Add a Forge adapter that requests the MNEL capability rather than naming a model.
6. Record parser/compiler/type/test/proof outcomes as completion evidence.
7. Compare generic versus language-specialized and operation-specialized providers under equal budgets.
8. Add evidence-based ranking only after deterministic matching and explicit controls are measured.
9. Add MNCS-language/HIR specialists as the language representation stabilizes.
10. Admit RAVEL-derived or other exotic provider architectures only through the same study and rollback process.

## Non-goals

This design does not:

- make model output authoritative;
- replace Forge verification;
- require one model per language forever;
- require every provider to be an LLM;
- require Fabric for interactive completion;
- treat user acceptance as proof of correctness;
- allow a learned router to widen capability or policy;
- couple Forge to a specific model artifact;
- declare MNCS-language sufficiently mature for exclusive use today.

The central rule is simple: **Forge asks for a bounded completion capability; MNEL chooses among eligible learned providers; external evidence decides what survives.**
