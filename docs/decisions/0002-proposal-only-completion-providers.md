# ADR 0002: Proposal-only completion providers for Forge construction

- **Status:** Proposed
- **Date:** 2026-08-18
- **Scope:** MNEL learned provider portfolio, Forge construction integration, and future MNCS-language/HIR completion

## Context

MNEL already contains diagnostic-only learned micro-providers whose job is to locate surprise, anomaly, similarity, disagreement, and omitted questions. Forge, however, also needs a low-latency learned construction capability for local code generation: inline completion, infill, repair, proof completion, refactoring, optimization, and related bounded edits.

Treating these completion models as ordinary diagnostic providers would blur an important authority boundary. Diagnostic providers return observations. Completion providers return candidate artifact content that may be inserted into a program. The two classes may share runtime infrastructure, but they do not have the same result semantics.

The completion portfolio must also remain architecture-neutral. Early providers may be source-language-specific tiny transformers, but MNEL should be able to admit state-space, graph/HIR, recurrent, retrieval-conditioned, RAVEL-derived, and other learned mechanisms without changing the Forge caller.

## Decision

MNEL adopts a separate **proposal-only completion provider** capability class.

1. Forge requests a typed completion capability from MNEL rather than naming a concrete model.
2. MNEL performs deterministic compatibility filtering by operation, language, representation, evidence view, runtime tier, and resource budget.
3. A learned router may later rank only already-compatible providers. It may not widen eligibility or authority.
4. Completion provider output carries `authority: proposal-only` and `verdict_semantics: not-a-verdict`.
5. Forge, compilers, type systems, static analyzers, tests, proof systems, and other declared validators determine whether a proposed completion survives.
6. The current diagnostic provider ABI remains unchanged. Completion request/response semantics receive a separate versioned protocol beginning with `mnel-inline-completion/0.1`.
7. The persistent MNEL provider host may be reused for admission, model residency, placement, identity binding, measurement, and quarantine where the runtime contract is compatible.
8. Interactive completion defaults to persistent local inference. Fabric is primarily used for training, evaluation, calibration, batch studies, distribution, and prewarming unless measured evidence supports remote interactive inference.
9. Language specialization is a bootstrap strategy, not a permanent architectural assumption. As MNCS-language/HIR matures, MNEL should study semantic-operation specialists that consume and emit machine-native representations.
10. RAVEL-derived or otherwise exotic providers receive no special trust and enter through the same admission, transfer, calibration, budget, and rollback process.

## Authority boundary

A completion response may contain candidate content, ranking scores, calibration/OOD metadata, runtime measurements, and limitations. It may not claim:

- PASS or FAIL;
- verified or conformant;
- safe or correct;
- accepted or mergeable;
- proof completion success unless an external proof checker has produced that evidence;
- promotion authority.

A provider can propose a proof term. It cannot declare that proof valid.

## Consequences

### Positive

- Forge remains model-agnostic and can request capabilities instead of model names.
- MNEL can experiment with multiple tiny architectures without changing the construction API.
- Diagnostic and generative provider semantics remain separate and auditable.
- Completion produces dense compiler/type/test/proof feedback useful for MNEL learning studies.
- MNCS-language can gradually shift specialization from source language to semantic operation.
- Existing provider-host investment in persistent residency and resource accounting can be reused.

### Costs

- MNEL now has two learned provider authority classes with separate result contracts.
- A separate completion protocol and provider declaration vocabulary must be maintained.
- Large candidate payloads and streaming/infill semantics may require runtime work beyond diagnostic ABI v1.
- Routing studies require care to distinguish provider quality from validator strength and task difficulty.

## Non-goals

This decision does not select a particular model, tokenizer, inference engine, parameter count, or training recipe. It does not require one model per source language, require an LLM architecture, or make MNCS-language the exclusive completion representation before it is ready.

## Reconsideration

The separation between diagnostic-only and proposal-only providers may be revisited only if a future common provider protocol can preserve the authority distinction mechanically and without weakening existing diagnostic guarantees. The Forge caller must remain capability-oriented even if the underlying runtime is unified.
