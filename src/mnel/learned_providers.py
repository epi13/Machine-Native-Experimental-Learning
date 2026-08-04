"""Typed registry for diagnostic-only learned micro-providers.

Learned micro-providers locate surprise, disagreement, similarity, and likely omitted
questions. They never produce MNEL evaluator verdicts and cannot be made evaluator
eligible by configuration.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable

from .core import canonical_digest


class CostClass(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OutputKind(str, Enum):
    LATENT_DISCREPANCY = "latent-discrepancy"
    STRUCTURAL_DISCREPANCY = "structural-discrepancy"
    ANOMALY_SCORE = "anomaly-score"
    PAIR_SIMILARITY = "pair-similarity"
    NEXT_STATE_DISTRIBUTION = "next-state-distribution"
    FEATURE_CONTRIBUTIONS = "feature-contributions"
    CANDIDATE_RANKING = "candidate-ranking"


_COST_RANK = {CostClass.LOW: 0, CostClass.MEDIUM: 1, CostClass.HIGH: 2}
_PROVIDER_ID = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")


def _unique_nonempty(values: tuple[str, ...], name: str) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty values")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class CapacityBudget:
    """Approximate learned-scalar budget, including non-neural model parameters."""

    minimum: int
    target: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 1 or not self.minimum <= self.target <= self.maximum:
            raise ValueError("capacity budget must satisfy 1 <= minimum <= target <= maximum")


@dataclass(frozen=True)
class LearnedProviderDeclaration:
    provider_id: str
    version: str
    architecture_family: str
    architecture_name: str
    objective_family: str
    purpose: str
    rationale: str
    training_objective: str
    input_views: tuple[str, ...]
    artifact_types: tuple[str, ...]
    uncertainty_classes: tuple[str, ...]
    output_kinds: tuple[OutputKind, ...]
    required_snapshot_types: tuple[str, ...]
    advantages: tuple[str, ...]
    limitations: tuple[str, ...]
    capacity: CapacityBudget
    cost: CostClass
    cpu_appropriate: bool = True
    calibration_required: bool = True
    evaluator_eligible: bool = False
    authority: str = field(default="diagnostic-only", init=False)

    def __post_init__(self) -> None:
        if not _PROVIDER_ID.fullmatch(self.provider_id):
            raise ValueError(f"invalid provider_id: {self.provider_id!r}")
        for name in (
            "version",
            "architecture_family",
            "architecture_name",
            "objective_family",
            "purpose",
            "rationale",
            "training_objective",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        for name in (
            "input_views",
            "artifact_types",
            "uncertainty_classes",
            "required_snapshot_types",
            "advantages",
            "limitations",
        ):
            _unique_nonempty(getattr(self, name), name)
        if not self.output_kinds or len(set(self.output_kinds)) != len(self.output_kinds):
            raise ValueError("output_kinds must be non-empty and unique")
        if self.evaluator_eligible:
            raise ValueError("learned micro-providers cannot be evaluator eligible")

    @property
    def declaration_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-learned-provider-declaration/0.1",
            "provider_id": self.provider_id,
            "version": self.version,
            "architecture_family": self.architecture_family,
            "architecture_name": self.architecture_name,
            "objective_family": self.objective_family,
            "purpose": self.purpose,
            "rationale": self.rationale,
            "training_objective": self.training_objective,
            "input_views": list(self.input_views),
            "artifact_types": list(self.artifact_types),
            "uncertainty_classes": list(self.uncertainty_classes),
            "output_kinds": [item.value for item in self.output_kinds],
            "required_snapshot_types": list(self.required_snapshot_types),
            "advantages": list(self.advantages),
            "limitations": list(self.limitations),
            "capacity": asdict(self.capacity),
            "cost": self.cost.value,
            "cpu_appropriate": self.cpu_appropriate,
            "calibration_required": self.calibration_required,
            "evaluator_eligible": self.evaluator_eligible,
            "authority": self.authority,
        }
        if include_identity:
            value["declaration_identity"] = self.declaration_identity
        return value


@dataclass(frozen=True)
class LearnedProviderQuery:
    uncertainty_classes: tuple[str, ...]
    artifact_types: tuple[str, ...] = ()
    available_snapshot_types: tuple[str, ...] = ()
    required_output_kinds: tuple[OutputKind, ...] = ()
    preferred_architecture_families: tuple[str, ...] = ()
    excluded_provider_ids: tuple[str, ...] = ()
    max_cost: CostClass = CostClass.HIGH

    def __post_init__(self) -> None:
        _unique_nonempty(self.uncertainty_classes, "uncertainty_classes")
        for name in (
            "artifact_types",
            "available_snapshot_types",
            "preferred_architecture_families",
            "excluded_provider_ids",
        ):
            values = getattr(self, name)
            if values:
                _unique_nonempty(values, name)
        if len(set(self.required_output_kinds)) != len(self.required_output_kinds):
            raise ValueError("required_output_kinds must be unique")


@dataclass(frozen=True)
class ProviderMatch:
    declaration: LearnedProviderDeclaration
    score: int
    matched_uncertainty_classes: tuple[str, ...]
    matched_artifact_types: tuple[str, ...]
    matched_output_kinds: tuple[OutputKind, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.declaration.to_dict(),
            "score": self.score,
            "matched_uncertainty_classes": list(self.matched_uncertainty_classes),
            "matched_artifact_types": list(self.matched_artifact_types),
            "matched_output_kinds": [item.value for item in self.matched_output_kinds],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class LearnedProviderObservation:
    """Diagnostic output record. This deliberately has no verdict field."""

    provider_id: str
    provider_version: str
    declaration_identity: str
    model_identity: str
    feature_extractor_identity: str
    query_identity: str
    snapshot_ids: tuple[str, ...]
    output_kind: OutputKind
    value: float
    calibration_band: str
    out_of_distribution: bool
    suggested_uncertainty_classes: tuple[str, ...]
    candidate_locations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    authority: str = field(default="diagnostic-only", init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("observation value must be finite")
        if not self.calibration_band.strip():
            raise ValueError("calibration_band must not be empty")
        _unique_nonempty(self.snapshot_ids, "snapshot_ids")
        _unique_nonempty(self.suggested_uncertainty_classes, "suggested_uncertainty_classes")
        if self.candidate_locations:
            _unique_nonempty(self.candidate_locations, "candidate_locations")
        if self.limitations:
            _unique_nonempty(self.limitations, "limitations")

    def to_dict(self) -> dict[str, object]:
        value = {
            "schema": "mnel-learned-provider-observation/0.1",
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "declaration_identity": self.declaration_identity,
            "model_identity": self.model_identity,
            "feature_extractor_identity": self.feature_extractor_identity,
            "query_identity": self.query_identity,
            "snapshot_ids": list(self.snapshot_ids),
            "output_kind": self.output_kind.value,
            "value": self.value,
            "calibration_band": self.calibration_band,
            "out_of_distribution": self.out_of_distribution,
            "suggested_uncertainty_classes": list(self.suggested_uncertainty_classes),
            "candidate_locations": list(self.candidate_locations),
            "limitations": list(self.limitations),
            "authority": self.authority,
            "verdict_semantics": "not-a-verdict",
        }
        value["observation_identity"] = canonical_digest(value)
        return value


class LearnedProviderRegistry:
    """Immutable declaration registry with deterministic matching and diversity selection."""

    def __init__(self, declarations: Iterable[LearnedProviderDeclaration]) -> None:
        items = tuple(declarations)
        if not items:
            raise ValueError("registry requires at least one declaration")
        ids = [item.provider_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("provider IDs must be unique")
        self._items = tuple(sorted(items, key=lambda item: item.provider_id))
        self._by_id = {item.provider_id: item for item in self._items}

    def list(self) -> tuple[LearnedProviderDeclaration, ...]:
        return self._items

    def describe(self, provider_id: str) -> LearnedProviderDeclaration:
        try:
            return self._by_id[provider_id]
        except KeyError as error:
            raise KeyError(f"unknown learned provider: {provider_id}") from error

    def match(self, query: LearnedProviderQuery) -> tuple[ProviderMatch, ...]:
        matches: list[ProviderMatch] = []
        query_uncertainty = set(query.uncertainty_classes)
        query_artifacts = set(query.artifact_types)
        snapshots = set(query.available_snapshot_types)
        required_outputs = set(query.required_output_kinds)
        preferred = set(query.preferred_architecture_families)
        excluded = set(query.excluded_provider_ids)

        for declaration in self._items:
            if declaration.provider_id in excluded:
                continue
            if _COST_RANK[declaration.cost] > _COST_RANK[query.max_cost]:
                continue
            matched_uncertainty = tuple(sorted(query_uncertainty.intersection(declaration.uncertainty_classes)))
            if not matched_uncertainty:
                continue
            matched_artifacts = tuple(sorted(query_artifacts.intersection(declaration.artifact_types)))
            if query_artifacts and not matched_artifacts:
                continue
            provider_outputs = set(declaration.output_kinds)
            matched_outputs = tuple(sorted(required_outputs.intersection(provider_outputs), key=lambda item: item.value))
            if required_outputs and not required_outputs.issubset(provider_outputs):
                continue
            required_snapshots = set(declaration.required_snapshot_types)
            if snapshots and not required_snapshots.issubset(snapshots):
                continue

            score = 10 * len(matched_uncertainty)
            score += 4 * len(matched_artifacts)
            score += 3 * len(matched_outputs)
            score += 3 if declaration.architecture_family in preferred else 0
            score += 2 - _COST_RANK[declaration.cost]
            reasons = [f"matched {len(matched_uncertainty)} uncertainty class(es)"]
            if matched_artifacts:
                reasons.append(f"matched {len(matched_artifacts)} artifact type(s)")
            if matched_outputs:
                reasons.append(f"supports {len(matched_outputs)} required output kind(s)")
            if declaration.architecture_family in preferred:
                reasons.append("preferred architecture family")
            if not snapshots and required_snapshots:
                reasons.append("snapshot availability not supplied; discovery match only")
            matches.append(
                ProviderMatch(
                    declaration=declaration,
                    score=score,
                    matched_uncertainty_classes=matched_uncertainty,
                    matched_artifact_types=matched_artifacts,
                    matched_output_kinds=matched_outputs,
                    reasons=tuple(reasons),
                )
            )

        return tuple(
            sorted(
                matches,
                key=lambda item: (
                    -item.score,
                    _COST_RANK[item.declaration.cost],
                    item.declaration.provider_id,
                ),
            )
        )

    def select_diverse(self, query: LearnedProviderQuery, limit: int) -> tuple[ProviderMatch, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        remaining = list(self.match(query))
        selected: list[ProviderMatch] = []
        families: set[str] = set()
        objectives: set[str] = set()
        views: set[str] = set()

        while remaining and len(selected) < limit:
            def diversity_key(match: ProviderMatch) -> tuple[int, int, int, str]:
                declaration = match.declaration
                novelty = 12 if declaration.architecture_family not in families else 0
                novelty += 8 if declaration.objective_family not in objectives else 0
                novelty += len(set(declaration.input_views).difference(views))
                return (novelty, match.score, -_COST_RANK[declaration.cost], declaration.provider_id)

            chosen = max(remaining, key=diversity_key)
            remaining.remove(chosen)
            selected.append(chosen)
            families.add(chosen.declaration.architecture_family)
            objectives.add(chosen.declaration.objective_family)
            views.update(chosen.declaration.input_views)
        return tuple(selected)


def _declaration(
    provider_id: str,
    architecture_family: str,
    architecture_name: str,
    objective_family: str,
    purpose: str,
    rationale: str,
    training_objective: str,
    input_views: tuple[str, ...],
    artifact_types: tuple[str, ...],
    uncertainty_classes: tuple[str, ...],
    output_kinds: tuple[OutputKind, ...],
    required_snapshot_types: tuple[str, ...],
    advantages: tuple[str, ...],
    limitations: tuple[str, ...],
    capacity: tuple[int, int, int],
    cost: CostClass,
) -> LearnedProviderDeclaration:
    return LearnedProviderDeclaration(
        provider_id=provider_id,
        version="0.1.0",
        architecture_family=architecture_family,
        architecture_name=architecture_name,
        objective_family=objective_family,
        purpose=purpose,
        rationale=rationale,
        training_objective=training_objective,
        input_views=input_views,
        artifact_types=artifact_types,
        uncertainty_classes=uncertainty_classes,
        output_kinds=output_kinds,
        required_snapshot_types=required_snapshot_types,
        advantages=advantages,
        limitations=limitations,
        capacity=CapacityBudget(*capacity),
        cost=cost,
    )


DEFAULT_LEARNED_PROVIDERS = (
    _declaration(
        "latent.transition-jepa",
        "transition-jepa",
        "Joint-embedding predictor over parent, intervention, and result states",
        "latent-prediction",
        "Find candidate changes whose observed effects differ from expected transition semantics.",
        "A transition objective directly models the MNEL unit of experience instead of next-token text.",
        "Predict the target-state embedding from parent-state and intervention embeddings.",
        ("candidate-transition", "verifier-vector", "runtime-summary"),
        ("candidate-transition", "experiment-record", "verifier-result-vector"),
        ("unexpected-transition", "omitted-effect", "invariant-drift", "causal-mismatch"),
        (OutputKind.LATENT_DISCREPANCY, OutputKind.CANDIDATE_RANKING),
        ("transition-feature-snapshot",),
        ("directly aligned with MNEL episodes", "small predictor head", "natural discrepancy signal"),
        ("can normalize unseen failures", "depends on representative transitions", "not causal evidence"),
        (100_000, 1_000_000, 5_000_000),
        CostClass.LOW,
    ),
    _declaration(
        "latent.graph-jepa",
        "graph-jepa",
        "Masked subgraph joint-embedding predictor",
        "latent-prediction",
        "Find missing, surprising, or semantically inconsistent graph regions.",
        "Masked subgraph prediction can expose structural gaps without reconstructing every source token.",
        "Predict a masked AST, CFG, use-def, or dependency subgraph embedding from its context.",
        ("ast-graph", "control-flow-graph", "use-def-graph", "dependency-graph"),
        ("ast", "cfg", "code-property-graph", "dependency-graph"),
        ("missing-relation", "structural-inconsistency", "cross-module-drift", "omitted-effect"),
        (OutputKind.STRUCTURAL_DISCREPANCY, OutputKind.CANDIDATE_RANKING),
        ("bounded-graph-snapshot",),
        ("captures topology", "supports masked-region questions", "works with bounded graph slices"),
        ("graph construction can dominate cost", "sensitive to extractor identity", "weak on temporal order"),
        (250_000, 2_000_000, 8_000_000),
        CostClass.MEDIUM,
    ),
    _declaration(
        "graph.message-passing-gnn",
        "message-passing-gnn",
        "Compact typed-edge message-passing graph network",
        "supervised-graph-prediction",
        "Propagate ownership, dependency, contract, and data-flow signals over bounded graph neighborhoods.",
        "Explicit edge types are useful when the diagnostic question is relational rather than generative.",
        "Predict bounded node, edge, or graph labels from typed message passing.",
        ("typed-graph", "ownership-graph", "contract-graph", "data-flow-graph"),
        ("cfg", "code-property-graph", "contract-graph", "dependency-graph"),
        ("ownership-transition", "dependency-envelope-incomplete", "contract-drift", "reachability-risk"),
        (OutputKind.ANOMALY_SCORE, OutputKind.FEATURE_CONTRIBUTIONS),
        ("typed-graph-snapshot",),
        ("relational inductive bias", "compact local inference", "edge-type ablations are inspectable"),
        ("oversmoothing at depth", "bounded neighborhood may omit long paths", "labels can encode verifier bias"),
        (100_000, 1_000_000, 5_000_000),
        CostClass.LOW,
    ),
    _declaration(
        "sequence.selective-state-space",
        "selective-state-space",
        "Compact selective state-space sequence model",
        "sequence-prediction",
        "Model long ordered compiler, tool, runtime, and verifier event streams.",
        "Selective state updates can retain relevant events across long traces with bounded memory.",
        "Predict future event-state embeddings and score deviations from expected temporal dynamics.",
        ("diagnostic-event-stream", "runtime-trace", "tool-call-sequence"),
        ("runtime-trace", "diagnostic-session", "tool-event-stream"),
        ("temporal-anomaly", "state-transition-drift", "initialization-order", "delayed-effect"),
        (OutputKind.NEXT_STATE_DISTRIBUTION, OutputKind.ANOMALY_SCORE),
        ("ordered-event-snapshot",),
        ("linear sequence scaling", "long effective context", "bounded recurrent state"),
        ("order-sensitive preprocessing", "state can hide attribution", "training is less mature than simple baselines"),
        (250_000, 2_000_000, 8_000_000),
        CostClass.MEDIUM,
    ),
    _declaration(
        "sequence.dilated-temporal-conv",
        "dilated-temporal-convolution",
        "Dilated causal temporal convolution network",
        "sequence-prediction",
        "Detect repeated local-to-midrange motifs in traces and diagnostic event windows.",
        "Dilated convolutions provide fast CPU inference and stable receptive fields for small models.",
        "Predict event classes or anomaly scores from causal multi-scale convolutional features.",
        ("runtime-trace", "compiler-event-window", "metric-series"),
        ("runtime-trace", "diagnostic-session", "metric-series"),
        ("temporal-anomaly", "repeated-failure-motif", "resource-regression", "phase-drift"),
        (OutputKind.ANOMALY_SCORE, OutputKind.NEXT_STATE_DISTRIBUTION),
        ("ordered-event-snapshot",),
        ("very low startup cost", "parallel CPU inference", "clear receptive-field budget"),
        ("fixed receptive field", "weak on sparse very-long dependencies", "padding choices can leak phase"),
        (50_000, 500_000, 3_000_000),
        CostClass.LOW,
    ),
    _declaration(
        "pair.contrastive-siamese",
        "contrastive-siamese",
        "Shared-weight candidate/reference encoder",
        "metric-learning",
        "Estimate whether two bounded artifacts or transitions are behaviorally equivalent or meaningfully different.",
        "A shared encoder directly supports candidate/reference, before/after, and mutation-pair comparisons.",
        "Pull verified-equivalent pairs together and push verified-different pairs apart.",
        ("candidate-pair", "mutation-pair", "reference-pair", "trace-pair"),
        ("source-region-pair", "ir-pair", "candidate-transition", "runtime-trace-pair"),
        ("semantic-drift", "unexpected-equivalence", "unexpected-difference", "duplicate-mechanism"),
        (OutputKind.PAIR_SIMILARITY, OutputKind.ANOMALY_SCORE),
        ("paired-artifact-snapshot",),
        ("natural differential interface", "shared weights reduce size", "useful for mutation studies"),
        ("pair sampling controls behavior", "similarity is not equivalence proof", "hard negatives may be contaminated"),
        (100_000, 1_000_000, 5_000_000),
        CostClass.LOW,
    ),
    _declaration(
        "anomaly.deep-svdd",
        "deep-svdd",
        "One-class compact representation model",
        "one-class-anomaly",
        "Flag states outside the learned region of known eligible behavior.",
        "One-class training is useful when verified failures are sparse but normal episodes are abundant.",
        "Minimize distance of eligible normal states to a learned hypersphere center.",
        ("verifier-vector", "experiment-metrics", "state-summary"),
        ("verifier-result-vector", "experiment-record", "metric-vector"),
        ("out-of-distribution", "novel-failure-family", "metric-inconsistency", "unexpected-transition"),
        (OutputKind.ANOMALY_SCORE,),
        ("tabular-feature-snapshot",),
        ("small and CPU-friendly", "does not require failure labels", "clear anomaly baseline"),
        ("normal-data contamination is damaging", "distance may not track importance", "poor localization alone"),
        (10_000, 250_000, 2_000_000),
        CostClass.LOW,
    ),
    _declaration(
        "anomaly.denoising-autoencoder",
        "denoising-autoencoder",
        "Compact denoising reconstruction model",
        "reconstruction",
        "Detect missing, corrupted, or internally inconsistent feature combinations.",
        "Denoising objectives test whether a state can be reconstructed from redundant cross-view evidence.",
        "Reconstruct masked or corrupted structured features and score residual patterns.",
        ("multi-view-feature-vector", "verifier-vector", "metric-vector"),
        ("verifier-result-vector", "experiment-record", "diagnostic-event"),
        ("missing-evidence", "cross-view-inconsistency", "feature-corruption", "observability-loss"),
        (OutputKind.ANOMALY_SCORE, OutputKind.FEATURE_CONTRIBUTIONS),
        ("tabular-feature-snapshot",),
        ("cheap training", "feature-level residuals aid localization", "works with partially missing inputs"),
        ("can learn identity shortcuts", "reconstruction quality is not semantic correctness", "scaling choices dominate"),
        (10_000, 250_000, 2_000_000),
        CostClass.LOW,
    ),
    _declaration(
        "interaction.tiny-transformer",
        "tiny-transformer",
        "Small attention encoder over bounded heterogeneous evidence tokens",
        "masked-and-supervised-interaction",
        "Model sparse nonlocal interactions among a small set of evidence items.",
        "Attention is useful when any bounded evidence item may directly interact with any other.",
        "Predict masked evidence roles, discrepancy classes, or rankings over bounded evidence sets.",
        ("evidence-token-set", "bounded-ir-token-sequence", "cross-view-record-set"),
        ("diagnostic-session", "ir-slice", "experiment-record-set"),
        ("nonlocal-interaction", "cross-view-inconsistency", "omitted-assumption", "evidence-conflict"),
        (OutputKind.ANOMALY_SCORE, OutputKind.CANDIDATE_RANKING, OutputKind.FEATURE_CONTRIBUTIONS),
        ("bounded-evidence-snapshot",),
        ("direct cross-item interaction", "flexible mixed evidence", "strong comparison baseline"),
        ("quadratic in item count", "easy to overfit small data", "attention is not explanation"),
        (250_000, 3_000_000, 12_000_000),
        CostClass.HIGH,
    ),
    _declaration(
        "tabular.gradient-boosted-trees",
        "gradient-boosted-trees",
        "Small boosted decision-tree ensemble",
        "supervised-tabular",
        "Provide an interpretable non-neural baseline for structured verifier and experiment features.",
        "Tree ensembles often work well on low-volume heterogeneous tabular data and expose feature contributions.",
        "Fit shallow additive trees to calibrated diagnostic labels or rankings.",
        ("verifier-vector", "experiment-metrics", "routing-features"),
        ("verifier-result-vector", "metric-vector", "experiment-record"),
        ("regression-risk", "routing-mismatch", "metric-inconsistency", "failure-family-ranking"),
        (OutputKind.ANOMALY_SCORE, OutputKind.CANDIDATE_RANKING, OutputKind.FEATURE_CONTRIBUTIONS),
        ("tabular-feature-snapshot",),
        ("strong small-data baseline", "fast CPU inference", "feature contributions are available"),
        ("cannot consume raw graphs or long sequences", "threshold artifacts", "supervised labels may be scarce"),
        (1_000, 50_000, 500_000),
        CostClass.LOW,
    ),
    _declaration(
        "state.hidden-markov-model",
        "hidden-markov-model",
        "Discrete or Gaussian hidden-state transition model",
        "probabilistic-state-modeling",
        "Detect illegal, unlikely, or skipped phases in small diagnostic state machines.",
        "Explicit latent states and transition probabilities are appropriate for bounded lifecycle protocols.",
        "Estimate hidden states and transition likelihoods over normalized event sequences.",
        ("lifecycle-events", "tool-state-sequence", "experiment-phase-sequence"),
        ("diagnostic-session", "experiment-record", "tool-event-stream"),
        ("illegal-transition", "skipped-phase", "stale-state", "protocol-drift"),
        (OutputKind.NEXT_STATE_DISTRIBUTION, OutputKind.ANOMALY_SCORE),
        ("ordered-event-snapshot",),
        ("very small model", "explicit transition matrix", "useful likelihood baseline"),
        ("Markov assumptions are restrictive", "state-count selection matters", "weak on rich observations"),
        (100, 5_000, 100_000),
        CostClass.LOW,
    ),
    _declaration(
        "sequence.reservoir-computer",
        "reservoir-computing",
        "Echo-state reservoir with trained readout",
        "fixed-dynamics-sequence-learning",
        "Test whether cheap fixed recurrent dynamics can expose temporal differences before training larger sequence models.",
        "Only the readout is trained, making this an inexpensive temporal baseline for local hardware.",
        "Project event streams through a fixed stable reservoir and train a small diagnostic readout.",
        ("runtime-trace", "tool-call-sequence", "metric-series"),
        ("runtime-trace", "diagnostic-session", "metric-series"),
        ("temporal-anomaly", "phase-drift", "delayed-effect", "repeated-failure-motif"),
        (OutputKind.ANOMALY_SCORE, OutputKind.NEXT_STATE_DISTRIBUTION),
        ("ordered-event-snapshot",),
        ("minimal training cost", "CPU-friendly", "useful control against learned recurrent weights"),
        ("reservoir hyperparameters are brittle", "limited task adaptation", "state is difficult to interpret"),
        (1_000, 100_000, 1_000_000),
        CostClass.LOW,
    ),
)

DEFAULT_LEARNED_PROVIDER_REGISTRY = LearnedProviderRegistry(DEFAULT_LEARNED_PROVIDERS)


__all__ = [
    "CapacityBudget",
    "CostClass",
    "DEFAULT_LEARNED_PROVIDERS",
    "DEFAULT_LEARNED_PROVIDER_REGISTRY",
    "LearnedProviderDeclaration",
    "LearnedProviderObservation",
    "LearnedProviderQuery",
    "LearnedProviderRegistry",
    "OutputKind",
    "ProviderMatch",
]
