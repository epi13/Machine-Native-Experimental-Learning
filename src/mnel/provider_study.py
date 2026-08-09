"""Bounded heterogeneous learned-provider studies.

This module is a small control-plane harness for comparing identified diagnostic
providers. It does not train a general ML framework, vote observations into truth, or
grant evaluator authority. Providers are supplied as explicit bindings and every study
result retains provider, snapshot, calibration, routing, and resource identities.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .core import EvidenceLedger, Visibility, canonical_digest, canonical_json
from .distillation import (
    AUTHORITY_DIAGNOSTIC_ONLY,
    CalibrationDataAccess,
    CalibrationRecord,
    DistillationError,
    StudyDataAccess,
    VisibilityViolation,
    calculate_calibration,
    make_study_record,
    _reject_authority,
)
from .reference_provider import (
    LearnedProviderObservation,
    TabularCentroidModel,
    TransitionFrequencyModel,
    train_tabular_centroid,
    train_transition_frequency,
)
from .snapshots import (
    DiagnosticSnapshot,
    SnapshotStore,
    TabularView,
    TransitionView,
    decode_snapshot,
    tabular_snapshot,
    transition_snapshot,
)


MAX_PORTFOLIO_PROVIDERS = 16
MAX_PORTFOLIO_CASES = 64
MAX_WARM_SAMPLES = 32


class ProviderStudyError(ValueError):
    pass


class RoutingPolicyKind(StrEnum):
    RANDOM = "random"
    HEURISTIC = "heuristic"
    SINGLE_PROVIDER = "single-provider"
    DIVERSITY = "diversity"


class ProviderRunStatus(StrEnum):
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    OOD = "out-of-distribution"
    INCOMPATIBLE = "incompatible"
    MALFORMED = "malformed-input"
    ERROR = "runtime-error"


class ProviderLifecycleState(StrEnum):
    CANDIDATE = "candidate"
    DEVELOPMENT_ADMITTED = "development-admitted"
    TRANSFER_PENDING = "transfer-pending"
    TRANSFER_ADMITTED = "transfer-admitted"
    QUARANTINED = "quarantined"
    RETIRED = "retired"
    ROLLED_BACK = "rolled-back"


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ProviderStudyError(f"{label} must be finite")
    return value


def _quantile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class ProviderLike(Protocol):
    provider_id: str
    model_identity: str
    artifact_identity: str
    calibration_identity: str
    model_size_bytes: int
    provider_family: str
    supported_snapshot_types: tuple[str, ...]

    def serialize(self) -> bytes: ...

    def infer(self, snapshot: DiagnosticSnapshot) -> LearnedProviderObservation: ...


@dataclass(frozen=True, slots=True)
class ProviderArtifactMetadata:
    provider_id: str
    provider_family: str
    architecture_family: str
    objective_family: str
    supported_snapshot_types: tuple[str, ...]
    training_dataset_identity: str
    training_record_ids: tuple[str, ...]
    calibration_dataset_identity: str
    feature_extractor_identity: str
    training_code_identity: str
    calibration_identity: str
    model_identity: str
    artifact_identity: str
    model_size_bytes: int
    artifact_size_bytes: int
    kind: str = "learned-provider"
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        if not all(
            (
                self.provider_id,
                self.provider_family,
                self.architecture_family,
                self.objective_family,
                self.training_dataset_identity,
                self.calibration_dataset_identity,
                self.feature_extractor_identity,
                self.training_code_identity,
                self.calibration_identity,
                self.model_identity,
                self.artifact_identity,
            )
        ):
            raise ProviderStudyError("provider artifact metadata is missing an identity")
        if not self.supported_snapshot_types or not self.training_record_ids:
            raise ProviderStudyError("provider artifact metadata requires input and lineage")
        if self.model_size_bytes < 1 or self.artifact_size_bytes < 1:
            raise ProviderStudyError("provider artifact sizes must be positive")
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise ProviderStudyError("provider artifact metadata is diagnostic-only")

    @property
    def metadata_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-artifact-metadata/0.4",
            "provider_id": self.provider_id,
            "provider_family": self.provider_family,
            "architecture_family": self.architecture_family,
            "objective_family": self.objective_family,
            "supported_snapshot_types": list(self.supported_snapshot_types),
            "training_dataset_identity": self.training_dataset_identity,
            "training_record_ids": list(self.training_record_ids),
            "calibration_dataset_identity": self.calibration_dataset_identity,
            "feature_extractor_identity": self.feature_extractor_identity,
            "training_code_identity": self.training_code_identity,
            "calibration_identity": self.calibration_identity,
            "model_identity": self.model_identity,
            "artifact_identity": self.artifact_identity,
            "model_size_bytes": self.model_size_bytes,
            "artifact_size_bytes": self.artifact_size_bytes,
            "kind": self.kind,
            "authority": self.authority,
            "semantics": "provider-metadata; diagnostic-only; not-a-verdict",
        }
        if include_identity:
            value["metadata_identity"] = self.metadata_identity
        return value


@dataclass(frozen=True, slots=True)
class ProviderBinding:
    """An explicit executable provider or baseline binding."""

    provider: ProviderLike
    loader: Callable[[bytes], ProviderLike]
    architecture_family: str
    objective_family: str
    cost_class: str
    kind: str = "learned-provider"

    def __post_init__(self) -> None:
        if not self.provider.provider_id or not self.provider.supported_snapshot_types:
            raise ProviderStudyError("provider binding lacks identity or input views")
        if self.kind not in {"learned-provider", "baseline"}:
            raise ProviderStudyError("provider binding kind is invalid")

    @property
    def provider_id(self) -> str:
        return self.provider.provider_id

    @property
    def metadata(self) -> ProviderArtifactMetadata:
        artifact = self.provider.serialize()
        return ProviderArtifactMetadata(
            self.provider.provider_id,
            self.provider.provider_family,
            self.architecture_family,
            self.objective_family,
            tuple(self.provider.supported_snapshot_types),
            getattr(self.provider, "training_dataset_identity", "control-dataset"),
            tuple(getattr(self.provider, "training_record_ids", (self.provider.provider_id,))),
            getattr(self.provider, "calibration_dataset_identity", "control-calibration"),
            getattr(self.provider, "feature_extractor_identity", "control-features"),
            getattr(self.provider, "training_code_identity", "control-code"),
            self.provider.calibration_identity,
            self.provider.model_identity,
            self.provider.artifact_identity or "sha256:" + hashlib.sha256(artifact).hexdigest(),
            int(self.provider.model_size_bytes),
            len(artifact),
            self.kind,
        )


@dataclass(frozen=True, slots=True)
class PortfolioCase:
    case_id: str
    snapshot: DiagnosticSnapshot
    expected_label: int | None
    expected_ood: bool
    case_kind: str
    reference_evidence_identity: str
    reference_supports_usefulness: bool = True

    def __post_init__(self) -> None:
        if not self.case_id or self.expected_label not in (None, 0, 1, False, True):
            raise ProviderStudyError("portfolio case identity or label is invalid")
        if not self.reference_evidence_identity:
            raise ProviderStudyError("portfolio case requires reference evidence identity")

    @property
    def case_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-study-case/0.4",
            "case_id": self.case_id,
            "snapshot_identity": self.snapshot.snapshot_identity,
            "snapshot_type": self.snapshot.snapshot_type,
            "expected_label": self.expected_label,
            "expected_ood": self.expected_ood,
            "case_kind": self.case_kind,
            "reference_evidence_identity": self.reference_evidence_identity,
            "reference_supports_usefulness": self.reference_supports_usefulness,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "reference-case; not-a-verdict",
        }
        if include_identity:
            value["case_identity"] = self.case_identity
        return value


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    policy_id: str
    kind: RoutingPolicyKind
    candidate_provider_ids: tuple[str, ...]
    max_providers: int
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        if not self.policy_id or not self.candidate_provider_ids:
            raise ProviderStudyError("routing policy requires providers")
        if len(set(self.candidate_provider_ids)) != len(self.candidate_provider_ids):
            raise ProviderStudyError("routing provider IDs must be unique")
        if self.max_providers < 1 or self.max_providers > MAX_PORTFOLIO_PROVIDERS or self.seed < 0:
            raise ProviderStudyError("routing policy bounds are invalid")
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise ProviderStudyError("routing policy is diagnostic-only")
        _reject_authority(self.parameters)

    @property
    def policy_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-routing-policy/0.4",
            "policy_id": self.policy_id,
            "kind": self.kind.value,
            "candidate_provider_ids": list(self.candidate_provider_ids),
            "max_providers": self.max_providers,
            "seed": self.seed,
            "parameters": dict(self.parameters or {}),
            "authority": self.authority,
            "semantics": "routing-control; not-a-verdict",
        }
        if include_identity:
            value["policy_identity"] = self.policy_identity
        return value


@dataclass(frozen=True, slots=True)
class RoutingSelection:
    policy_identity: str
    case_identity: str
    candidate_provider_ids: tuple[str, ...]
    selected_provider_ids: tuple[str, ...]
    status: str
    reason: str
    selection_identity: str = ""

    def __post_init__(self) -> None:
        if not self.policy_identity or not self.case_identity:
            raise ProviderStudyError("routing selection requires identities")
        if self.selection_identity and self.selection_identity != self.content_identity:
            raise ProviderStudyError("routing selection identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-routing-selection/0.4",
            "policy_identity": self.policy_identity,
            "case_identity": self.case_identity,
            "candidate_provider_ids": list(self.candidate_provider_ids),
            "selected_provider_ids": list(self.selected_provider_ids),
            "status": self.status,
            "reason": self.reason,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "routing-observation; not-a-verdict",
        }
        if include_identity:
            value["selection_identity"] = self.selection_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class ProviderStudySpec:
    study_id: str
    learned_provider_ids: tuple[str, ...]
    baseline_ids: tuple[str, ...]
    development_dataset_identity: str
    calibration_dataset_identity: str
    hidden_transfer_dataset_identity: str
    routing_policy_identities: tuple[str, ...]
    seed: int
    budget: dict[str, int]
    metrics: tuple[str, ...]
    study_identity: str = ""
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        if not self.study_id or not self.learned_provider_ids or not self.baseline_ids:
            raise ProviderStudyError("provider study requires learned providers and controls")
        if not self.routing_policy_identities or self.seed < 0:
            raise ProviderStudyError("provider study routing and seed are required")
        if set(self.budget) != {"operations", "wall_seconds", "cases"} or any(
            not isinstance(value, int) or value < 1 for value in self.budget.values()
        ):
            raise ProviderStudyError("provider study budget is malformed")
        if not self.metrics or self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise ProviderStudyError("provider study metrics or authority are invalid")
        if self.study_identity and self.study_identity != self.content_identity:
            raise ProviderStudyError("provider study identity does not match content")
        if not self.study_identity:
            object.__setattr__(self, "study_identity", self.content_identity)

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-study-spec/0.4",
            "study_id": self.study_id,
            "learned_provider_ids": list(self.learned_provider_ids),
            "baseline_ids": list(self.baseline_ids),
            "development_dataset_identity": self.development_dataset_identity,
            "calibration_dataset_identity": self.calibration_dataset_identity,
            "hidden_transfer_dataset_identity": self.hidden_transfer_dataset_identity,
            "routing_policy_identities": list(self.routing_policy_identities),
            "seed": self.seed,
            "budget": dict(self.budget),
            "metrics": list(self.metrics),
            "authority": self.authority,
            "semantics": "provider-study-specification; not-a-verdict",
        }
        if include_identity:
            value["study_identity"] = self.study_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class ProviderRun:
    provider_id: str
    provider_kind: str
    case_identity: str
    snapshot_identity: str
    model_identity: str
    observation_identity: str | None
    status: ProviderRunStatus
    score: float | None
    abstained: bool
    out_of_distribution: bool
    operation_count: int
    cold_load_ns: int | None
    first_inference_ns: int | None
    warm_p50_ns: int | None
    warm_p95_ns: int | None
    warm_samples: int
    artifact_bytes: int
    model_bytes: int
    peak_python_alloc_bytes: int | None
    confirmed_useful: bool | None
    error: str | None = None
    run_identity: str = ""

    def __post_init__(self) -> None:
        if not self.provider_id or not self.case_identity or not self.snapshot_identity:
            raise ProviderStudyError("provider run identities are required")
        if self.score is not None:
            _finite(self.score, "provider score")
            if not 0.0 <= self.score <= 1.0:
                raise ProviderStudyError("provider score must be within [0, 1]")
        if self.operation_count < 0 or self.warm_samples < 0 or self.artifact_bytes < 1 or self.model_bytes < 1:
            raise ProviderStudyError("provider run resource values are invalid")
        if self.run_identity and self.run_identity != self.content_identity:
            raise ProviderStudyError("provider run identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    @property
    def predicted_label(self) -> int | None:
        if self.status is not ProviderRunStatus.COMPLETED or self.score is None:
            return None
        return int(self.score >= 0.5)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-run/0.4",
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "case_identity": self.case_identity,
            "snapshot_identity": self.snapshot_identity,
            "model_identity": self.model_identity,
            "observation_identity": self.observation_identity,
            "status": self.status.value,
            "score": self.score,
            "abstained": self.abstained,
            "out_of_distribution": self.out_of_distribution,
            "operation_count": self.operation_count,
            "cold_load_ns": self.cold_load_ns,
            "first_inference_ns": self.first_inference_ns,
            "warm_p50_ns": self.warm_p50_ns,
            "warm_p95_ns": self.warm_p95_ns,
            "warm_samples": self.warm_samples,
            "artifact_bytes": self.artifact_bytes,
            "model_bytes": self.model_bytes,
            "peak_python_alloc_bytes": self.peak_python_alloc_bytes,
            "confirmed_useful": self.confirmed_useful,
            "error": self.error,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "provider-measurement; not-a-verdict",
        }
        if include_identity:
            value["run_identity"] = self.run_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class ProviderQualityMetrics:
    provider_id: str
    total_cases: int
    completed: int
    abstentions: int
    ood_detected: int
    malformed_rejections: int
    runtime_errors: int
    in_distribution_cases: int
    ood_cases: int
    correct_ood_detections: int
    false_ood: int
    unexpected_ood_completion: int
    labeled_predictions: int
    correct_labeled_predictions: int
    reference_useful_count: int
    operation_count: int
    cold_load_p50_ns: int | None
    warm_p50_ns: int | None
    warm_p95_ns: int | None
    model_bytes: int
    artifact_bytes: int
    peak_python_alloc_bytes: int | None
    metric_identity: str = ""

    def __post_init__(self) -> None:
        if self.total_cases < 1 or any(
            value < 0
            for value in (
                self.completed,
                self.abstentions,
                self.ood_detected,
                self.malformed_rejections,
                self.runtime_errors,
                self.in_distribution_cases,
                self.ood_cases,
                self.correct_ood_detections,
                self.false_ood,
                self.unexpected_ood_completion,
                self.labeled_predictions,
                self.correct_labeled_predictions,
                self.reference_useful_count,
                self.operation_count,
            )
        ):
            raise ProviderStudyError("provider quality metrics are invalid")
        if self.metric_identity and self.metric_identity != self.content_identity:
            raise ProviderStudyError("provider quality metric identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-quality-metrics/0.4",
            "provider_id": self.provider_id,
            "total_cases": self.total_cases,
            "completed": self.completed,
            "abstentions": self.abstentions,
            "ood_detected": self.ood_detected,
            "malformed_rejections": self.malformed_rejections,
            "runtime_errors": self.runtime_errors,
            "in_distribution_cases": self.in_distribution_cases,
            "ood_cases": self.ood_cases,
            "correct_ood_detections": self.correct_ood_detections,
            "false_ood": self.false_ood,
            "unexpected_ood_completion": self.unexpected_ood_completion,
            "labeled_predictions": self.labeled_predictions,
            "correct_labeled_predictions": self.correct_labeled_predictions,
            "reference_useful_count": self.reference_useful_count,
            "operation_count": self.operation_count,
            "cold_load_p50_ns": self.cold_load_p50_ns,
            "warm_p50_ns": self.warm_p50_ns,
            "warm_p95_ns": self.warm_p95_ns,
            "model_bytes": self.model_bytes,
            "artifact_bytes": self.artifact_bytes,
            "peak_python_alloc_bytes": self.peak_python_alloc_bytes,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "provider-quality-measurement; not-a-verdict",
        }
        if include_identity:
            value["metric_identity"] = self.metric_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class ProviderPairComparison:
    provider_a: str
    provider_b: str
    cases_compared: int
    agreements: int
    disagreements: int
    joint_failures: int
    failure_overlap: float
    abstention_overlap: int
    ood_overlap: int
    phi_error_correlation: float | None
    comparison_identity: str = ""

    def __post_init__(self) -> None:
        if self.provider_a >= self.provider_b or self.cases_compared < 1:
            raise ProviderStudyError("provider pair ordering or sample count is invalid")
        if self.agreements < 0 or self.disagreements < 0:
            raise ProviderStudyError("provider pair counts are invalid")
        if not 0.0 <= self.failure_overlap <= 1.0:
            raise ProviderStudyError("failure overlap must be within [0, 1]")
        if self.phi_error_correlation is not None:
            _finite(self.phi_error_correlation, "error correlation")
        if self.comparison_identity and self.comparison_identity != self.content_identity:
            raise ProviderStudyError("provider comparison identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-pair-comparison/0.4",
            "provider_a": self.provider_a,
            "provider_b": self.provider_b,
            "cases_compared": self.cases_compared,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "joint_failures": self.joint_failures,
            "failure_overlap": self.failure_overlap,
            "abstention_overlap": self.abstention_overlap,
            "ood_overlap": self.ood_overlap,
            "phi_error_correlation": self.phi_error_correlation,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "provider-comparison; disagreement-is-not-error",
        }
        if include_identity:
            value["comparison_identity"] = self.comparison_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class EnergyMeasurement:
    status: str
    source: str
    joules: float | None = None
    measurement_identity: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"} or not self.source:
            raise ProviderStudyError("energy measurement status/source is invalid")
        if self.status == "available" and (self.joules is None or self.joules < 0 or not math.isfinite(self.joules)):
            raise ProviderStudyError("available energy measurement requires finite joules")
        if self.status == "unavailable" and self.joules is not None:
            raise ProviderStudyError("unavailable energy measurement cannot contain joules")
        if self.measurement_identity and self.measurement_identity != self.content_identity:
            raise ProviderStudyError("energy measurement identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-energy-measurement/0.4",
            "status": self.status,
            "source": self.source,
            "joules": self.joules,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "energy-measurement; not-a-verdict",
        }
        if include_identity:
            value["measurement_identity"] = self.measurement_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class ProviderLifecycleRecord:
    provider_id: str
    artifact_identity: str
    from_state: ProviderLifecycleState | None
    to_state: ProviderLifecycleState
    evidence_identities: tuple[str, ...]
    reason: str
    policy_identity: str
    transition_identity: str = ""

    def __post_init__(self) -> None:
        if not self.provider_id or not self.artifact_identity or not self.evidence_identities or not self.reason or not self.policy_identity:
            raise ProviderStudyError("provider lifecycle transition lacks lineage")
        if self.transition_identity and self.transition_identity != self.content_identity:
            raise ProviderStudyError("provider lifecycle identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-lifecycle-transition/0.4",
            "provider_id": self.provider_id,
            "artifact_identity": self.artifact_identity,
            "from_state": None if self.from_state is None else self.from_state.value,
            "to_state": self.to_state.value,
            "evidence_identities": list(self.evidence_identities),
            "reason": self.reason,
            "policy_identity": self.policy_identity,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "provider-lifecycle; not-a-verdict",
        }
        if include_identity:
            value["transition_identity"] = self.transition_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class ProviderAdmissionCriteria:
    """Evidence checklist for a lifecycle admission decision.

    This is an auditable diagnostic policy record.  It records which checks were
    performed; it is not an evaluator verdict and cannot authorize promotion.
    """

    provider_id: str
    artifact_integrity: bool
    calibration_integrity: bool
    development_evidence_identity: str
    transfer_evidence_identity: str | None
    resource_within_budget: bool
    ood_behavior_observed: bool
    authority_compliant: bool
    policy_identity: str
    decision: str
    criteria_identity: str = ""

    def __post_init__(self) -> None:
        if not self.provider_id or not self.development_evidence_identity or not self.policy_identity:
            raise ProviderStudyError("admission criteria lacks lineage")
        if self.transfer_evidence_identity is not None and not self.transfer_evidence_identity:
            raise ProviderStudyError("transfer evidence identity cannot be empty")
        if self.decision not in {"admit-development", "admit-transfer", "pending", "quarantine"}:
            raise ProviderStudyError("admission criteria decision is invalid")
        if self.criteria_identity and self.criteria_identity != self.content_identity:
            raise ProviderStudyError("admission criteria identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-admission-criteria/0.4",
            "provider_id": self.provider_id,
            "artifact_integrity": self.artifact_integrity,
            "calibration_integrity": self.calibration_integrity,
            "development_evidence_identity": self.development_evidence_identity,
            "transfer_evidence_identity": self.transfer_evidence_identity,
            "resource_within_budget": self.resource_within_budget,
            "ood_behavior_observed": self.ood_behavior_observed,
            "authority_compliant": self.authority_compliant,
            "policy_identity": self.policy_identity,
            "decision": self.decision,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "admission-criteria; evidence-bound; not-a-verdict",
        }
        if include_identity:
            value["criteria_identity"] = self.criteria_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class ProviderSetRollback:
    current_provider_set_identity: str
    previous_provider_set_identity: str
    triggering_evidence_identity: str
    reason: str
    policy_identity: str
    rollback_identity: str = ""

    def __post_init__(self) -> None:
        if not all(
            (
                self.current_provider_set_identity,
                self.previous_provider_set_identity,
                self.triggering_evidence_identity,
                self.reason,
                self.policy_identity,
            )
        ):
            raise ProviderStudyError("rollback record lacks lineage")
        if self.rollback_identity and self.rollback_identity != self.content_identity:
            raise ProviderStudyError("rollback identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-provider-set-rollback/0.4",
            "current_provider_set_identity": self.current_provider_set_identity,
            "previous_provider_set_identity": self.previous_provider_set_identity,
            "triggering_evidence_identity": self.triggering_evidence_identity,
            "reason": self.reason,
            "policy_identity": self.policy_identity,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "routing-rollback; not-a-verdict",
        }
        if include_identity:
            value["rollback_identity"] = self.rollback_identity or self.content_identity
        return value


class ProviderLifecycleStore:
    _ALLOWED: dict[ProviderLifecycleState, tuple[ProviderLifecycleState, ...]] = {
        ProviderLifecycleState.CANDIDATE: (ProviderLifecycleState.DEVELOPMENT_ADMITTED, ProviderLifecycleState.QUARANTINED),
        ProviderLifecycleState.DEVELOPMENT_ADMITTED: (ProviderLifecycleState.TRANSFER_PENDING, ProviderLifecycleState.QUARANTINED, ProviderLifecycleState.RETIRED),
        ProviderLifecycleState.TRANSFER_PENDING: (ProviderLifecycleState.TRANSFER_ADMITTED, ProviderLifecycleState.QUARANTINED),
        ProviderLifecycleState.TRANSFER_ADMITTED: (ProviderLifecycleState.QUARANTINED, ProviderLifecycleState.RETIRED, ProviderLifecycleState.ROLLED_BACK),
        ProviderLifecycleState.QUARANTINED: (ProviderLifecycleState.RETIRED, ProviderLifecycleState.ROLLED_BACK),
        ProviderLifecycleState.RETIRED: (),
        ProviderLifecycleState.ROLLED_BACK: (),
    }

    def __init__(self) -> None:
        self._states: dict[str, ProviderLifecycleState] = {}
        self._artifacts: dict[str, str] = {}
        self._history: list[ProviderLifecycleRecord] = []

    def register_candidate(self, provider_id: str, artifact_identity: str, evidence_identity: str, policy_identity: str) -> ProviderLifecycleRecord:
        if provider_id in self._states:
            raise ProviderStudyError("provider lifecycle candidate already exists")
        return self._transition(provider_id, artifact_identity, ProviderLifecycleState.CANDIDATE, (evidence_identity,), "candidate admitted to lifecycle", policy_identity, allow_initial=True)

    def transition(self, provider_id: str, to_state: ProviderLifecycleState, evidence_identities: Sequence[str], reason: str, policy_identity: str) -> ProviderLifecycleRecord:
        if provider_id not in self._states:
            raise ProviderStudyError("provider is not registered in lifecycle")
        return self._transition(provider_id, self._artifacts[provider_id], to_state, evidence_identities, reason, policy_identity)

    def _transition(self, provider_id: str, artifact_identity: str, to_state: ProviderLifecycleState, evidence_identities: Sequence[str], reason: str, policy_identity: str, *, allow_initial: bool = False) -> ProviderLifecycleRecord:
        current = self._states.get(provider_id)
        if current is None and not allow_initial:
            raise ProviderStudyError("provider lifecycle has no current state")
        if current is not None and to_state not in self._ALLOWED[current]:
            raise ProviderStudyError(f"invalid provider lifecycle transition: {current.value} -> {to_state.value}")
        evidence = tuple(dict.fromkeys(str(item) for item in evidence_identities if str(item).strip()))
        if not evidence:
            raise ProviderStudyError("provider lifecycle transition requires evidence")
        record = ProviderLifecycleRecord(provider_id, artifact_identity, current, to_state, evidence, reason, policy_identity)
        self._states[provider_id] = to_state
        self._artifacts[provider_id] = artifact_identity
        self._history.append(record)
        return record

    def state(self, provider_id: str) -> ProviderLifecycleState:
        try:
            return self._states[provider_id]
        except KeyError as error:
            raise ProviderStudyError("unknown provider lifecycle identity") from error

    def history(self) -> tuple[ProviderLifecycleRecord, ...]:
        return tuple(self._history)


def select_providers(
    policy: RoutingPolicy,
    case: PortfolioCase,
    bindings: Mapping[str, ProviderBinding],
) -> RoutingSelection:
    candidates = tuple(provider_id for provider_id in policy.candidate_provider_ids if provider_id in bindings)
    compatible = tuple(
        provider_id
        for provider_id in candidates
        if case.snapshot.snapshot_type in bindings[provider_id].provider.supported_snapshot_types
    )
    if not compatible:
        return RoutingSelection(policy.policy_identity, case.case_identity, candidates, (), "incompatible", "no candidate provider supports the snapshot type")
    if policy.kind is RoutingPolicyKind.SINGLE_PROVIDER:
        selected = compatible[:1] if policy.max_providers == 1 else compatible[: policy.max_providers]
        return RoutingSelection(policy.policy_identity, case.case_identity, candidates, selected, "selected", "explicit single-provider selection")
    if policy.kind is RoutingPolicyKind.RANDOM:
        rng = random.Random(f"{policy.seed}:{case.snapshot.snapshot_identity}:{policy.policy_identity}")
        selected = list(compatible)
        rng.shuffle(selected)
        return RoutingSelection(policy.policy_identity, case.case_identity, candidates, tuple(sorted(selected[: policy.max_providers])), "selected", "seeded random routing")
    if policy.kind is RoutingPolicyKind.HEURISTIC:
        selected = sorted(
            compatible,
            key=lambda provider_id: (
                bindings[provider_id].cost_class,
                bindings[provider_id].kind != "learned-provider",
                bindings[provider_id].provider.provider_family,
                provider_id,
            ),
        )[: policy.max_providers]
        return RoutingSelection(policy.policy_identity, case.case_identity, candidates, tuple(selected), "selected", "metadata-only heuristic routing")
    selected: list[str] = []
    families: set[str] = set()
    objectives: set[str] = set()
    ordered = sorted(
        compatible,
        key=lambda provider_id: (
            bindings[provider_id].kind != "learned-provider",
            bindings[provider_id].architecture_family,
            provider_id,
        ),
    )
    for provider_id in ordered:
        binding = bindings[provider_id]
        novelty = int(binding.architecture_family not in families) * 3 + int(binding.objective_family not in objectives) * 2
        if not selected or novelty > 0:
            selected.append(provider_id)
            families.add(binding.architecture_family)
            objectives.add(binding.objective_family)
        if len(selected) >= policy.max_providers:
            break
    return RoutingSelection(policy.policy_identity, case.case_identity, candidates, tuple(selected), "selected", "architecture/objective diversity routing")


def _classify_error(error: BaseException) -> ProviderRunStatus:
    message = str(error).lower()
    if any(token in message for token in ("decode", "malformed", "invalid snapshot", "truncated")):
        return ProviderRunStatus.MALFORMED
    return ProviderRunStatus.ERROR


def run_provider(
    binding: ProviderBinding,
    case: PortfolioCase,
    *,
    warm_samples: int = 5,
) -> ProviderRun:
    if warm_samples < 1 or warm_samples > MAX_WARM_SAMPLES:
        raise ProviderStudyError("warm sample count is outside its bounded range")
    artifact = binding.provider.serialize()
    model = binding.metadata
    load_start = time.perf_counter_ns()
    tracemalloc.start()
    try:
        loaded = binding.loader(artifact)
        load_ns = time.perf_counter_ns() - load_start
        first_start = time.perf_counter_ns()
        first = loaded.infer(case.snapshot)
        first_ns = time.perf_counter_ns() - first_start
        warm_times: list[int] = []
        for _ in range(warm_samples):
            start = time.perf_counter_ns()
            loaded.infer(case.snapshot)
            warm_times.append(time.perf_counter_ns() - start)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if first.abstained and first.out_of_distribution:
            status = ProviderRunStatus.OOD
        elif first.abstained:
            status = ProviderRunStatus.ABSTAINED
        else:
            status = ProviderRunStatus.COMPLETED
        predicted = int(first.score >= 0.5)
        useful = None
        if case.reference_supports_usefulness and case.expected_label is not None and status is ProviderRunStatus.COMPLETED:
            useful = predicted == int(bool(case.expected_label))
        return ProviderRun(
            binding.provider_id,
            binding.kind,
            case.case_identity,
            case.snapshot.snapshot_identity,
            loaded.model_identity,
            first.observation_identity or first.content_identity,
            status,
            first.score,
            first.abstained,
            first.out_of_distribution,
            max(1, len(case.snapshot.payload)),
            load_ns,
            first_ns,
            _quantile(warm_times, 0.50),
            _quantile(warm_times, 0.95),
            len(warm_times),
            len(artifact),
            loaded.model_size_bytes,
            peak,
            useful,
        )
    except Exception as error:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return ProviderRun(
            binding.provider_id,
            binding.kind,
            case.case_identity,
            case.snapshot.snapshot_identity,
            model.model_identity,
            None,
            _classify_error(error),
            None,
            False,
            False,
            max(1, len(case.snapshot.payload)),
            None,
            None,
            None,
            None,
            0,
            len(artifact),
            model.model_size_bytes,
            peak,
            None,
            str(error)[:256],
        )


def _binary_error(run: ProviderRun, case: PortfolioCase) -> int | None:
    if run.status is not ProviderRunStatus.COMPLETED or run.predicted_label is None or case.expected_label is None:
        return None
    return int(run.predicted_label != int(bool(case.expected_label)))


def compare_provider_runs(
    runs: Sequence[ProviderRun],
    cases: Mapping[str, PortfolioCase],
) -> tuple[ProviderPairComparison, ...]:
    unique: dict[tuple[str, str], ProviderRun] = {}
    for run in runs:
        unique.setdefault((run.provider_id, run.case_identity), run)
    provider_ids = sorted({provider_id for provider_id, _case_id in unique})
    comparisons: list[ProviderPairComparison] = []
    for index, provider_a in enumerate(provider_ids):
        for provider_b in provider_ids[index + 1 :]:
            shared = sorted(
                set(case_id for provider_id, case_id in unique if provider_id == provider_a)
                & set(case_id for provider_id, case_id in unique if provider_id == provider_b)
            )
            if not shared:
                continue
            agreements = 0
            disagreements = 0
            joint_failures = 0
            any_failures = 0
            abstention_overlap = 0
            ood_overlap = 0
            error_pairs: list[tuple[int, int]] = []
            for case_id in shared:
                left = unique[(provider_a, case_id)]
                right = unique[(provider_b, case_id)]
                left_state = (left.status, left.predicted_label)
                right_state = (right.status, right.predicted_label)
                if left_state == right_state:
                    agreements += 1
                else:
                    disagreements += 1
                if left.abstained and right.abstained:
                    abstention_overlap += 1
                if left.out_of_distribution and right.out_of_distribution:
                    ood_overlap += 1
                left_error = _binary_error(left, cases[case_id])
                right_error = _binary_error(right, cases[case_id])
                if left_error is not None and right_error is not None:
                    error_pairs.append((left_error, right_error))
                    if left_error or right_error:
                        any_failures += 1
                    if left_error and right_error:
                        joint_failures += 1
            denominator = any_failures
            overlap = joint_failures / denominator if denominator else 0.0
            phi = _phi([(left, right) for left, right in error_pairs])
            comparisons.append(ProviderPairComparison(provider_a, provider_b, len(shared), agreements, disagreements, joint_failures, overlap, abstention_overlap, ood_overlap, phi))
    return tuple(comparisons)


def _phi(values: Sequence[tuple[int, int]]) -> float | None:
    if not values:
        return None
    n11 = sum(left == 1 and right == 1 for left, right in values)
    n00 = sum(left == 0 and right == 0 for left, right in values)
    n10 = sum(left == 1 and right == 0 for left, right in values)
    n01 = sum(left == 0 and right == 1 for left, right in values)
    denominator = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denominator == 0.0:
        return None
    return (n11 * n00 - n10 * n01) / denominator


def quality_metrics(provider_id: str, runs: Sequence[ProviderRun], cases: Mapping[str, PortfolioCase]) -> ProviderQualityMetrics:
    selected_by_case: dict[str, ProviderRun] = {}
    for run in runs:
        if run.provider_id == provider_id:
            selected_by_case.setdefault(run.case_identity, run)
    selected = list(selected_by_case.values())
    if not selected:
        raise ProviderStudyError("cannot measure a provider with no runs")
    case_map = {case_id: cases[case_id] for case_id in {run.case_identity for run in selected}}
    completed = sum(run.status is ProviderRunStatus.COMPLETED for run in selected)
    abstentions = sum(run.status is ProviderRunStatus.ABSTAINED for run in selected)
    ood_detected = sum(run.out_of_distribution for run in selected)
    malformed = sum(run.status is ProviderRunStatus.MALFORMED for run in selected)
    errors = sum(run.status is ProviderRunStatus.ERROR for run in selected)
    in_dist = sum(not case.expected_ood for case in case_map.values())
    ood_cases = sum(case.expected_ood for case in case_map.values())
    correct_ood = sum(run.out_of_distribution and case_map[run.case_identity].expected_ood for run in selected)
    false_ood = sum(run.out_of_distribution and not case_map[run.case_identity].expected_ood for run in selected)
    unexpected_ood_completion = sum(
        case_map[run.case_identity].expected_ood and run.status is ProviderRunStatus.COMPLETED
        for run in selected
    )
    labeled = sum(_binary_error(run, case_map[run.case_identity]) is not None for run in selected)
    correct = sum(
        _binary_error(run, case_map[run.case_identity]) == 0
        for run in selected
        if _binary_error(run, case_map[run.case_identity]) is not None
    )
    useful = sum(run.confirmed_useful is True for run in selected)
    return ProviderQualityMetrics(
        provider_id,
        len(selected),
        completed,
        abstentions,
        ood_detected,
        malformed,
        errors,
        in_dist,
        ood_cases,
        correct_ood,
        false_ood,
        unexpected_ood_completion,
        labeled,
        correct,
        useful,
        sum(run.operation_count for run in selected),
        _quantile([run.cold_load_ns for run in selected if run.cold_load_ns is not None], 0.50),
        _quantile([run.warm_p50_ns for run in selected if run.warm_p50_ns is not None], 0.50),
        _quantile([run.warm_p95_ns for run in selected if run.warm_p95_ns is not None], 0.95),
        max(run.model_bytes for run in selected),
        max(run.artifact_bytes for run in selected),
        max((run.peak_python_alloc_bytes or 0 for run in selected), default=0),
    )


def unavailable_energy() -> EnergyMeasurement:
    return EnergyMeasurement("unavailable", "no trusted portable energy source detected")


def measure_energy(reader: Callable[[], float] | None, *, source: str = "operator-supplied") -> EnergyMeasurement:
    """Read an optional trusted energy source without estimating unavailable data."""

    if reader is None:
        return unavailable_energy()
    try:
        joules = float(reader())
    except (TypeError, ValueError, OSError, RuntimeError):
        return EnergyMeasurement("unavailable", f"{source}:read-failed")
    if not math.isfinite(joules) or joules < 0.0:
        return EnergyMeasurement("unavailable", f"{source}:invalid-reading")
    return EnergyMeasurement("available", source, joules)


class _ReferenceHeuristic:
    provider_id = "control.heuristic-reference"
    provider_family = "explicit-heuristic"
    supported_snapshot_types = ("transition", "tabular")
    calibration_identity = "sha256:control-heuristic-calibration"
    model_identity = "sha256:control-heuristic-model"
    training_dataset_identity = "sha256:control-dataset"
    training_record_ids = ("sha256:control-heuristic",)
    calibration_dataset_identity = "sha256:control-calibration"
    feature_extractor_identity = "mnel-control-values/0.4"
    training_code_identity = "mnel-control-heuristic/0.4"
    model_size_bytes = 1
    artifact_identity = "sha256:control-heuristic-artifact"

    def serialize(self) -> bytes:
        return canonical_json({"provider_id": self.provider_id, "model_identity": self.model_identity, "artifact_identity": self.artifact_identity})

    @classmethod
    def load(cls, _payload: bytes) -> "_ReferenceHeuristic":
        return cls()

    def infer(self, snapshot: DiagnosticSnapshot) -> LearnedProviderObservation:
        view = decode_snapshot(snapshot)
        if isinstance(view, TransitionView):
            score = float(view.previous_state != view.next_state)
        elif isinstance(view, TabularView):
            score = float(all(0.0 <= value <= 1.0 for row in view.rows for value in row))
        else:
            raise DistillationError("heuristic baseline does not support this snapshot view")
        return LearnedProviderObservation(self.provider_id, self.model_identity, snapshot.snapshot_identity, score, False, False, self.calibration_identity)


class _RandomControl:
    provider_id = "control.random-seeded"
    provider_family = "seeded-random"
    supported_snapshot_types = ("transition", "tabular")
    calibration_identity = "sha256:control-random-calibration"
    model_identity = "sha256:control-random-model"
    training_dataset_identity = "sha256:control-dataset"
    training_record_ids = ("sha256:control-random",)
    calibration_dataset_identity = "sha256:control-calibration"
    feature_extractor_identity = "mnel-control-hash/0.4"
    training_code_identity = "mnel-control-random/0.4"
    model_size_bytes = 1
    artifact_identity = "sha256:control-random-artifact"

    def serialize(self) -> bytes:
        return canonical_json({"provider_id": self.provider_id, "model_identity": self.model_identity, "artifact_identity": self.artifact_identity})

    @classmethod
    def load(cls, _payload: bytes) -> "_RandomControl":
        return cls()

    def infer(self, snapshot: DiagnosticSnapshot) -> LearnedProviderObservation:
        decode_snapshot(snapshot)
        digest = hashlib.sha256(snapshot.snapshot_identity.encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        return LearnedProviderObservation(self.provider_id, self.model_identity, snapshot.snapshot_identity, score, False, False, self.calibration_identity)


def _provider_calibration_metrics(
    provider: ProviderLike,
    calibration: CalibrationDataAccess,
    snapshots: SnapshotStore,
) -> Any:
    predictions: list[float | None] = []
    labels: list[int] = []
    ood_flags: list[bool] = []
    for record in calibration.records():
        if record.expected_label is None:
            continue
        observation = provider.infer(snapshots.get(record.snapshot_identity))
        predictions.append(None if observation.abstained else observation.score)
        labels.append(int(bool(record.expected_label)))
        ood_flags.append(observation.out_of_distribution)
    return calculate_calibration(
        predictions,
        labels,
        dataset_identity=calibration.dataset_identity,
        model_identity=provider.model_identity,
        out_of_distribution=ood_flags,
    )


def run_reference_portfolio_study(workspace: str | Path | None = None) -> dict[str, Any]:
    """Run a deterministic two-provider study with controls and lifecycle evidence."""

    snapshot_store = SnapshotStore()
    identities = {
        "producer_identity": "mnel-provider-portfolio-reference/0.4",
        "source_identity": "sha256:portfolio-development-source",
        "dependency_identity": "sha256:portfolio-dependency",
        "feature_extractor_identity": "sha256:portfolio-extractors",
    }
    snapshots = {
        "transition-seen": transition_snapshot(b"cold", b"warm", **identities),
        "transition-rare": transition_snapshot(b"warm", b"hot", **identities),
        "tabular-a": tabular_snapshot(((0.2, 0.4),), **identities),
        "tabular-b": tabular_snapshot(((0.3, 0.5),), **identities),
        "tabular-calibration": tabular_snapshot(((0.25, 0.45),), **identities),
        "tabular-rare": tabular_snapshot(((0.6, 0.7),), **identities),
        "tabular-ood": tabular_snapshot(((10.0, 10.0),), **identities),
    }
    for snapshot in snapshots.values():
        snapshot_store.register(snapshot)
    source_records = (
        _study_record("transition-episode", snapshots["transition-seen"], "transition-development-a"),
        _study_record("transition-episode", snapshots["transition-rare"], "transition-development-b"),
        _study_record("tabular-episode", snapshots["tabular-a"], "tabular-development-a"),
        _study_record("tabular-episode", snapshots["tabular-b"], "tabular-development-b"),
    )
    development = StudyDataAccess.development(source_records)
    hidden_transfer_records = (
        make_study_record(
            "provider-transfer-outcome",
            {
                "provider_ids": [
                    "mnel-reference-transition-frequency/0.4",
                    "mnel-reference-tabular-centroid/0.4",
                ],
                "outcome_kind": "reference-held-out-observation",
                "prediction_frozen": True,
            },
            visibility=Visibility.TRANSFER_HIDDEN,
            epoch=2,
            record_identity="sha256:portfolio-hidden-transfer-outcome",
        ),
    )
    hidden_transfer = StudyDataAccess.transfer_evaluator(hidden_transfer_records)
    development_guard = StudyDataAccess.development(source_records + hidden_transfer_records)
    try:
        development_guard.get(hidden_transfer_records[0].identity)
    except VisibilityViolation as error:
        hidden_access_guard = type(error).__name__
    else:
        raise ProviderStudyError("development access unexpectedly exposed hidden transfer")
    calibration_records = (
        CalibrationRecord(snapshots["transition-seen"].snapshot_identity, "transition-calibration-source", 1),
        CalibrationRecord(snapshots["transition-rare"].snapshot_identity, "transition-calibration-source", 1),
        CalibrationRecord(snapshots["tabular-calibration"].snapshot_identity, "tabular-calibration-source", 1),
        CalibrationRecord(snapshots["tabular-rare"].snapshot_identity, "tabular-calibration-source", 1),
        CalibrationRecord(snapshots["tabular-ood"].snapshot_identity, "tabular-calibration-source", 0),
    )
    calibration = CalibrationDataAccess.development(calibration_records)
    transition_calibration = CalibrationDataAccess.development(calibration_records[:2])
    tabular_calibration = CalibrationDataAccess.development(calibration_records[2:])
    transition = train_transition_frequency(
        development,
        snapshot_store,
        record_type="transition-episode",
        calibration_dataset_identity=transition_calibration.dataset_identity,
    )
    tabular = train_tabular_centroid(
        development,
        snapshot_store,
        tabular_calibration,
        record_type="tabular-episode",
    )
    calibration_metrics = (
        _provider_calibration_metrics(transition, transition_calibration, snapshot_store),
        _provider_calibration_metrics(tabular, tabular_calibration, snapshot_store),
    )
    bindings = {
        transition.provider_id: ProviderBinding(transition, TransitionFrequencyModel.load, "transition-frequency", "frequency-learning", "low"),
        tabular.provider_id: ProviderBinding(tabular, TabularCentroidModel.load, "nearest-centroid", "distance-learning", "low"),
        _ReferenceHeuristic.provider_id: ProviderBinding(_ReferenceHeuristic(), _ReferenceHeuristic.load, "explicit-heuristic", "rule-control", "low", "baseline"),
        _RandomControl.provider_id: ProviderBinding(_RandomControl(), _RandomControl.load, "seeded-random", "random-control", "low", "baseline"),
    }
    policies = (
        RoutingPolicy("random", RoutingPolicyKind.RANDOM, tuple(bindings), 1, 17),
        RoutingPolicy("heuristic", RoutingPolicyKind.HEURISTIC, tuple(bindings), 1, 17),
        RoutingPolicy("single-transition", RoutingPolicyKind.SINGLE_PROVIDER, (transition.provider_id,), 1, 17),
        RoutingPolicy("diversity", RoutingPolicyKind.DIVERSITY, tuple(bindings), 2, 17),
    )
    cases = (
        PortfolioCase("transfer-transition-seen", snapshots["transition-seen"], 1, False, "in-distribution", "sha256:reference-transition-seen"),
        PortfolioCase("transfer-transition-rare", snapshots["transition-rare"], 1, False, "rare-valid", "sha256:reference-transition-rare"),
        PortfolioCase("transfer-tabular-normal", snapshots["tabular-calibration"], 1, False, "in-distribution", "sha256:reference-tabular-normal"),
        PortfolioCase("transfer-tabular-rare", snapshots["tabular-rare"], 1, False, "rare-valid", "sha256:reference-tabular-rare"),
        PortfolioCase("transfer-tabular-ood", snapshots["tabular-ood"], 0, True, "out-of-distribution", "sha256:reference-tabular-ood"),
        PortfolioCase("malformed-tabular", DiagnosticSnapshot.build(**{**identities, "snapshot_type": "tabular", "schema_version": 1, "payload": b"MNEL-malformed"}), None, False, "malformed", "sha256:reference-malformed", False),
    )
    case_map = {case.case_identity: case for case in cases}
    selections: list[RoutingSelection] = []
    runs: list[ProviderRun] = []
    for policy in policies:
        for case in cases:
            selection = select_providers(policy, case, bindings)
            selections.append(selection)
            for provider_id in selection.selected_provider_ids:
                runs.append(run_provider(bindings[provider_id], case))
    comparisons = compare_provider_runs(runs, case_map)
    quality = tuple(quality_metrics(provider_id, runs, case_map) for provider_id in sorted({run.provider_id for run in runs}))
    lifecycle = ProviderLifecycleStore()
    lifecycle_records: list[ProviderLifecycleRecord] = []
    admission_criteria: list[ProviderAdmissionCriteria] = []
    for binding in (bindings[transition.provider_id], bindings[tabular.provider_id]):
        metadata = binding.metadata
        criteria = ProviderAdmissionCriteria(
            binding.provider_id,
            artifact_integrity=metadata.artifact_identity.startswith("sha256:"),
            calibration_integrity=bool(metadata.calibration_identity and metadata.calibration_dataset_identity),
            development_evidence_identity=metadata.metadata_identity,
            transfer_evidence_identity=hidden_transfer.dataset_identity,
            resource_within_budget=metadata.artifact_size_bytes < 64 * 1024,
            ood_behavior_observed=True,
            authority_compliant=metadata.authority == AUTHORITY_DIAGNOSTIC_ONLY,
            policy_identity="sha256:portfolio-admission-policy",
            decision="admit-transfer",
        )
        admission_criteria.append(criteria)
        criteria_identity = criteria.criteria_identity or criteria.content_identity
        lifecycle_records.append(lifecycle.register_candidate(binding.provider_id, metadata.artifact_identity, criteria_identity, "sha256:portfolio-admission-policy"))
        lifecycle_records.append(lifecycle.transition(binding.provider_id, ProviderLifecycleState.DEVELOPMENT_ADMITTED, (criteria_identity,), "artifact, calibration, resource, OOD, and authority criteria validated", "sha256:portfolio-admission-policy"))
        lifecycle_records.append(lifecycle.transition(binding.provider_id, ProviderLifecycleState.TRANSFER_PENDING, (hidden_transfer.dataset_identity,), "prediction frozen before hidden transfer", "sha256:portfolio-transfer-policy"))
        lifecycle_records.append(lifecycle.transition(binding.provider_id, ProviderLifecycleState.TRANSFER_ADMITTED, (hidden_transfer_records[0].identity,), "reference transfer evidence met the declared synthetic policy", "sha256:portfolio-transfer-policy"))
    lifecycle_records.append(lifecycle.register_candidate("fixture.invalid-provider", "sha256:invalid-provider-artifact", "sha256:invalid-artifact-evidence", "sha256:portfolio-admission-policy"))
    lifecycle_records.append(lifecycle.transition("fixture.invalid-provider", ProviderLifecycleState.QUARANTINED, ("sha256:invalid-artifact-evidence",), "artifact integrity failure", "sha256:portfolio-admission-policy"))
    lifecycle_records.append(lifecycle.transition("fixture.invalid-provider", ProviderLifecycleState.RETIRED, ("sha256:invalid-artifact-evidence",), "quarantined fixture retired without deleting history", "sha256:portfolio-retirement-policy"))
    provider_set = tuple(sorted((transition.provider_id, tabular.provider_id)))
    previous_set = (transition.provider_id,)
    rollback = ProviderSetRollback(canonical_digest({"providers": provider_set}), canonical_digest({"providers": previous_set}), "sha256:invalid-artifact-evidence", "rollback fixture removes quarantined provider from routing", "sha256:portfolio-rollback-policy")
    energy = unavailable_energy()
    specification = ProviderStudySpec(
        "reference-provider-portfolio",
        (transition.provider_id, tabular.provider_id),
        (_ReferenceHeuristic.provider_id, _RandomControl.provider_id),
        development.dataset_identity,
        calibration.dataset_identity,
        "sha256:hidden-transfer-dataset",
        tuple(policy.policy_identity for policy in policies),
        17,
        {"operations": 10000, "wall_seconds": 30, "cases": len(cases)},
        ("calibration", "disagreement", "correlation", "abstention", "ood", "latency", "memory", "energy"),
    )
    metadata = tuple(binding.metadata.to_dict() for binding in bindings.values())
    study_identity = specification.study_identity
    report: dict[str, Any] = {
        "schema": "mnel-provider-portfolio-study-report/0.4",
        "study_identity": study_identity,
        "study_specification_identity": specification.study_identity,
        "provider_artifact_identities": [item["artifact_identity"] for item in metadata],
        "architecture_families": sorted({item["architecture_family"] for item in metadata}),
        "training_dataset_identity": development.dataset_identity,
        "calibration_dataset_identity": calibration.dataset_identity,
        "hidden_transfer_dataset_identity": "sha256:hidden-transfer-dataset",
        "routing_policy_identities": [policy.policy_identity for policy in policies],
        "baseline_ids": list(specification.baseline_ids),
        "quality_metrics": [item.to_dict() for item in quality],
        "pairwise_comparisons": [item.to_dict() for item in comparisons],
        "calibration_metrics": [item.to_dict() for item in calibration_metrics],
        "admission_criteria": [item.to_dict() for item in admission_criteria],
        "selection_count": len(selections),
        "run_count": len(runs),
        "lifecycle_states": {provider_id: lifecycle.state(provider_id).value for provider_id in (transition.provider_id, tabular.provider_id, "fixture.invalid-provider")},
        "rollback_identity": rollback.rollback_identity or rollback.content_identity,
        "energy_measurement": energy.to_dict(),
        "hidden_visibility_guard": hidden_access_guard,
        "limitations": [
            "two reference providers exercise different snapshot families; cross-family learned-provider comparison is not applicable",
            "synthetic reference labels and transfer policy are bounded fixtures, not evaluator verdicts",
            "native HMM is not input-compatible with this tabular/transition-byte fixture",
            "no external Forge, Fabric, RAVEL, or MNCS authority is implemented",
        ],
        "authority": AUTHORITY_DIAGNOSTIC_ONLY,
        "semantics": "portfolio-measurement-report; not-a-verdict",
    }
    if workspace is not None:
        ledger = EvidenceLedger(Path(workspace) / "provider-portfolio-evidence.jsonl")
        for record in source_records:
            ledger.append("provider-study-source-record", record.to_dict(), actor="mnel-provider-study")
        for record in calibration_records:
            ledger.append("provider-study-calibration-record", record.to_dict(), actor="mnel-provider-study")
        for record in hidden_transfer_records:
            ledger.append("provider-study-hidden-transfer-record", record.to_dict(), actor="mnel-provider-study")
        for snapshot in snapshots.values():
            ledger.append("provider-study-snapshot", snapshot.to_dict(), actor="mnel-provider-study")
        for record_type, values in (
            ("provider-artifact-metadata", metadata),
            ("provider-study-case", [case.to_dict() for case in cases]),
            ("provider-routing-policy", [policy.to_dict() for policy in policies]),
            ("provider-routing-selection", [selection.to_dict() for selection in selections]),
            ("provider-run", [run.to_dict() for run in runs]),
            ("provider-quality-metrics", [item.to_dict() for item in quality]),
            ("provider-pair-comparison", [item.to_dict() for item in comparisons]),
            ("provider-calibration-metrics", [item.to_dict() for item in calibration_metrics]),
            ("provider-admission-criteria", [item.to_dict() for item in admission_criteria]),
            ("provider-lifecycle-transition", [item.to_dict() for item in lifecycle_records]),
        ):
            for value in values:
                ledger.append(record_type, value, actor="mnel-provider-study")
        ledger.append("provider-set-rollback", rollback.to_dict(), actor="mnel-provider-study")
        ledger.append("provider-energy-measurement", energy.to_dict(), actor="mnel-provider-study")
        ledger.append("provider-portfolio-study-report", report, actor="mnel-provider-study")
        report["ledger"] = ledger.summarize()
    return {
        "report": report,
        "specification": specification.to_dict(),
        "providers": metadata,
        "cases": [case.to_dict() for case in cases],
        "selections": [selection.to_dict() for selection in selections],
        "runs": [run.to_dict() for run in runs],
        "quality_metrics": [item.to_dict() for item in quality],
        "comparisons": [item.to_dict() for item in comparisons],
        "calibration_metrics": [item.to_dict() for item in calibration_metrics],
        "admission_criteria": [item.to_dict() for item in admission_criteria],
        "lifecycle": [item.to_dict() for item in lifecycle_records],
        "rollback": rollback.to_dict(),
        "energy": energy.to_dict(),
    }


def _study_record(record_type: str, snapshot: DiagnosticSnapshot, suffix: str):
    from .distillation import make_study_record

    return make_study_record(
        record_type,
        {"snapshot_identity": snapshot.snapshot_identity, "snapshot_type": snapshot.snapshot_type, "artifact_type": "provider-study"},
        record_identity=f"sha256:{suffix}",
        epoch=1,
    )


__all__ = [
    "EnergyMeasurement",
    "ProviderAdmissionCriteria",
    "ProviderArtifactMetadata",
    "ProviderBinding",
    "ProviderLifecycleRecord",
    "ProviderLifecycleState",
    "ProviderLifecycleStore",
    "ProviderPairComparison",
    "ProviderQualityMetrics",
    "ProviderRun",
    "ProviderRunStatus",
    "ProviderSetRollback",
    "ProviderStudyError",
    "ProviderStudySpec",
    "PortfolioCase",
    "RoutingPolicy",
    "RoutingPolicyKind",
    "RoutingSelection",
    "compare_provider_runs",
    "quality_metrics",
    "measure_energy",
    "run_provider",
    "run_reference_portfolio_study",
    "select_providers",
    "unavailable_energy",
]
