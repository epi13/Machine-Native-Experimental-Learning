"""A small, evidence-preserving one-step distilled specialist reference.

The module intentionally keeps teacher observation, target construction, training,
student inference, and teacher fallback as separate operations.  The student is a
tiny dependency-free affine/tanh classifier.  Its learned boundary is exactly one
forward evaluation; preprocessing, calibration, OOD checks, and fallback are
measured separately and never become authority.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from .core import canonical_digest, canonical_json
from .recurrent_specialist import (
    AUTHORITY,
    DIMENSIONS,
    MAX_CONTEXT_OBSERVATIONS,
    RecurrentSpecialistModel,
    SpecialistContextState,
    SpecialistDecision,
    SpecialistError,
)

ARTIFACT_SCHEMA = "mnel-one-step-specialist-artifact/0.1"
DECISION_SCHEMA = "mnel-one-step-specialist-decision/0.1"
DISTILLATION_RECORD_SCHEMA = "mnel-distillation-record/0.1"
DISTILLATION_DATASET_SCHEMA = "mnel-distillation-dataset/0.1"
PROVIDER_ID = "mnel-one-step-distilled-specialist/0.1"
PROVIDER_ABI = "mnel-specialist-provider-abi/0.1"
HIDDEN_WIDTH = 4
MAX_ROWS = 128
MAX_EPOCHS = 256


class OneStepSpecialistError(ValueError):
    """A malformed lineage record, artifact, request, or bounded invocation."""


class TargetStatus(StrEnum):
    VERIFIED = "verified"
    REJECTED_TEACHER = "rejected-teacher-target"
    UNKNOWN = "unknown"
    ABSTENTION = "abstention"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise OneStepSpecialistError(f"{label} must be a sha256 identity")
    return value


def _text(value: object, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise OneStepSpecialistError(f"{label} must be a bounded non-empty string")
    return value


def _features(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != DIMENSIONS:
        raise OneStepSpecialistError(f"{label} must contain exactly {DIMENSIONS} lanes")
    result = tuple(value)
    if any(
        not isinstance(item, int) or isinstance(item, bool) or not -1000 <= item <= 1000
        for item in result
    ):
        raise OneStepSpecialistError(f"{label} contains an invalid lane")
    return result


def _number(value: object, label: str, *, limit: float = 64.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OneStepSpecialistError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or abs(result) > limit:
        raise OneStepSpecialistError(f"{label} is outside its numeric bound")
    return result


def _vector(value: object, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise OneStepSpecialistError(f"{label} has the wrong width")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _matrix(value: object, label: str, rows: int, columns: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise OneStepSpecialistError(f"{label} has the wrong row count")
    return tuple(_vector(row, f"{label}[{index}]", columns) for index, row in enumerate(value))


def _reject_authority(value: object) -> None:
    forbidden = {
        "verdict",
        "evaluator_verdict",
        "evaluator_eligible",
        "promotion",
        "promotion_authorized",
        "conformance",
        "permission",
        "credentials",
        "trust",
        "mncs_verdict",
        "mncds_verdict",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise OneStepSpecialistError(f"specialist payload contains authority field: {key}")
            if str(key).lower() == "authority" and child != AUTHORITY:
                raise OneStepSpecialistError("specialist payload attempted to expand authority")
            _reject_authority(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_authority(child)


def _without_timing(value: Any) -> Any:
    """Remove host-clock measurements before deriving a reproducible study identity."""

    if isinstance(value, Mapping):
        return {
            key: _without_timing(child)
            for key, child in value.items()
            if not key.endswith("_ns") and key not in {"latency_ns", "warm_latency_ns"}
        }
    if isinstance(value, list):
        return [_without_timing(child) for child in value]
    return value


def _source_identity(row: Mapping[str, Any]) -> str:
    candidate = row.get("source_observation_identity")
    if isinstance(candidate, str) and candidate.startswith("sha256:") and len(candidate) == 71:
        return candidate
    return canonical_digest(
        {
            "record_id": _text(row.get("record_id"), "record_id"),
            "features": list(_features(row.get("features"), "features")),
        }
    )


def _context_for_student(model: OneStepStudentModel) -> SpecialistContextState:
    context = SpecialistContextState(
        provider_identity=model.teacher_provider_identity,
        generation_identity=model.teacher_generation_identity,
        role_identity=model.target_role,
        source_observation_identities=(),
        feature_mean=(0, 0, 0, 0),
    )
    return replace(
        context, update_identity=context.content_identity, state_identity=context.content_identity
    )


@dataclass(frozen=True, slots=True)
class TeacherObservation:
    """One retained teacher proposal plus its independently supplied target status."""

    target_role: str
    teacher_provider_identity: str
    teacher_model_identity: str
    teacher_generation_identity: str
    teacher_architecture_identity: str
    teacher_operating_envelope_identity: str
    snapshot_identity: str
    query_features: tuple[int, ...]
    context_state_identity: str
    source_observation_identities: tuple[str, ...]
    teacher_decision: str
    teacher_confidence: int
    teacher_abstained: bool
    teacher_escalation_reason: str | None
    reasoning_iterations: int
    teacher_operations: int
    decision_identity: str
    target_status: TargetStatus
    independent_target: str | None = None
    independent_target_identity: str | None = None
    observation_identity: str = ""
    authority: str = AUTHORITY

    def __post_init__(self) -> None:
        _text(self.target_role, "target_role")
        _text(self.teacher_provider_identity, "teacher_provider_identity")
        for name in (
            "teacher_model_identity",
            "teacher_generation_identity",
            "teacher_architecture_identity",
            "teacher_operating_envelope_identity",
            "snapshot_identity",
            "context_state_identity",
            "decision_identity",
        ):
            _identity(getattr(self, name), name)
        _features(self.query_features, "query_features")
        if not self.source_observation_identities or any(
            not isinstance(item, str) or not item.strip()
            for item in self.source_observation_identities
        ):
            raise OneStepSpecialistError("source observation identities are required")
        _text(self.teacher_decision, "teacher_decision", 128)
        if not 0 <= self.teacher_confidence <= 1000:
            raise OneStepSpecialistError("teacher_confidence is outside [0, 1000]")
        if self.teacher_abstained != (self.teacher_decision == "ABSTAIN"):
            raise OneStepSpecialistError("teacher abstention and decision disagree")
        if self.reasoning_iterations < 1 or self.teacher_operations < 1:
            raise OneStepSpecialistError("teacher measurements are invalid")
        if self.independent_target_identity is not None:
            _identity(self.independent_target_identity, "independent_target_identity")
        if self.target_status == TargetStatus.VERIFIED:
            if self.independent_target is None or self.teacher_abstained:
                raise OneStepSpecialistError("verified targets require a non-abstaining teacher")
            if self.teacher_decision != self.independent_target:
                raise OneStepSpecialistError("verified target does not agree with teacher proposal")
        elif self.target_status == TargetStatus.ABSTENTION and not self.teacher_abstained:
            raise OneStepSpecialistError("abstention target requires an abstaining teacher")
        if self.authority != AUTHORITY:
            raise OneStepSpecialistError("teacher observations are diagnostic-only")
        if self.observation_identity and self.observation_identity != self.content_identity:
            raise OneStepSpecialistError("teacher observation identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": DISTILLATION_RECORD_SCHEMA,
            "target_role": self.target_role,
            "teacher_provider_identity": self.teacher_provider_identity,
            "teacher_model_identity": self.teacher_model_identity,
            "teacher_generation_identity": self.teacher_generation_identity,
            "teacher_architecture_identity": self.teacher_architecture_identity,
            "teacher_operating_envelope_identity": self.teacher_operating_envelope_identity,
            "snapshot_identity": self.snapshot_identity,
            "query_features": list(self.query_features),
            "context_state_identity": self.context_state_identity,
            "source_observation_identities": list(self.source_observation_identities),
            "teacher_decision": self.teacher_decision,
            "teacher_confidence_milli": self.teacher_confidence,
            "teacher_confidence": self.teacher_confidence / 1000,
            "teacher_abstained": self.teacher_abstained,
            "teacher_escalation_reason": self.teacher_escalation_reason,
            "reasoning_iterations": self.reasoning_iterations,
            "teacher_operations": self.teacher_operations,
            "decision_identity": self.decision_identity,
            "target_status": self.target_status.value,
            "independent_target": self.independent_target,
            "independent_target_identity": self.independent_target_identity,
            "authority": self.authority,
            "semantics": "retained-teacher-observation; independently-targeted; not-a-verdict",
        }
        if include_identity:
            value["observation_identity"] = self.observation_identity or self.content_identity
        return value

    def serialize(self) -> bytes:
        return canonical_json(self.to_dict())

    @classmethod
    def load(cls, payload: bytes | Mapping[str, Any]) -> TeacherObservation:
        try:
            value = json.loads(payload) if isinstance(payload, bytes) else dict(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise OneStepSpecialistError("distillation record is not valid JSON") from error
        if not isinstance(value, dict) or value.get("schema") != DISTILLATION_RECORD_SCHEMA:
            raise OneStepSpecialistError("unsupported distillation record schema")
        _reject_authority(value)
        expected_keys = {
            "schema",
            "target_role",
            "teacher_provider_identity",
            "teacher_model_identity",
            "teacher_generation_identity",
            "teacher_architecture_identity",
            "teacher_operating_envelope_identity",
            "snapshot_identity",
            "query_features",
            "context_state_identity",
            "source_observation_identities",
            "teacher_decision",
            "teacher_confidence_milli",
            "teacher_confidence",
            "teacher_abstained",
            "teacher_escalation_reason",
            "reasoning_iterations",
            "teacher_operations",
            "decision_identity",
            "target_status",
            "independent_target",
            "independent_target_identity",
            "authority",
            "semantics",
            "observation_identity",
        }
        if (
            set(value) != expected_keys
            or value.get("semantics")
            != "retained-teacher-observation; independently-targeted; not-a-verdict"
        ):
            raise OneStepSpecialistError("distillation record contains unknown or missing fields")
        confidence = value.get("teacher_confidence")
        confidence_milli = value.get("teacher_confidence_milli")
        if (
            not isinstance(confidence, (int, float))
            or round(float(confidence) * 1000) != confidence_milli
        ):
            raise OneStepSpecialistError("teacher confidence representations disagree")
        record = cls(
            target_role=value.get("target_role"),
            teacher_provider_identity=value.get("teacher_provider_identity"),
            teacher_model_identity=value.get("teacher_model_identity"),
            teacher_generation_identity=value.get("teacher_generation_identity"),
            teacher_architecture_identity=value.get("teacher_architecture_identity"),
            teacher_operating_envelope_identity=value.get("teacher_operating_envelope_identity"),
            snapshot_identity=value.get("snapshot_identity"),
            query_features=tuple(value.get("query_features", ())),
            context_state_identity=value.get("context_state_identity"),
            source_observation_identities=tuple(value.get("source_observation_identities", ())),
            teacher_decision=value.get("teacher_decision"),
            teacher_confidence=confidence_milli,
            teacher_abstained=value.get("teacher_abstained"),
            teacher_escalation_reason=value.get("teacher_escalation_reason"),
            reasoning_iterations=value.get("reasoning_iterations"),
            teacher_operations=value.get("teacher_operations"),
            decision_identity=value.get("decision_identity"),
            target_status=TargetStatus(value.get("target_status")),
            independent_target=value.get("independent_target"),
            independent_target_identity=value.get("independent_target_identity"),
            authority=value.get("authority"),
            observation_identity=value.get("observation_identity"),
        )
        return record


def collect_teacher_observation(
    teacher: RecurrentSpecialistModel,
    row: Mapping[str, Any],
    *,
    independent_target: str | None = None,
    context: SpecialistContextState | None = None,
) -> TeacherObservation:
    """Run the bounded teacher once and retain, but do not promote, its proposal."""

    features = _features(row.get("features"), "features")
    record_id = _text(row.get("record_id"), "record_id")
    source_identity = _source_identity(row)
    snapshot_identity = canonical_digest({"record_id": record_id, "features": list(features)})
    context_value = context or SpecialistContextState(
        provider_identity=teacher.provider_id,
        generation_identity=teacher.generation_identity,
        role_identity=teacher.target_role,
        source_observation_identities=(),
        feature_mean=(0, 0, 0, 0),
    )
    context_value = replace(
        context_value,
        update_identity=context_value.update_identity or context_value.content_identity,
        state_identity=context_value.state_identity or context_value.content_identity,
    )
    if (
        context_value.role_identity != teacher.target_role
        or context_value.generation_identity != teacher.generation_identity
    ):
        raise OneStepSpecialistError("teacher context is bound to another role or generation")
    request_identity = canonical_digest(
        {
            "snapshot_identity": snapshot_identity,
            "context_state_identity": context_value.state_identity,
        }
    )
    result = teacher.infer(
        features,
        context=context_value,
        request_identity=request_identity,
        source_observation_identities=(source_identity,),
    )
    target = independent_target if independent_target is not None else row.get("expected")
    if target is not None and not isinstance(target, str):
        raise OneStepSpecialistError("independent_target must be a string or null")
    target_identity = (
        canonical_digest({"snapshot_identity": snapshot_identity, "independent_target": target})
        if target is not None
        else None
    )
    if target is None:
        status = TargetStatus.UNKNOWN
    elif target == "ABSTAIN":
        status = TargetStatus.ABSTENTION if result.abstained else TargetStatus.REJECTED_TEACHER
    elif result.abstained:
        status = TargetStatus.ABSTENTION
    elif result.decision == target:
        status = TargetStatus.VERIFIED
    else:
        status = TargetStatus.REJECTED_TEACHER
    observation = TeacherObservation(
        target_role=teacher.target_role,
        teacher_provider_identity=teacher.provider_id,
        teacher_model_identity=teacher.model_identity or teacher.content_identity,
        teacher_generation_identity=teacher.generation_identity,
        teacher_architecture_identity=teacher.architecture_identity,
        teacher_operating_envelope_identity=teacher.operating_envelope.envelope_identity
        or teacher.operating_envelope.content_identity,
        snapshot_identity=snapshot_identity,
        query_features=features,
        context_state_identity=context_value.state_identity,
        source_observation_identities=(source_identity,),
        teacher_decision=result.decision,
        teacher_confidence=result.confidence,
        teacher_abstained=result.abstained,
        teacher_escalation_reason=result.escalation_reason,
        reasoning_iterations=result.reasoning_iterations,
        teacher_operations=result.operations,
        decision_identity=result.decision_identity or result.content_identity,
        target_status=status,
        independent_target=target,
        independent_target_identity=target_identity,
    )
    return replace(observation, observation_identity=observation.content_identity)


@dataclass(frozen=True, slots=True)
class DistillationDataset:
    """A bounded dataset retaining every observation and selecting only verified rows."""

    target_role: str
    teacher_provider_identity: str
    teacher_model_identity: str
    teacher_generation_identity: str
    teacher_architecture_identity: str
    teacher_operating_envelope_identity: str
    observations: tuple[TeacherObservation, ...]
    training_observation_identities: tuple[str, ...]
    transform_identity: str
    dataset_identity: str = ""
    distillation_identity: str = ""
    authority: str = AUTHORITY

    def __post_init__(self) -> None:
        _text(self.target_role, "target_role")
        _text(self.teacher_provider_identity, "teacher_provider_identity")
        for name in (
            "teacher_model_identity",
            "teacher_generation_identity",
            "teacher_architecture_identity",
            "teacher_operating_envelope_identity",
            "transform_identity",
        ):
            _identity(getattr(self, name), name)
        if not self.observations or len(self.observations) > MAX_ROWS:
            raise OneStepSpecialistError("distillation dataset is outside its row bound")
        for observation in self.observations:
            if (
                observation.target_role != self.target_role
                or observation.teacher_provider_identity != self.teacher_provider_identity
                or observation.teacher_model_identity != self.teacher_model_identity
                or observation.teacher_generation_identity != self.teacher_generation_identity
                or observation.teacher_architecture_identity != self.teacher_architecture_identity
                or observation.teacher_operating_envelope_identity
                != self.teacher_operating_envelope_identity
            ):
                raise OneStepSpecialistError(
                    "distillation observations have mismatched teacher lineage"
                )
        available = {
            item.observation_identity or item.content_identity for item in self.observations
        }
        if (
            not self.training_observation_identities
            or not set(self.training_observation_identities) <= available
        ):
            raise OneStepSpecialistError("training observations must be retained dataset records")
        if any(
            item.target_status != TargetStatus.VERIFIED
            for item in self.observations
            if (item.observation_identity or item.content_identity)
            in self.training_observation_identities
        ):
            raise OneStepSpecialistError(
                "rejected, unknown, or abstaining observations cannot train the student"
            )
        if self.authority != AUTHORITY:
            raise OneStepSpecialistError("distillation datasets are diagnostic-only")
        if self.dataset_identity and self.dataset_identity != self.content_identity:
            raise OneStepSpecialistError("dataset identity does not match content")
        if (
            self.distillation_identity
            and self.distillation_identity != self._distillation_content_identity
        ):
            raise OneStepSpecialistError("distillation identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    @property
    def _distillation_content_identity(self) -> str:
        return canonical_digest(
            {
                "schema": "mnel-distillation-transform/0.1",
                "teacher_model_identity": self.teacher_model_identity,
                "teacher_generation_identity": self.teacher_generation_identity,
                "transform_identity": self.transform_identity,
                "training_observation_identities": list(self.training_observation_identities),
            }
        )

    @property
    def training_observations(self) -> tuple[TeacherObservation, ...]:
        selected = set(self.training_observation_identities)
        return tuple(
            item
            for item in self.observations
            if (item.observation_identity or item.content_identity) in selected
        )

    @property
    def training_targets(self) -> tuple[dict[str, Any], ...]:
        """Expose the exact teacher-derived target construction for inspection."""

        labels = tuple(
            sorted(
                {
                    item.independent_target
                    for item in self.training_observations
                    if item.independent_target
                }
            )
        )
        return tuple(
            {
                "observation_identity": item.observation_identity or item.content_identity,
                "features": list(item.query_features),
                "independent_target": item.independent_target,
                "teacher_target": item.teacher_decision,
                "teacher_confidence_milli": item.teacher_confidence,
                "target_distribution": list(_soft_target(item, labels, True)),
                "target_construction": "teacher-decision-and-confidence",
            }
            for item in self.training_observations
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": DISTILLATION_DATASET_SCHEMA,
            "target_role": self.target_role,
            "teacher_provider_identity": self.teacher_provider_identity,
            "teacher_model_identity": self.teacher_model_identity,
            "teacher_generation_identity": self.teacher_generation_identity,
            "teacher_architecture_identity": self.teacher_architecture_identity,
            "teacher_operating_envelope_identity": self.teacher_operating_envelope_identity,
            "observations": [item.to_dict() for item in self.observations],
            "training_observation_identities": list(self.training_observation_identities),
            "training_targets": list(self.training_targets),
            "transform_identity": self.transform_identity,
            "authority": self.authority,
            "semantics": "source-preserving-distillation-dataset; verified-target-filtered; not-a-verdict",
        }
        if include_identity:
            value["dataset_identity"] = self.dataset_identity or self.content_identity
            value["distillation_identity"] = (
                self.distillation_identity or self._distillation_content_identity
            )
        return value

    def serialize(self) -> bytes:
        return canonical_json(self.to_dict())

    @classmethod
    def load(cls, payload: bytes | Mapping[str, Any]) -> DistillationDataset:
        try:
            value = json.loads(payload) if isinstance(payload, bytes) else dict(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise OneStepSpecialistError("distillation dataset is not valid JSON") from error
        if not isinstance(value, dict) or value.get("schema") != DISTILLATION_DATASET_SCHEMA:
            raise OneStepSpecialistError("unsupported distillation dataset schema")
        _reject_authority(value)
        expected_keys = {
            "schema",
            "target_role",
            "teacher_provider_identity",
            "teacher_model_identity",
            "teacher_generation_identity",
            "teacher_architecture_identity",
            "teacher_operating_envelope_identity",
            "observations",
            "training_observation_identities",
            "training_targets",
            "transform_identity",
            "authority",
            "semantics",
            "dataset_identity",
            "distillation_identity",
        }
        if (
            set(value) != expected_keys
            or value.get("semantics")
            != "source-preserving-distillation-dataset; verified-target-filtered; not-a-verdict"
        ):
            raise OneStepSpecialistError("distillation dataset contains unknown or missing fields")
        observations = tuple(
            TeacherObservation.load(item) for item in value.get("observations", ())
        )
        dataset = cls(
            target_role=value.get("target_role"),
            teacher_provider_identity=value.get("teacher_provider_identity"),
            teacher_model_identity=value.get("teacher_model_identity"),
            teacher_generation_identity=value.get("teacher_generation_identity"),
            teacher_architecture_identity=value.get("teacher_architecture_identity"),
            teacher_operating_envelope_identity=value.get("teacher_operating_envelope_identity"),
            observations=observations,
            training_observation_identities=tuple(value.get("training_observation_identities", ())),
            transform_identity=value.get("transform_identity"),
            dataset_identity=value.get("dataset_identity"),
            distillation_identity=value.get("distillation_identity"),
            authority=value.get("authority"),
        )
        if tuple(value.get("training_targets", ())) != dataset.training_targets:
            raise OneStepSpecialistError("distillation targets do not match retained observations")
        return dataset


def build_distillation_dataset(
    observations: Sequence[TeacherObservation],
    *,
    target_role: str | None = None,
    transform_identity: str | None = None,
) -> DistillationDataset:
    """Build a dataset with an explicit role binding and a stable transform identity."""

    if not observations:
        raise OneStepSpecialistError("teacher observation set is empty")
    first = observations[0]
    if any(item.target_role != (target_role or first.target_role) for item in observations):
        raise OneStepSpecialistError("distillation observations have mismatched target roles")
    transform = transform_identity or canonical_digest(
        {
            "schema": "mnel-distillation-transform/0.1",
            "operation": "verified-teacher-decision-with-confidence-targets",
            "version": "0.1",
        }
    )
    selected = tuple(
        item.observation_identity or item.content_identity
        for item in observations
        if item.target_status == TargetStatus.VERIFIED
    )
    if not selected:
        raise OneStepSpecialistError(
            "distillation dataset has no independently verified training rows"
        )
    dataset = DistillationDataset(
        target_role=target_role or first.target_role,
        teacher_provider_identity=first.teacher_provider_identity,
        teacher_model_identity=first.teacher_model_identity,
        teacher_generation_identity=first.teacher_generation_identity,
        teacher_architecture_identity=first.teacher_architecture_identity,
        teacher_operating_envelope_identity=first.teacher_operating_envelope_identity,
        observations=tuple(observations),
        training_observation_identities=selected,
        transform_identity=transform,
    )
    return replace(
        dataset,
        dataset_identity=dataset.content_identity,
        distillation_identity=dataset._distillation_content_identity,
    )


build_role_distillation_dataset = build_distillation_dataset


@dataclass(frozen=True, slots=True)
class StudentOperatingEnvelope:
    minimum_confidence: int = 700
    maximum_distance: int = 1400
    maximum_query_abs: int = 1000
    maximum_context_observations: int = MAX_CONTEXT_OBSERVATIONS
    envelope_identity: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_confidence <= 1000:
            raise OneStepSpecialistError("student minimum_confidence is invalid")
        if (
            self.maximum_distance < 1
            or not 1 <= self.maximum_context_observations <= MAX_CONTEXT_OBSERVATIONS
        ):
            raise OneStepSpecialistError("student operating envelope is invalid")
        if not 1 <= self.maximum_query_abs <= 1000:
            raise OneStepSpecialistError("student maximum_query_abs is invalid")
        if self.envelope_identity and self.envelope_identity != self.content_identity:
            raise OneStepSpecialistError("student envelope identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "minimum_confidence": self.minimum_confidence,
            "maximum_distance": self.maximum_distance,
            "maximum_query_abs": self.maximum_query_abs,
            "maximum_context_observations": self.maximum_context_observations,
        }
        if include_identity:
            value["envelope_identity"] = self.envelope_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class StudentCalibration:
    target_role: str
    calibration_dataset_identity: str
    minimum_confidence: int
    maximum_distance: int
    method: str = "heldout-confidence-and-distance-envelope"
    calibration_identity: str = ""

    def __post_init__(self) -> None:
        _text(self.target_role, "target_role")
        _identity(self.calibration_dataset_identity, "calibration_dataset_identity")
        if not 0 <= self.minimum_confidence <= 1000 or self.maximum_distance < 1:
            raise OneStepSpecialistError("student calibration thresholds are invalid")
        if self.calibration_identity and self.calibration_identity != self.content_identity:
            raise OneStepSpecialistError("student calibration identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-one-step-specialist-calibration/0.1",
            "target_role": self.target_role,
            "calibration_dataset_identity": self.calibration_dataset_identity,
            "minimum_confidence": self.minimum_confidence,
            "maximum_distance": self.maximum_distance,
            "method": self.method,
            "authority": AUTHORITY,
            "semantics": "calibration-observation; not-a-verdict",
        }
        if include_identity:
            value["calibration_identity"] = self.calibration_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class OneStepStudentDecision:
    request_identity: str
    context_state_identity: str
    target_role: str
    decision: str
    diagnostic_proposal: Mapping[str, Any]
    confidence: int
    abstained: bool
    escalation_reason: str | None
    out_of_distribution: bool
    student_model_identity: str
    student_architecture_identity: str
    teacher_model_identity: str
    teacher_generation_identity: str
    distillation_identity: str
    calibration_identity: str
    operating_envelope_identity: str
    prepared_features: tuple[int, ...]
    logits: tuple[float, ...]
    learned_forward_passes: int
    preprocessing_operations: int
    inference_operations: int
    preprocessing_elapsed_ns: int
    learned_elapsed_ns: int
    elapsed_ns: int
    source_observation_identities: tuple[str, ...] = ()
    authority: str = AUTHORITY
    decision_identity: str = ""

    def __post_init__(self) -> None:
        for name in (
            "request_identity",
            "context_state_identity",
            "student_model_identity",
            "student_architecture_identity",
            "teacher_model_identity",
            "teacher_generation_identity",
            "distillation_identity",
            "calibration_identity",
            "operating_envelope_identity",
        ):
            _identity(getattr(self, name), name)
        _text(self.target_role, "target_role")
        _text(self.decision, "decision", 128)
        _reject_authority(self.diagnostic_proposal)
        if self.diagnostic_proposal.get("kind") != "structured-diagnostic-proposal":
            raise OneStepSpecialistError("student proposal kind is invalid")
        if not 0 <= self.confidence <= 1000:
            raise OneStepSpecialistError("student confidence is invalid")
        if self.abstained != (self.decision == "ABSTAIN"):
            raise OneStepSpecialistError("student abstention and decision disagree")
        if not 0 <= self.learned_forward_passes <= 1:
            raise OneStepSpecialistError("one-step student performed more than one learned pass")
        if self.preprocessing_operations < 1 or self.inference_operations < 0:
            raise OneStepSpecialistError("student operation measurements are invalid")
        if any(
            item < 0
            for item in (self.preprocessing_elapsed_ns, self.learned_elapsed_ns, self.elapsed_ns)
        ):
            raise OneStepSpecialistError("student latency measurements are invalid")
        _features(self.prepared_features, "prepared_features")
        _vector(self.logits, "logits", len(self.logits))
        if self.authority != AUTHORITY:
            raise OneStepSpecialistError("student decisions are diagnostic-only")
        if self.decision_identity and self.decision_identity != self.content_identity:
            raise OneStepSpecialistError("student decision identity does not match content")

    @property
    def content_identity(self) -> str:
        value = self.to_dict(include_identity=False)
        value.pop("elapsed_ns", None)
        value.pop("preprocessing_elapsed_ns", None)
        value.pop("learned_elapsed_ns", None)
        return canonical_digest(value)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": DECISION_SCHEMA,
            "request_identity": self.request_identity,
            "context_state_identity": self.context_state_identity,
            "target_role": self.target_role,
            "decision": self.decision,
            "diagnostic_proposal": dict(self.diagnostic_proposal),
            "confidence": self.confidence / 1000,
            "confidence_milli": self.confidence,
            "abstained": self.abstained,
            "escalation_reason": self.escalation_reason,
            "out_of_distribution": self.out_of_distribution,
            "student_model_identity": self.student_model_identity,
            "student_architecture_identity": self.student_architecture_identity,
            "teacher_model_identity": self.teacher_model_identity,
            "teacher_generation_identity": self.teacher_generation_identity,
            "distillation_identity": self.distillation_identity,
            "calibration_identity": self.calibration_identity,
            "operating_envelope_identity": self.operating_envelope_identity,
            "prepared_features": list(self.prepared_features),
            "logits": list(self.logits),
            "learned_forward_passes": self.learned_forward_passes,
            "preprocessing_operations": self.preprocessing_operations,
            "inference_operations": self.inference_operations,
            "preprocessing_elapsed_ns": self.preprocessing_elapsed_ns,
            "learned_elapsed_ns": self.learned_elapsed_ns,
            "elapsed_ns": self.elapsed_ns,
            "source_observation_identities": list(self.source_observation_identities),
            "authority": self.authority,
            "semantics": "one-step-structured-diagnostic-proposal; not-a-verdict",
        }
        if include_identity:
            value["decision_identity"] = self.decision_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class OneStepStudentModel:
    target_role: str
    teacher_provider_identity: str
    teacher_model_identity: str
    teacher_generation_identity: str
    teacher_architecture_identity: str
    teacher_operating_envelope_identity: str
    architecture_identity: str
    training_code_identity: str
    training_dataset_identity: str
    distillation_identity: str
    training_spec_identity: str
    checkpoint_identity: str
    calibration_identity: str
    operating_envelope: StudentOperatingEnvelope
    class_labels: tuple[str, ...]
    input_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]
    feature_center: tuple[int, ...]
    training_record_ids: tuple[str, ...]
    source_observation_identities: tuple[str, ...]
    model_identity: str = ""
    artifact_identity: str = ""
    provider_id: str = PROVIDER_ID
    provider_abi: str = PROVIDER_ABI
    authority: str = AUTHORITY

    def __post_init__(self) -> None:
        _text(self.target_role, "target_role")
        _text(self.teacher_provider_identity, "teacher_provider_identity")
        for name in (
            "teacher_model_identity",
            "teacher_generation_identity",
            "teacher_architecture_identity",
            "teacher_operating_envelope_identity",
            "architecture_identity",
            "training_code_identity",
            "training_dataset_identity",
            "distillation_identity",
            "training_spec_identity",
            "checkpoint_identity",
            "calibration_identity",
        ):
            _identity(getattr(self, name), name)
        if not self.class_labels or len(self.class_labels) > 8:
            raise OneStepSpecialistError("student class label count is invalid")
        if tuple(sorted(self.class_labels)) != self.class_labels or len(
            set(self.class_labels)
        ) != len(self.class_labels):
            raise OneStepSpecialistError("student class labels must be sorted and unique")
        _matrix(self.input_weights, "input_weights", HIDDEN_WIDTH, DIMENSIONS)
        _vector(self.hidden_bias, "hidden_bias", HIDDEN_WIDTH)
        _matrix(self.output_weights, "output_weights", len(self.class_labels), HIDDEN_WIDTH)
        _vector(self.output_bias, "output_bias", len(self.class_labels))
        _features(self.feature_center, "feature_center")
        if not self.training_record_ids or not self.source_observation_identities:
            raise OneStepSpecialistError("student training/source identities are required")
        if self.authority != AUTHORITY or self.provider_abi != PROVIDER_ABI:
            raise OneStepSpecialistError("student authority or ABI is invalid")
        if self.model_identity and self.model_identity != self.content_identity:
            raise OneStepSpecialistError("student model identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    @property
    def model_size_bytes(self) -> int:
        return len(canonical_json(self.to_dict()))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "provider_id": self.provider_id,
            "provider_abi": self.provider_abi,
            "target_role": self.target_role,
            "teacher_provider_identity": self.teacher_provider_identity,
            "teacher_model_identity": self.teacher_model_identity,
            "teacher_generation_identity": self.teacher_generation_identity,
            "teacher_architecture_identity": self.teacher_architecture_identity,
            "teacher_operating_envelope_identity": self.teacher_operating_envelope_identity,
            "architecture_identity": self.architecture_identity,
            "architecture": {
                "kind": "tiny-affine-tanh",
                "input_dimensions": DIMENSIONS,
                "hidden_width": HIDDEN_WIDTH,
                "output_dimensions": len(self.class_labels),
                "learned_forward_passes": 1,
                "context_state": "persistent-derived-summary",
                "reasoning_state": "none-after-preparation",
            },
            "training_code_identity": self.training_code_identity,
            "training_dataset_identity": self.training_dataset_identity,
            "distillation_identity": self.distillation_identity,
            "training_spec_identity": self.training_spec_identity,
            "checkpoint_identity": self.checkpoint_identity,
            "calibration_identity": self.calibration_identity,
            "operating_envelope": self.operating_envelope.to_dict(),
            "class_labels": list(self.class_labels),
            "input_weights": [list(row) for row in self.input_weights],
            "hidden_bias": list(self.hidden_bias),
            "output_weights": [list(row) for row in self.output_weights],
            "output_bias": list(self.output_bias),
            "feature_center": list(self.feature_center),
            "training_record_ids": list(self.training_record_ids),
            "source_observation_identities": list(self.source_observation_identities),
            "authority": self.authority,
            "semantics": "identity-bound-one-step-learned-specialist; diagnostic-only; not-a-verdict",
        }
        if include_identity:
            value["model_identity"] = self.model_identity or self.content_identity
        return value

    def serialize(self) -> bytes:
        value = self.to_dict()
        artifact_identity = canonical_digest(value)
        value["artifact_identity"] = artifact_identity
        object.__setattr__(self, "artifact_identity", artifact_identity)
        return canonical_json(value)

    @classmethod
    def load(cls, payload: bytes | Mapping[str, Any]) -> OneStepStudentModel:
        try:
            value = json.loads(payload) if isinstance(payload, bytes) else dict(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise OneStepSpecialistError("student artifact is not valid JSON") from error
        if not isinstance(value, dict) or value.get("schema") != ARTIFACT_SCHEMA:
            raise OneStepSpecialistError("unsupported one-step artifact schema")
        _reject_authority(value)
        supplied_artifact = value.pop("artifact_identity", None)
        if not isinstance(supplied_artifact, str):
            raise OneStepSpecialistError("student artifact identity is missing")
        if supplied_artifact != canonical_digest(value):
            raise OneStepSpecialistError("student artifact bytes do not match artifact identity")
        expected_keys = {
            "schema",
            "provider_id",
            "provider_abi",
            "target_role",
            "teacher_provider_identity",
            "teacher_model_identity",
            "teacher_generation_identity",
            "teacher_architecture_identity",
            "teacher_operating_envelope_identity",
            "architecture_identity",
            "architecture",
            "training_code_identity",
            "training_dataset_identity",
            "distillation_identity",
            "training_spec_identity",
            "checkpoint_identity",
            "calibration_identity",
            "operating_envelope",
            "class_labels",
            "input_weights",
            "hidden_bias",
            "output_weights",
            "output_bias",
            "feature_center",
            "training_record_ids",
            "source_observation_identities",
            "authority",
            "semantics",
            "model_identity",
        }
        if set(value) != expected_keys:
            raise OneStepSpecialistError("student artifact contains unknown or missing fields")
        architecture = value.get("architecture")
        if (
            not isinstance(architecture, dict)
            or architecture.get("kind") != "tiny-affine-tanh"
            or architecture.get("learned_forward_passes") != 1
        ):
            raise OneStepSpecialistError(
                "student artifact does not declare exactly one learned pass"
            )
        envelope_value = value.get("operating_envelope")
        if not isinstance(envelope_value, dict):
            raise OneStepSpecialistError("student operating envelope is missing")
        envelope = StudentOperatingEnvelope(
            **{key: item for key, item in envelope_value.items() if key != "envelope_identity"},
            envelope_identity=envelope_value.get("envelope_identity", ""),
        )
        model = cls(
            target_role=value.get("target_role"),
            teacher_provider_identity=value.get("teacher_provider_identity"),
            teacher_model_identity=value.get("teacher_model_identity"),
            teacher_generation_identity=value.get("teacher_generation_identity"),
            teacher_architecture_identity=value.get("teacher_architecture_identity"),
            teacher_operating_envelope_identity=value.get("teacher_operating_envelope_identity"),
            architecture_identity=value.get("architecture_identity"),
            training_code_identity=value.get("training_code_identity"),
            training_dataset_identity=value.get("training_dataset_identity"),
            distillation_identity=value.get("distillation_identity"),
            training_spec_identity=value.get("training_spec_identity"),
            checkpoint_identity=value.get("checkpoint_identity"),
            calibration_identity=value.get("calibration_identity"),
            operating_envelope=envelope,
            class_labels=tuple(value.get("class_labels", ())),
            input_weights=tuple(tuple(row) for row in value.get("input_weights", ())),
            hidden_bias=tuple(value.get("hidden_bias", ())),
            output_weights=tuple(tuple(row) for row in value.get("output_weights", ())),
            output_bias=tuple(value.get("output_bias", ())),
            feature_center=tuple(value.get("feature_center", ())),
            training_record_ids=tuple(value.get("training_record_ids", ())),
            source_observation_identities=tuple(value.get("source_observation_identities", ())),
            model_identity=value.get("model_identity", ""),
            artifact_identity=supplied_artifact,
            provider_id=value.get("provider_id", ""),
            provider_abi=value.get("provider_abi", ""),
            authority=value.get("authority", ""),
        )
        if model.model_identity != model.content_identity:
            raise OneStepSpecialistError("student model identity is invalid")
        return model

    def _prepare(self, query: Sequence[int], context: SpecialistContextState) -> tuple[int, ...]:
        query_value = _features(query, "query")
        if any(abs(item) > self.operating_envelope.maximum_query_abs for item in query_value):
            raise OneStepSpecialistError("query exceeds student operating envelope")
        if (
            context.role_identity != self.target_role
            or context.generation_identity != self.teacher_generation_identity
        ):
            raise OneStepSpecialistError("context is bound to another student role or generation")
        if context.provider_identity != self.teacher_provider_identity:
            raise OneStepSpecialistError("context provider identity is incompatible with student")
        if (
            len(context.source_observation_identities)
            > self.operating_envelope.maximum_context_observations
        ):
            raise OneStepSpecialistError("context exceeds student operating envelope")
        return tuple(
            max(-1000, min(1000, query_value[index] + context.feature_mean[index] // 16))
            for index in range(DIMENSIONS)
        )

    def _forward(self, prepared: Sequence[int]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        normalized = [value / 1000.0 for value in prepared]
        hidden = tuple(
            math.tanh(
                self.hidden_bias[row]
                + sum(self.input_weights[row][col] * normalized[col] for col in range(DIMENSIONS))
            )
            for row in range(HIDDEN_WIDTH)
        )
        logits = tuple(
            self.output_bias[row]
            + sum(self.output_weights[row][col] * hidden[col] for col in range(HIDDEN_WIDTH))
            for row in range(len(self.class_labels))
        )
        return logits, hidden

    def _raw_score(
        self, prepared: Sequence[int]
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        logits, hidden = self._forward(prepared)
        highest = max(logits)
        exponentials = tuple(math.exp(value - highest) for value in logits)
        total = sum(exponentials)
        return logits, hidden, tuple(value / total for value in exponentials)

    def infer(
        self,
        query: Sequence[int],
        *,
        context: SpecialistContextState | None = None,
        request_identity: str | None = None,
        source_observation_identities: Sequence[str] = (),
    ) -> OneStepStudentDecision:
        started = time.perf_counter_ns()
        context_value = context or _context_for_student(self)
        invalid_context = not isinstance(context_value, SpecialistContextState)
        context_identity = (
            canonical_digest({"invalid_context_type": type(context_value).__name__})
            if invalid_context
            else context_value.state_identity or context_value.content_identity
        )
        request = request_identity or canonical_digest(
            {"query": list(query), "context_state_identity": context_identity}
        )
        _identity(request, "request_identity")
        prep_started = time.perf_counter_ns()
        try:
            if invalid_context:
                raise OneStepSpecialistError("context must be a SpecialistContextState")
            prepared = self._prepare(query, context_value)
        except OneStepSpecialistError as error:
            elapsed = time.perf_counter_ns() - started
            result = OneStepStudentDecision(
                request_identity=request,
                context_state_identity=context_identity,
                target_role=self.target_role,
                decision="ABSTAIN",
                diagnostic_proposal={
                    "kind": "structured-diagnostic-proposal",
                    "role": self.target_role,
                    "label": "ABSTAIN",
                },
                confidence=0,
                abstained=True,
                escalation_reason=f"unsupported-envelope:{error}",
                out_of_distribution=False,
                student_model_identity=self.model_identity or self.content_identity,
                student_architecture_identity=self.architecture_identity,
                teacher_model_identity=self.teacher_model_identity,
                teacher_generation_identity=self.teacher_generation_identity,
                distillation_identity=self.distillation_identity,
                calibration_identity=self.calibration_identity,
                operating_envelope_identity=self.operating_envelope.envelope_identity
                or self.operating_envelope.content_identity,
                prepared_features=(0, 0, 0, 0),
                logits=tuple(0.0 for _ in self.class_labels),
                learned_forward_passes=0,
                preprocessing_operations=1,
                inference_operations=0,
                preprocessing_elapsed_ns=elapsed,
                learned_elapsed_ns=0,
                elapsed_ns=elapsed,
                source_observation_identities=tuple(source_observation_identities),
            )
            return replace(result, decision_identity=result.content_identity)
        preprocessing_elapsed = time.perf_counter_ns() - prep_started
        distance = sum(
            abs(prepared[index] - self.feature_center[index]) for index in range(DIMENSIONS)
        )
        if distance > self.operating_envelope.maximum_distance:
            elapsed = time.perf_counter_ns() - started
            result = OneStepStudentDecision(
                request_identity=request,
                context_state_identity=context_identity,
                target_role=self.target_role,
                decision="ABSTAIN",
                diagnostic_proposal={
                    "kind": "structured-diagnostic-proposal",
                    "role": self.target_role,
                    "label": "ABSTAIN",
                },
                confidence=0,
                abstained=True,
                escalation_reason="out-of-distribution-distance",
                out_of_distribution=True,
                student_model_identity=self.model_identity or self.content_identity,
                student_architecture_identity=self.architecture_identity,
                teacher_model_identity=self.teacher_model_identity,
                teacher_generation_identity=self.teacher_generation_identity,
                distillation_identity=self.distillation_identity,
                calibration_identity=self.calibration_identity,
                operating_envelope_identity=self.operating_envelope.envelope_identity
                or self.operating_envelope.content_identity,
                prepared_features=prepared,
                logits=tuple(0.0 for _ in self.class_labels),
                learned_forward_passes=0,
                preprocessing_operations=5,
                inference_operations=0,
                preprocessing_elapsed_ns=preprocessing_elapsed,
                learned_elapsed_ns=0,
                elapsed_ns=elapsed,
                source_observation_identities=tuple(source_observation_identities),
            )
            return replace(result, decision_identity=result.content_identity)
        learned_started = time.perf_counter_ns()
        logits, _hidden, probabilities = self._raw_score(prepared)
        learned_elapsed = time.perf_counter_ns() - learned_started
        best_index = max(
            range(len(probabilities)), key=lambda index: (probabilities[index], -index)
        )
        confidence = max(0, min(1000, round(probabilities[best_index] * 1000)))
        abstention = None
        decision = self.class_labels[best_index]
        if confidence < self.operating_envelope.minimum_confidence:
            abstention = "insufficient-calibrated-confidence"
            decision = "ABSTAIN"
        elapsed = time.perf_counter_ns() - started
        result = OneStepStudentDecision(
            request_identity=request,
            context_state_identity=context_identity,
            target_role=self.target_role,
            decision=decision,
            diagnostic_proposal={
                "kind": "structured-diagnostic-proposal",
                "role": self.target_role,
                "label": decision,
                "class_index": best_index,
            },
            confidence=confidence,
            abstained=abstention is not None,
            escalation_reason=abstention,
            out_of_distribution=False,
            student_model_identity=self.model_identity or self.content_identity,
            student_architecture_identity=self.architecture_identity,
            teacher_model_identity=self.teacher_model_identity,
            teacher_generation_identity=self.teacher_generation_identity,
            distillation_identity=self.distillation_identity,
            calibration_identity=self.calibration_identity,
            operating_envelope_identity=self.operating_envelope.envelope_identity
            or self.operating_envelope.content_identity,
            prepared_features=prepared,
            logits=logits,
            learned_forward_passes=1,
            preprocessing_operations=5,
            inference_operations=HIDDEN_WIDTH * (DIMENSIONS + 1)
            + len(self.class_labels) * (HIDDEN_WIDTH + 1)
            + len(self.class_labels),
            preprocessing_elapsed_ns=preprocessing_elapsed,
            learned_elapsed_ns=learned_elapsed,
            elapsed_ns=elapsed,
            source_observation_identities=tuple(source_observation_identities),
        )
        return replace(result, decision_identity=result.content_identity)


def _initial_weight(seed: int, row: int, column: int) -> float:
    value = (seed * 1103515245 + row * 12345 + column * 2654435761) & 0xFFFFFFFF
    return ((value % 2001) - 1000) / 10000.0


def _soft_target(
    observation: TeacherObservation, labels: Sequence[str], distilled: bool
) -> tuple[float, ...]:
    if observation.independent_target not in labels:
        raise OneStepSpecialistError("training target label is not a student class")
    target_index = labels.index(observation.independent_target)
    if not distilled:
        return tuple(1.0 if index == target_index else 0.0 for index in range(len(labels)))
    teacher_index = labels.index(observation.teacher_decision)
    confidence = observation.teacher_confidence / 1000.0
    mass = max(0.5, min(1.0, confidence))
    remainder = (1.0 - mass) / max(1, len(labels) - 1)
    return tuple(mass if index == teacher_index else remainder for index in range(len(labels)))


def train_one_step_student(
    dataset: DistillationDataset,
    *,
    distilled: bool = True,
    seed: int = 17,
    epochs: int = 120,
    learning_rate: float = 0.35,
) -> OneStepStudentModel:
    """Train the tiny student from retained teacher targets or direct labels.

    ``distilled=True`` consumes teacher decision/confidence targets.  The control
    uses the same architecture and examples but only the independent labels.
    """

    if not isinstance(seed, int) or seed < 0 or epochs < 1 or epochs > MAX_EPOCHS:
        raise OneStepSpecialistError("student training budget or seed is invalid")
    if not 0.0 < learning_rate <= 2.0 or len(dataset.training_observations) > MAX_ROWS:
        raise OneStepSpecialistError("student training parameters are invalid")
    rows = dataset.training_observations
    labels = tuple(
        sorted({item.independent_target for item in rows if item.independent_target is not None})
    )
    if len(labels) < 2:
        raise OneStepSpecialistError("student training requires at least two target classes")
    input_weights = [
        [_initial_weight(seed, row, column) for column in range(DIMENSIONS)]
        for row in range(HIDDEN_WIDTH)
    ]
    hidden_bias = [_initial_weight(seed + 7, row, 0) for row in range(HIDDEN_WIDTH)]
    output_weights = [
        [_initial_weight(seed + 13, row, column) for column in range(HIDDEN_WIDTH)]
        for row in range(len(labels))
    ]
    output_bias = [_initial_weight(seed + 19, row, 0) for row in range(len(labels))]
    for _ in range(epochs):
        for observation in rows:
            x = [value / 1000.0 for value in observation.query_features]
            hidden = [
                math.tanh(
                    hidden_bias[row]
                    + sum(input_weights[row][col] * x[col] for col in range(DIMENSIONS))
                )
                for row in range(HIDDEN_WIDTH)
            ]
            logits = [
                output_bias[row]
                + sum(output_weights[row][col] * hidden[col] for col in range(HIDDEN_WIDTH))
                for row in range(len(labels))
            ]
            highest = max(logits)
            exponentials = [math.exp(value - highest) for value in logits]
            total = sum(exponentials)
            probabilities = [value / total for value in exponentials]
            target = _soft_target(observation, labels, distilled)
            gradient_logits = [probabilities[index] - target[index] for index in range(len(labels))]
            gradient_hidden = [
                sum(
                    gradient_logits[row] * output_weights[row][column] for row in range(len(labels))
                )
                for column in range(HIDDEN_WIDTH)
            ]
            for row in range(len(labels)):
                output_bias[row] -= learning_rate * gradient_logits[row]
                for column in range(HIDDEN_WIDTH):
                    output_weights[row][column] -= (
                        learning_rate * gradient_logits[row] * hidden[column]
                    )
            for row in range(HIDDEN_WIDTH):
                gradient = gradient_hidden[row] * (1.0 - hidden[row] * hidden[row])
                hidden_bias[row] -= learning_rate * gradient
                for column in range(DIMENSIONS):
                    input_weights[row][column] -= learning_rate * gradient * x[column]
    center = tuple(
        sum(item.query_features[index] for item in rows) // len(rows) for index in range(DIMENSIONS)
    )
    radius = max(
        sum(abs(item.query_features[index] - center[index]) for index in range(DIMENSIONS))
        for item in rows
    )
    spec = {
        "schema": "mnel-one-step-specialist-training-spec/0.1",
        "architecture": "tiny-affine-tanh",
        "target_role": dataset.target_role,
        "distilled": distilled,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "deterministic": True,
        "resource_budget": {"max_epochs": MAX_EPOCHS, "max_rows": MAX_ROWS},
    }
    checkpoint = {
        "input_weights": input_weights,
        "hidden_bias": hidden_bias,
        "output_weights": output_weights,
        "output_bias": output_bias,
    }
    model = OneStepStudentModel(
        target_role=dataset.target_role,
        teacher_provider_identity=dataset.teacher_provider_identity,
        teacher_model_identity=dataset.teacher_model_identity,
        teacher_generation_identity=dataset.teacher_generation_identity,
        teacher_architecture_identity=dataset.teacher_architecture_identity,
        teacher_operating_envelope_identity=dataset.teacher_operating_envelope_identity,
        architecture_identity=canonical_digest(
            {
                "kind": "tiny-affine-tanh",
                "input": DIMENSIONS,
                "hidden": HIDDEN_WIDTH,
                "output": len(labels),
            }
        ),
        training_code_identity=canonical_digest(
            {"module": __name__, "algorithm": "bounded-sgd-tanh-softmax", "version": "0.1"}
        ),
        training_dataset_identity=dataset.dataset_identity or dataset.content_identity,
        distillation_identity=dataset.distillation_identity
        or dataset._distillation_content_identity,
        training_spec_identity=canonical_digest(spec),
        checkpoint_identity=canonical_digest(checkpoint),
        calibration_identity=canonical_digest(
            {"status": "pending", "dataset": dataset.dataset_identity}
        ),
        operating_envelope=StudentOperatingEnvelope(maximum_distance=max(600, radius + 240)),
        class_labels=labels,
        input_weights=tuple(tuple(row) for row in input_weights),
        hidden_bias=tuple(hidden_bias),
        output_weights=tuple(tuple(row) for row in output_weights),
        output_bias=tuple(output_bias),
        feature_center=center,
        training_record_ids=tuple(sorted(dataset.training_observation_identities)),
        source_observation_identities=tuple(
            sorted(
                {
                    identity
                    for item in dataset.observations
                    for identity in item.source_observation_identities
                }
            )
        ),
    )
    return replace(model, model_identity=model.content_identity)


def calibrate_one_step_student(
    model: OneStepStudentModel,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[OneStepStudentModel, StudentCalibration]:
    if not rows:
        raise OneStepSpecialistError("student calibration dataset is empty")
    confidences: list[int] = []
    distances: list[int] = []
    dataset_identity = canonical_digest({"rows": [dict(row) for row in rows]})
    for row in rows:
        features = _features(row.get("features"), "calibration features")
        prepared = model._prepare(features, _context_for_student(model))
        _, _, probabilities = model._raw_score(prepared)
        expected = row.get("expected")
        if isinstance(expected, str) and expected in model.class_labels:
            confidences.append(round(probabilities[model.class_labels.index(expected)] * 1000))
        distances.append(
            sum(abs(prepared[index] - model.feature_center[index]) for index in range(DIMENSIONS))
        )
    minimum_confidence = max(550, min(900, (min(confidences) - 40) if confidences else 700))
    known_distances = [
        distance for row, distance in zip(rows, distances) if row.get("expected") != "ABSTAIN"
    ]
    maximum_distance = max(
        1, max(known_distances, default=model.operating_envelope.maximum_distance) + 160
    )
    calibration = StudentCalibration(
        target_role=model.target_role,
        calibration_dataset_identity=dataset_identity,
        minimum_confidence=minimum_confidence,
        maximum_distance=maximum_distance,
    )
    calibration = replace(calibration, calibration_identity=calibration.content_identity)
    envelope = StudentOperatingEnvelope(
        minimum_confidence=calibration.minimum_confidence,
        maximum_distance=calibration.maximum_distance,
        maximum_query_abs=model.operating_envelope.maximum_query_abs,
        maximum_context_observations=model.operating_envelope.maximum_context_observations,
    )
    envelope = replace(envelope, envelope_identity=envelope.content_identity)
    calibrated = replace(
        model,
        calibration_identity=calibration.calibration_identity,
        operating_envelope=envelope,
        model_identity="",
        artifact_identity="",
    )
    return replace(calibrated, model_identity=calibrated.content_identity), calibration


@dataclass(frozen=True, slots=True)
class FallbackResult:
    student_decision: OneStepStudentDecision
    teacher_decision: SpecialistDecision | None
    final_decision: str
    final_confidence: int
    fallback_invoked: bool
    fallback_error: str | None
    student_elapsed_ns: int
    teacher_elapsed_ns: int
    total_elapsed_ns: int
    authority: str = AUTHORITY
    result_identity: str = ""

    def __post_init__(self) -> None:
        if self.teacher_decision is None and self.fallback_invoked and self.fallback_error is None:
            raise OneStepSpecialistError("fallback invocation must retain its result or error")
        if self.authority != AUTHORITY:
            raise OneStepSpecialistError("fallback results are diagnostic-only")
        if self.result_identity and self.result_identity != self.content_identity:
            raise OneStepSpecialistError("fallback result identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-one-step-fallback-result/0.1",
            "student_decision": self.student_decision.to_dict(),
            "teacher_decision": self.teacher_decision.to_dict() if self.teacher_decision else None,
            "final_decision": self.final_decision,
            "final_confidence_milli": self.final_confidence,
            "fallback_invoked": self.fallback_invoked,
            "fallback_error": self.fallback_error,
            "student_elapsed_ns": self.student_elapsed_ns,
            "teacher_elapsed_ns": self.teacher_elapsed_ns,
            "total_elapsed_ns": self.total_elapsed_ns,
            "authority": self.authority,
            "semantics": "explicit-student-abstention-and-teacher-fallback; not-a-verdict",
        }
        if include_identity:
            value["result_identity"] = self.result_identity or self.content_identity
        return value


def infer_with_teacher_fallback(
    student: OneStepStudentModel,
    teacher: RecurrentSpecialistModel,
    query: Sequence[int],
    *,
    context: SpecialistContextState | None = None,
    request_identity: str | None = None,
    source_observation_identities: Sequence[str] = (),
) -> FallbackResult:
    """Invoke the teacher only after an explicit student abstention."""

    if (
        teacher.provider_id != student.teacher_provider_identity
        or (teacher.model_identity or teacher.content_identity) != student.teacher_model_identity
        or teacher.generation_identity != student.teacher_generation_identity
        or teacher.architecture_identity != student.teacher_architecture_identity
        or (
            teacher.operating_envelope.envelope_identity
            or teacher.operating_envelope.content_identity
        )
        != student.teacher_operating_envelope_identity
    ):
        raise OneStepSpecialistError("fallback teacher is not lineage-compatible with student")
    started = time.perf_counter_ns()
    student_decision = student.infer(
        query,
        context=context,
        request_identity=request_identity,
        source_observation_identities=source_observation_identities,
    )
    teacher_decision = None
    fallback_error = None
    teacher_elapsed = 0
    if student_decision.abstained:
        fallback_invoked = True
        teacher_started = time.perf_counter_ns()
        try:
            teacher_decision = teacher.infer(
                query,
                context=context,
                request_identity=student_decision.request_identity,
                source_observation_identities=source_observation_identities,
                lineage_identity=student.distillation_identity,
            )
        except (SpecialistError, ValueError) as error:
            fallback_error = str(error)
        teacher_elapsed = time.perf_counter_ns() - teacher_started
    else:
        fallback_invoked = False
    final_decision = teacher_decision.decision if teacher_decision else student_decision.decision
    final_confidence = (
        teacher_decision.confidence if teacher_decision else student_decision.confidence
    )
    result = FallbackResult(
        student_decision=student_decision,
        teacher_decision=teacher_decision,
        final_decision=final_decision,
        final_confidence=final_confidence,
        fallback_invoked=fallback_invoked,
        fallback_error=fallback_error,
        student_elapsed_ns=student_decision.elapsed_ns,
        teacher_elapsed_ns=teacher_elapsed,
        total_elapsed_ns=time.perf_counter_ns() - started,
    )
    return replace(result, result_identity=result.content_identity)


def _reference_teacher() -> RecurrentSpecialistModel:
    from .recurrent_specialist import calibrate_recurrent_specialist, train_recurrent_specialist

    rows = (
        {
            "record_id": "forge-train-relevant-1",
            "features": [900, 820, 760, 880],
            "label": "relevant",
        },
        {
            "record_id": "forge-train-relevant-2",
            "features": [820, 760, 700, 800],
            "label": "relevant",
        },
        {
            "record_id": "forge-train-irrelevant-1",
            "features": [120, 180, 160, 100],
            "label": "irrelevant",
        },
        {
            "record_id": "forge-train-irrelevant-2",
            "features": [220, 120, 180, 160],
            "label": "irrelevant",
        },
    )
    teacher = train_recurrent_specialist(
        rows,
        target_role="forge.evidence-relevance",
        generation_identity=canonical_digest(
            {"role": "forge.evidence-relevance", "generation": "G0"}
        ),
        negative_memory=("known-omission-is-escalation",),
    )
    teacher, _ = calibrate_recurrent_specialist(
        teacher,
        (
            *rows,
            {
                "record_id": "forge-calibration-boundary",
                "features": [760, 700, 660, 720],
                "label": "relevant",
            },
        ),
    )
    return teacher


def _study_training_rows() -> tuple[dict[str, Any], ...]:
    return (
        {
            "record_id": "study-train-relevant-1",
            "features": [900, 820, 760, 880],
            "expected": "relevant",
        },
        {
            "record_id": "study-train-relevant-2",
            "features": [820, 760, 700, 800],
            "expected": "relevant",
        },
        {
            "record_id": "study-train-relevant-3",
            "features": [760, 700, 660, 720],
            "expected": "relevant",
        },
        {
            "record_id": "study-train-relevant-4",
            "features": [700, 740, 720, 760],
            "expected": "relevant",
        },
        {
            "record_id": "study-train-irrelevant-1",
            "features": [120, 180, 160, 100],
            "expected": "irrelevant",
        },
        {
            "record_id": "study-train-irrelevant-2",
            "features": [220, 120, 180, 160],
            "expected": "irrelevant",
        },
        {
            "record_id": "study-train-irrelevant-3",
            "features": [160, 220, 120, 180],
            "expected": "irrelevant",
        },
        {
            "record_id": "study-train-irrelevant-4",
            "features": [260, 180, 220, 200],
            "expected": "irrelevant",
        },
        # Deliberately mis-targeted and retained as rejected teacher evidence.
        {
            "record_id": "study-rejected-target",
            "features": [900, 820, 760, 880],
            "expected": "irrelevant",
        },
        {"record_id": "study-unknown-target", "features": [640, 620, 600, 580]},
        {
            "record_id": "study-teacher-abstention",
            "features": [500, 500, 500, 500],
            "expected": "ABSTAIN",
        },
    )


def _study_holdout_rows() -> tuple[dict[str, Any], ...]:
    return (
        {
            "record_id": "study-heldout-relevant",
            "features": [780, 740, 700, 760],
            "expected": "relevant",
        },
        {
            "record_id": "study-heldout-relevant-boundary",
            "features": [720, 700, 680, 720],
            "expected": "relevant",
        },
        {
            "record_id": "study-heldout-irrelevant",
            "features": [160, 220, 120, 180],
            "expected": "irrelevant",
        },
        {
            "record_id": "study-heldout-irrelevant-boundary",
            "features": [260, 180, 220, 200],
            "expected": "irrelevant",
        },
        {
            "record_id": "study-ood-alternating",
            "features": [1000, -1000, 1000, -1000],
            "expected": "ABSTAIN",
        },
        {
            "record_id": "study-ood-high",
            "features": [1000, 1000, 1000, 1000],
            "expected": "ABSTAIN",
        },
    )


def _evaluate_reference(
    teacher: RecurrentSpecialistModel,
    student: OneStepStudentModel,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    teacher_results: list[SpecialistDecision] = []
    student_results: list[OneStepStudentDecision] = []
    fallback_results: list[FallbackResult] = []
    for row in rows:
        features = _features(row.get("features"), "study features")
        source = _source_identity(row)
        teacher_results.append(teacher.infer(features, source_observation_identities=(source,)))
        student_results.append(student.infer(features, source_observation_identities=(source,)))
        fallback_results.append(
            infer_with_teacher_fallback(
                student, teacher, features, source_observation_identities=(source,)
            )
        )
    expected = [str(row.get("expected", "UNKNOWN")) for row in rows]
    known = [index for index, item in enumerate(expected) if item not in {"UNKNOWN", "ABSTAIN"}]
    ood = [index for index, item in enumerate(expected) if item == "ABSTAIN"]
    teacher_decisions = [item.decision for item in teacher_results]
    student_decisions = [item.decision for item in student_results]
    baseline_results = []
    for row in rows:
        started = time.perf_counter_ns()
        features = _features(row.get("features"), "study features")
        distances = sorted(
            (teacher._distance(features, centroid), label)
            for label, centroid in teacher.class_centroids.items()
        )
        best_distance, best_label = distances[0]
        second_distance = distances[1][0] if len(distances) > 1 else None
        confidence = teacher._confidence(best_distance, second_distance)
        abstained = (
            best_distance > teacher.operating_envelope.maximum_distance
            or confidence < teacher.operating_envelope.minimum_confidence
        )
        baseline_results.append(
            {
                "decision": "ABSTAIN" if abstained else best_label,
                "abstained": abstained,
                "out_of_distribution": best_distance > teacher.operating_envelope.maximum_distance,
                "operations": 1 + len(teacher.class_centroids) * DIMENSIONS,
                "elapsed_ns": time.perf_counter_ns() - started,
            }
        )
    baseline_decisions = [item["decision"] for item in baseline_results]
    return {
        "total_reference_cases": len(rows),
        "known_cases": len(known),
        "expected": expected,
        "teacher": {
            "decisions": teacher_decisions,
            "correct_known_cases": sum(
                teacher_decisions[index] == expected[index] for index in known
            ),
            "false_accepts": sum(teacher_decisions[index] != "ABSTAIN" for index in ood),
            "abstentions": sum(item.abstained for item in teacher_results),
            "reasoning_iterations": [item.reasoning_iterations for item in teacher_results],
            "operations": [item.operations for item in teacher_results],
            "latency_ns": [item.elapsed_ns for item in teacher_results],
            "model_bytes": teacher.model_size_bytes,
        },
        "student": {
            "decisions": student_decisions,
            "correct_known_cases": sum(
                student_decisions[index] == expected[index] for index in known
            ),
            "false_accepts": sum(student_decisions[index] != "ABSTAIN" for index in ood),
            "abstentions": sum(item.abstained for item in student_results),
            "out_of_distribution_cases": sum(item.out_of_distribution for item in student_results),
            "teacher_disagreements": sum(
                student_decisions[index] != teacher_decisions[index] for index in range(len(rows))
            ),
            "learned_forward_passes": [item.learned_forward_passes for item in student_results],
            "inference_operations": [item.inference_operations for item in student_results],
            "preprocessing_operations": [item.preprocessing_operations for item in student_results],
            "cold_latency_ns": student_results[0].elapsed_ns,
            "warm_latency_ns": [item.elapsed_ns for item in student_results[1:]],
            "model_bytes": student.model_size_bytes,
            "calibration_threshold_milli": student.operating_envelope.minimum_confidence,
        },
        "baseline": {
            "kind": "nearest-centroid-deterministic-reference",
            "decisions": baseline_decisions,
            "correct_known_cases": sum(
                baseline_decisions[index] == expected[index] for index in known
            ),
            "false_accepts": sum(baseline_decisions[index] != "ABSTAIN" for index in ood),
            "abstentions": sum(item["abstained"] for item in baseline_results),
            "out_of_distribution_cases": sum(
                item["out_of_distribution"] for item in baseline_results
            ),
            "inference_operations": [item["operations"] for item in baseline_results],
            "latency_ns": [item["elapsed_ns"] for item in baseline_results],
            "model_bytes": len(canonical_json({"class_centroids": teacher.class_centroids})),
        },
        "fallback": {
            "fallback_invocations": sum(item.fallback_invoked for item in fallback_results),
            "final_decisions": [item.final_decision for item in fallback_results],
            "student_abstentions": sum(
                item.student_decision.abstained for item in fallback_results
            ),
            "teacher_results_retained": sum(
                item.teacher_decision is not None for item in fallback_results
            ),
            "student_only_end_to_end_latency_ns": [
                item.student_elapsed_ns for item in fallback_results
            ],
            "escalated_end_to_end_latency_ns": [
                item.total_elapsed_ns for item in fallback_results if item.fallback_invoked
            ],
        },
        "calibration_behavior": {
            "threshold_milli": student.operating_envelope.minimum_confidence,
            "ood_cases": len(ood),
            "ood_abstentions": sum(student_results[index].abstained for index in ood),
            "known_coverage": sum(not student_results[index].abstained for index in known)
            / max(1, len(known)),
        },
        "decision_digest": canonical_digest(
            {"teacher": teacher_decisions, "student": student_decisions}
        ),
    }


def build_one_step_reference_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Generate the deterministic forge reference study and its lineage artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    teacher = _reference_teacher()
    observations = tuple(
        collect_teacher_observation(teacher, row) for row in _study_training_rows()
    )
    dataset = build_distillation_dataset(observations, target_role=teacher.target_role)
    student, calibration = calibrate_one_step_student(
        train_one_step_student(dataset, distilled=True),
        _study_holdout_rows()[:4],
    )
    control, control_calibration = calibrate_one_step_student(
        train_one_step_student(dataset, distilled=False),
        _study_holdout_rows()[:4],
    )
    student_path = destination / "forge-student-distilled-g0.json"
    control_path = destination / "forge-student-no-distillation-control-g0.json"
    teacher_path = destination / "forge-teacher-recurrent-g0.json"
    records_path = destination / "distillation-records.json"
    teacher_path.write_bytes(teacher.serialize())
    student_path.write_bytes(student.serialize())
    control_path.write_bytes(control.serialize())
    records_path.write_text(
        json.dumps(
            {
                "schema": "mnel-distillation-record-set/0.1",
                "records": [item.to_dict() for item in observations],
                "dataset": dataset.to_dict(),
                "authority": AUTHORITY,
                "semantics": "retained-teacher-observations-and-target-construction; not-a-verdict",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    distilled_evaluation = _evaluate_reference(teacher, student, _study_holdout_rows())
    control_evaluation = _evaluate_reference(teacher, control, _study_holdout_rows())
    evaluations = {
        "iterative_teacher": distilled_evaluation["teacher"],
        "one_step_distilled_student": distilled_evaluation["student"],
        "classical_nearest_centroid_baseline": distilled_evaluation["baseline"],
        "no_distillation_control": control_evaluation["student"],
        "fallback": distilled_evaluation["fallback"],
        "calibration_behavior": distilled_evaluation["calibration_behavior"],
        "comparisons": {
            "student_teacher_disagreements": distilled_evaluation["student"][
                "teacher_disagreements"
            ],
            "student_control_disagreements": sum(
                left != right
                for left, right in zip(
                    distilled_evaluation["student"]["decisions"],
                    control_evaluation["student"]["decisions"],
                )
            ),
        },
    }
    study_body = {
        "teacher_model_identity": teacher.model_identity,
        "teacher_artifact_identity": canonical_digest(json.loads(teacher.serialize())),
        "distillation_dataset_identity": dataset.dataset_identity,
        "distillation_identity": dataset.distillation_identity,
        "student_model_identity": student.model_identity,
        "student_artifact_identity": canonical_digest(json.loads(student.serialize())),
        "control_model_identity": control.model_identity,
        "control_artifact_identity": canonical_digest(json.loads(control.serialize())),
        "calibration_identity": calibration.calibration_identity,
        "control_calibration_identity": control_calibration.calibration_identity,
        "evaluations": _without_timing(evaluations),
    }
    report = {
        "schema": "mnel-one-step-specialist-reference-report/0.1",
        "study_identity": canonical_digest(study_body),
        "target_role": teacher.target_role,
        "total_reference_cases": distilled_evaluation["total_reference_cases"],
        "known_cases": distilled_evaluation["known_cases"],
        "heldout_examples": len(_study_holdout_rows()),
        "training_examples": len(dataset.training_observations),
        "teacher": {
            "provider_id": teacher.provider_id,
            "model_identity": teacher.model_identity,
            "artifact_identity": canonical_digest(json.loads(teacher.serialize())),
            "architecture_identity": teacher.architecture_identity,
            "operating_envelope_identity": teacher.operating_envelope.envelope_identity,
            "model_bytes": teacher.model_size_bytes,
            "artifact_bytes": len(teacher.serialize()),
        },
        "distillation": {
            "dataset_identity": dataset.dataset_identity,
            "distillation_identity": dataset.distillation_identity,
            "transform_identity": dataset.transform_identity,
            "retained_observation_count": len(dataset.observations),
            "training_observation_count": len(dataset.training_observations),
            "rejected_or_unresolved_observation_count": len(dataset.observations)
            - len(dataset.training_observations),
            "rejected_or_unresolved_statuses": [
                item.target_status.value
                for item in dataset.observations
                if item.target_status != TargetStatus.VERIFIED
            ],
        },
        "students": {
            "distilled": {
                "model_identity": student.model_identity,
                "artifact_identity": canonical_digest(json.loads(student.serialize())),
                "architecture_identity": student.architecture_identity,
                "training_dataset_identity": student.training_dataset_identity,
                "calibration_identity": student.calibration_identity,
                "model_bytes": student.model_size_bytes,
                "artifact_bytes": len(student.serialize()),
            },
            "no_distillation_control": {
                "model_identity": control.model_identity,
                "artifact_identity": canonical_digest(json.loads(control.serialize())),
                "architecture_identity": control.architecture_identity,
                "training_dataset_identity": control.training_dataset_identity,
                "calibration_identity": control.calibration_identity,
                "model_bytes": control.model_size_bytes,
                "artifact_bytes": len(control.serialize()),
            },
        },
        "evaluations": evaluations,
        "claim_boundary": {
            "one_step_means": "one learned affine/tanh/softmax forward evaluation after preparation",
            "student_authority": AUTHORITY,
            "teacher_fallback_is_explicit": True,
            "success_claim_supported": False,
        },
        "limitations": [
            "synthetic forge relevance features do not establish production utility",
            "latency measurements are host-dependent and excluded from semantic identities",
            "the student is diagnostic-only and cannot verify evidence, grant permissions, or promote generations",
            "the recurrent teacher remains available as an explicit fallback rather than being silently replaced",
        ],
        "authority": AUTHORITY,
        "semantics": "experimental-reference-measurement; not-a-verdict",
    }
    report_path = destination / "reference-study.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "artifacts": {
            "teacher": str(teacher_path),
            "student": str(student_path),
            "control": str(control_path),
            "records": str(records_path),
        },
        "evidence": str(report_path),
        "report": report,
    }


__all__ = [
    "ARTIFACT_SCHEMA",
    "DECISION_SCHEMA",
    "DISTILLATION_DATASET_SCHEMA",
    "DISTILLATION_RECORD_SCHEMA",
    "DistillationDataset",
    "FallbackResult",
    "OneStepSpecialistError",
    "OneStepStudentDecision",
    "OneStepStudentModel",
    "StudentCalibration",
    "StudentOperatingEnvelope",
    "TargetStatus",
    "TeacherObservation",
    "build_distillation_dataset",
    "build_one_step_reference_artifacts",
    "calibrate_one_step_student",
    "collect_teacher_observation",
    "infer_with_teacher_fallback",
    "train_one_step_student",
]
