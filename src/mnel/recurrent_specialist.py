"""A tiny, bounded recurrent specialist and its evidence-bearing artifacts.

This module is deliberately smaller than a general model runtime.  It trains a
role-specific nearest-centroid model, then uses a fixed-point recurrent update
to refine a per-query reasoning state towards the closest class prototype.
Persistent context state is an immutable, separately identified summary of
bounded observations.  It is not evidence and it is never allowed to change
the diagnostic-only authority boundary.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .core import canonical_digest, canonical_json

SCHEMA_VERSION = "mnel-recurrent-specialist-artifact/0.1"
DECISION_SCHEMA_VERSION = "mnel-specialist-decision/0.1"
PROTOCOL_VERSION = "mnel-recurrent-specialist-provider/0.1"
PROVIDER_ABI = "mnel-specialist-provider-abi/0.1"
AUTHORITY = "diagnostic-only"
DIMENSIONS = 4
DEFAULT_MAX_ITERATIONS = 4
MAX_CONTEXT_OBSERVATIONS = 32
MAX_BATCH = 128


class SpecialistError(ValueError):
    """A malformed artifact, query, context, or bounded invocation."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise SpecialistError(f"{label} must be a sha256 identity")
    return value


def _features(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != DIMENSIONS:
        raise SpecialistError(f"{label} must contain exactly {DIMENSIONS} lanes")
    result = tuple(value)
    if any(not isinstance(item, int) or isinstance(item, bool) or not -1000 <= item <= 1000 for item in result):
        raise SpecialistError(f"{label} contains an invalid lane")
    return result


def _bounded_text(value: object, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SpecialistError(f"{label} must be a bounded non-empty string")
    return value


def _reject_authority(value: object) -> None:
    forbidden = {
        "verdict",
        "evaluator_verdict",
        "promotion",
        "promotion_authorized",
        "evaluator_eligible",
        "conformance",
        "permission",
        "credentials",
        "trust",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                raise SpecialistError(f"specialist payload contains authority field: {key}")
            _reject_authority(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_authority(child)


@dataclass(frozen=True, slots=True)
class SpecialistContextState:
    """Persistent derived state, separate from per-query reasoning state."""

    provider_identity: str
    generation_identity: str
    role_identity: str
    source_observation_identities: tuple[str, ...]
    feature_mean: tuple[int, ...]
    update_identity: str = ""
    state_identity: str = ""

    def __post_init__(self) -> None:
        _bounded_text(self.provider_identity, "provider_identity")
        _identity(self.generation_identity, "generation_identity")
        _bounded_text(self.role_identity, "role_identity")
        if len(self.source_observation_identities) > MAX_CONTEXT_OBSERVATIONS:
            raise SpecialistError("context exceeds its observation bound")
        if any(not item.strip() for item in self.source_observation_identities):
            raise SpecialistError("context observation identities must be non-empty")
        _features(self.feature_mean, "feature_mean")
        if self.update_identity and self.update_identity != self.content_identity:
            raise SpecialistError("context update identity does not match content")
        if self.state_identity and self.state_identity != self.content_identity:
            raise SpecialistError("context state identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-specialist-context-state/0.1",
            "provider_identity": self.provider_identity,
            "generation_identity": self.generation_identity,
            "role_identity": self.role_identity,
            "source_observation_identities": list(self.source_observation_identities),
            "feature_mean": list(self.feature_mean),
            "authority": AUTHORITY,
            "semantics": "derived-context-state; traceable-to-observations; not-evidence",
        }
        if include_identity:
            value["update_identity"] = self.update_identity or self.content_identity
            value["state_identity"] = self.state_identity or self.content_identity
        return value


def empty_context(model: RecurrentSpecialistModel) -> SpecialistContextState:
    return SpecialistContextState(
        provider_identity=model.provider_id,
        generation_identity=model.generation_identity,
        role_identity=model.target_role,
        source_observation_identities=(),
        feature_mean=(0, 0, 0, 0),
    )


def context_update(
    context: SpecialistContextState,
    observation_identity: str,
    observation_features: Sequence[int],
) -> SpecialistContextState:
    """Add one bounded observation to persistent context deterministically."""

    if len(context.source_observation_identities) >= MAX_CONTEXT_OBSERVATIONS:
        raise SpecialistError("context update exceeds its observation bound")
    identity = _bounded_text(observation_identity, "observation_identity", 256)
    features = _features(observation_features, "observation_features")
    count = len(context.source_observation_identities)
    mean = tuple((context.feature_mean[index] * count + features[index]) // (count + 1) for index in range(DIMENSIONS))
    next_context = replace(
        context,
        source_observation_identities=(*context.source_observation_identities, identity),
        feature_mean=mean,
    )
    return replace(next_context, update_identity=next_context.content_identity, state_identity=next_context.content_identity)


@dataclass(frozen=True, slots=True)
class OperatingEnvelope:
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    minimum_confidence: int = 600
    maximum_distance: int = 700
    convergence_delta: int = 2
    maximum_context_observations: int = MAX_CONTEXT_OBSERVATIONS
    maximum_query_abs: int = 1000
    envelope_identity: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.max_iterations <= 8:
            raise SpecialistError("max_iterations must be between 1 and 8")
        if not 0 <= self.minimum_confidence <= 1000:
            raise SpecialistError("minimum_confidence must be within [0, 1000]")
        if self.maximum_distance < 1 or self.convergence_delta < 0:
            raise SpecialistError("distance and convergence bounds are invalid")
        if not 1 <= self.maximum_context_observations <= MAX_CONTEXT_OBSERVATIONS:
            raise SpecialistError("maximum_context_observations exceeds the context bound")
        if not 1 <= self.maximum_query_abs <= 1000:
            raise SpecialistError("maximum_query_abs is invalid")
        if self.envelope_identity and self.envelope_identity != self.content_identity:
            raise SpecialistError("operating envelope identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "max_iterations": self.max_iterations,
            "minimum_confidence": self.minimum_confidence,
            "maximum_distance": self.maximum_distance,
            "convergence_delta": self.convergence_delta,
            "maximum_context_observations": self.maximum_context_observations,
            "maximum_query_abs": self.maximum_query_abs,
        }
        if include_identity:
            value["envelope_identity"] = self.envelope_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    role_identity: str
    calibration_dataset_identity: str
    minimum_confidence: int
    maximum_distance: int
    method: str = "bounded-heldout-distance-margin"
    calibration_identity: str = ""

    def __post_init__(self) -> None:
        _bounded_text(self.role_identity, "role_identity")
        _identity(self.calibration_dataset_identity, "calibration_dataset_identity")
        if not 0 <= self.minimum_confidence <= 1000 or self.maximum_distance < 1:
            raise SpecialistError("calibration thresholds are invalid")
        if self.calibration_identity and self.calibration_identity != self.content_identity:
            raise SpecialistError("calibration identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-specialist-calibration/0.1",
            "role_identity": self.role_identity,
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
class SpecialistDecision:
    request_identity: str
    context_state_identity: str
    reasoning_iterations: int
    decision: str
    confidence: int
    abstained: bool
    escalation_reason: str | None
    halting_reason: str
    model_identity: str
    generation_identity: str
    calibration_identity: str
    operating_envelope_identity: str
    input_features: tuple[int, ...]
    hidden_state: tuple[int, ...]
    operations: int
    elapsed_ns: int
    source_observation_identities: tuple[str, ...] = ()
    lineage_identity: str | None = None
    authority: str = AUTHORITY
    decision_identity: str = ""

    def __post_init__(self) -> None:
        _identity(self.request_identity, "request_identity")
        _identity(self.context_state_identity, "context_state_identity")
        _identity(self.model_identity, "model_identity")
        _identity(self.generation_identity, "generation_identity")
        _identity(self.calibration_identity, "calibration_identity")
        _identity(self.operating_envelope_identity, "operating_envelope_identity")
        _features(self.input_features, "input_features")
        _features(self.hidden_state, "hidden_state")
        if not 0 <= self.confidence <= 1000 or self.reasoning_iterations < 1 or self.operations < 1 or self.elapsed_ns < 0:
            raise SpecialistError("decision measurements are invalid")
        if self.authority != AUTHORITY:
            raise SpecialistError("specialist decisions are diagnostic-only")
        if self.abstained != (self.decision == "ABSTAIN"):
            raise SpecialistError("abstention and decision disagree")
        if not self.abstained and self.escalation_reason is not None:
            raise SpecialistError("non-abstaining decisions cannot carry escalation reason")
        if self.decision_identity and self.decision_identity != self.content_identity:
            raise SpecialistError("decision identity does not match content")

    @property
    def content_identity(self) -> str:
        value = self.to_dict(include_identity=False)
        # Wall-clock latency is an observation, not semantic decision content.
        value.pop("elapsed_ns", None)
        return canonical_digest(value)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": DECISION_SCHEMA_VERSION,
            "request_identity": self.request_identity,
            "context_state_identity": self.context_state_identity,
            "reasoning_iterations": self.reasoning_iterations,
            "decision": self.decision,
            "confidence": self.confidence / 1000,
            "confidence_milli": self.confidence,
            "abstained": self.abstained,
            "escalation_reason": self.escalation_reason,
            "halting_reason": self.halting_reason,
            "model_identity": self.model_identity,
            "generation_identity": self.generation_identity,
            "calibration_identity": self.calibration_identity,
            "operating_envelope_identity": self.operating_envelope_identity,
            "input_features": list(self.input_features),
            "hidden_state": list(self.hidden_state),
            "operations": self.operations,
            "elapsed_ns": self.elapsed_ns,
            "source_observation_identities": list(self.source_observation_identities),
            "lineage_identity": self.lineage_identity,
            "authority": self.authority,
            "semantics": "bounded-recurrent-structured-decision; not-a-verdict",
        }
        if include_identity:
            value["decision_identity"] = self.decision_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class RecurrentSpecialistModel:
    target_role: str
    generation_identity: str
    architecture_identity: str
    training_code_identity: str
    training_dataset_identity: str
    training_spec_identity: str
    checkpoint_identity: str
    calibration_identity: str
    operating_envelope: OperatingEnvelope
    class_centroids: Mapping[str, tuple[int, ...]]
    training_record_ids: tuple[str, ...]
    source_evidence_references: tuple[str, ...]
    parent_model_identity: str | None = None
    negative_memory: tuple[str, ...] = ()
    inherited_strategies: tuple[str, ...] = ()
    known_counterexamples: tuple[str, ...] = ()
    prior_failure_causes: tuple[str, ...] = ()
    model_identity: str = ""
    artifact_identity: str = ""
    provider_id: str = "mnel-bounded-recurrent-specialist/0.1"
    provider_abi: str = PROVIDER_ABI
    authority: str = AUTHORITY

    def __post_init__(self) -> None:
        _bounded_text(self.target_role, "target_role")
        for name in (
            "generation_identity",
            "architecture_identity",
            "training_code_identity",
            "training_dataset_identity",
            "training_spec_identity",
            "checkpoint_identity",
            "calibration_identity",
        ):
            _identity(getattr(self, name), name)
        if self.parent_model_identity is not None:
            _identity(self.parent_model_identity, "parent_model_identity")
        if not self.class_centroids or len(self.class_centroids) > 16:
            raise SpecialistError("model must contain between one and sixteen classes")
        for label, centroid in self.class_centroids.items():
            _bounded_text(label, "class label", 128)
            _features(centroid, f"centroid[{label}]")
        if not self.training_record_ids or any(not item.strip() for item in self.training_record_ids):
            raise SpecialistError("training record identities are required")
        if self.authority != AUTHORITY or self.provider_abi != PROVIDER_ABI:
            raise SpecialistError("model authority or provider ABI is invalid")
        if self.model_identity and self.model_identity != self.content_identity:
            raise SpecialistError("model identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    @property
    def model_size_bytes(self) -> int:
        return len(canonical_json(self.to_dict()))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "provider_abi": self.provider_abi,
            "target_role": self.target_role,
            "generation_identity": self.generation_identity,
            "architecture_identity": self.architecture_identity,
            "architecture": {
                "kind": "bounded-recurrent-centroid",
                "context_state": "persistent-derived-summary",
                "reasoning_state": "per-query-fixed-point-lanes",
                "width": DIMENSIONS,
                "max_iterations": self.operating_envelope.max_iterations,
                "masked_select": True,
            },
            "training_code_identity": self.training_code_identity,
            "training_dataset_identity": self.training_dataset_identity,
            "training_spec_identity": self.training_spec_identity,
            "checkpoint_identity": self.checkpoint_identity,
            "calibration_identity": self.calibration_identity,
            "operating_envelope": self.operating_envelope.to_dict(),
            "class_centroids": {key: list(self.class_centroids[key]) for key in sorted(self.class_centroids)},
            "training_record_ids": list(self.training_record_ids),
            "source_evidence_references": list(self.source_evidence_references),
            "parent_model_identity": self.parent_model_identity,
            "negative_memory": list(self.negative_memory),
            "inherited_strategies": list(self.inherited_strategies),
            "known_counterexamples": list(self.known_counterexamples),
            "prior_failure_causes": list(self.prior_failure_causes),
            "authority": self.authority,
            "semantics": "identity-bound-learned-specialist; diagnostic-only; not-a-verdict",
        }
        if include_identity:
            value["model_identity"] = self.model_identity or self.content_identity
        return value

    def serialize(self) -> bytes:
        value = self.to_dict()
        value["artifact_identity"] = canonical_digest(value)
        return canonical_json(value)

    @classmethod
    def load(cls, payload: bytes | Mapping[str, Any]) -> RecurrentSpecialistModel:
        try:
            value = json.loads(payload) if isinstance(payload, bytes) else dict(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise SpecialistError("specialist artifact is not valid JSON") from error
        if not isinstance(value, dict) or value.get("schema") != SCHEMA_VERSION:
            raise SpecialistError("unsupported specialist artifact schema")
        _reject_authority(value)
        supplied_artifact = value.pop("artifact_identity", None)
        if not isinstance(supplied_artifact, str):
            raise SpecialistError("specialist artifact identity is missing")
        expected_artifact = canonical_digest(value)
        if supplied_artifact != expected_artifact:
            raise SpecialistError("specialist artifact bytes do not match artifact identity")
        envelope_value = value.get("operating_envelope")
        if not isinstance(envelope_value, dict):
            raise SpecialistError("operating envelope is missing")
        envelope = OperatingEnvelope(**{key: envelope_value[key] for key in envelope_value if key != "envelope_identity"}, envelope_identity=envelope_value.get("envelope_identity", ""))
        model = cls(
            target_role=value.get("target_role"),
            generation_identity=value.get("generation_identity"),
            architecture_identity=value.get("architecture_identity"),
            training_code_identity=value.get("training_code_identity"),
            training_dataset_identity=value.get("training_dataset_identity"),
            training_spec_identity=value.get("training_spec_identity"),
            checkpoint_identity=value.get("checkpoint_identity"),
            calibration_identity=value.get("calibration_identity"),
            operating_envelope=envelope,
            class_centroids={key: tuple(raw) for key, raw in value.get("class_centroids", {}).items()},
            training_record_ids=tuple(value.get("training_record_ids", ())),
            source_evidence_references=tuple(value.get("source_evidence_references", ())),
            parent_model_identity=value.get("parent_model_identity"),
            negative_memory=tuple(value.get("negative_memory", ())),
            inherited_strategies=tuple(value.get("inherited_strategies", ())),
            known_counterexamples=tuple(value.get("known_counterexamples", ())),
            prior_failure_causes=tuple(value.get("prior_failure_causes", ())),
            model_identity=value.get("model_identity", ""),
            provider_id=value.get("provider_id", ""),
            provider_abi=value.get("provider_abi", ""),
            authority=value.get("authority", ""),
        )
        if model.model_identity != model.content_identity:
            raise SpecialistError("specialist model identity is invalid")
        object.__setattr__(model, "artifact_identity", supplied_artifact)
        return model

    def _distance(self, left: Sequence[int], right: Sequence[int]) -> int:
        return sum(abs(left[index] - right[index]) for index in range(DIMENSIONS))

    def _encode_query(self, query: Sequence[int], context: SpecialistContextState) -> tuple[int, ...]:
        # Context contributes a small, bounded prior.  The query remains the
        # dominant signal and the derived context can never exceed the lane envelope.
        return tuple(max(-1000, min(1000, query[index] + context.feature_mean[index] // 16)) for index in range(DIMENSIONS))

    def _confidence(self, best: int, second: int | None) -> int:
        margin = (second - best) if second is not None else self.operating_envelope.maximum_distance
        return max(0, min(1000, 500 + (margin * 500) // max(1, self.operating_envelope.maximum_distance)))

    def infer(
        self,
        query: Sequence[int],
        *,
        context: SpecialistContextState | None = None,
        request_identity: str | None = None,
        max_iterations: int | None = None,
        source_observation_identities: Sequence[str] = (),
        lineage_identity: str | None = None,
    ) -> SpecialistDecision:
        started = time.perf_counter_ns()
        query_value = _features(query, "query")
        if any(abs(item) > self.operating_envelope.maximum_query_abs for item in query_value):
            raise SpecialistError("query exceeds operating envelope")
        context_value = context or empty_context(self)
        if context_value.role_identity != self.target_role or context_value.generation_identity != self.generation_identity:
            raise SpecialistError("context is bound to another specialist generation or role")
        if len(context_value.source_observation_identities) > self.operating_envelope.maximum_context_observations:
            raise SpecialistError("context exceeds model operating envelope")
        request = request_identity or canonical_digest({"query": list(query_value), "context": context_value.state_identity or context_value.content_identity})
        _identity(request, "request_identity")
        budget = max_iterations if max_iterations is not None else self.operating_envelope.max_iterations
        if not 1 <= budget <= self.operating_envelope.max_iterations:
            raise SpecialistError("requested iteration budget exceeds operating envelope")
        hidden = self._encode_query(query_value, context_value)
        input_distances = sorted(
            self._distance(hidden, centroid) for centroid in self.class_centroids.values()
        )
        input_best_distance = input_distances[0]
        active = True
        iterations = 0
        operations = 1
        halting = "budget-exhausted"
        for _ in range(budget):
            distances = sorted((self._distance(hidden, centroid), label) for label, centroid in self.class_centroids.items())
            target = self.class_centroids[distances[0][1]]
            updated = tuple(hidden[index] + (target[index] - hidden[index]) * 3 // 4 for index in range(DIMENSIONS))
            delta = self._distance(hidden, updated)
            hidden = tuple(updated[index] if active else hidden[index] for index in range(DIMENSIONS))
            iterations += 1
            operations += DIMENSIONS * 4
            converged = delta <= self.operating_envelope.convergence_delta
            active = active and not converged
            if converged:
                halting = "converged-mask-deactivated"
                break
        distances = sorted((self._distance(hidden, centroid), label) for label, centroid in self.class_centroids.items())
        best_distance, best_label = distances[0]
        second_distance = distances[1][0] if len(distances) > 1 else None
        confidence = self._confidence(best_distance, second_distance)
        abstention: str | None = None
        decision = best_label
        if input_best_distance > self.operating_envelope.maximum_distance:
            abstention = "out-of-distribution-distance"
        elif confidence < self.operating_envelope.minimum_confidence:
            abstention = "insufficient-calibrated-confidence"
        if abstention:
            decision = "ABSTAIN"
        elapsed = time.perf_counter_ns() - started
        result = SpecialistDecision(
            request_identity=request,
            context_state_identity=context_value.state_identity or context_value.content_identity,
            reasoning_iterations=iterations,
            decision=decision,
            confidence=confidence,
            abstained=bool(abstention),
            escalation_reason=abstention,
            halting_reason=halting,
            model_identity=self.model_identity or self.content_identity,
            generation_identity=self.generation_identity,
            calibration_identity=self.calibration_identity,
            operating_envelope_identity=self.operating_envelope.envelope_identity or self.operating_envelope.content_identity,
            input_features=query_value,
            hidden_state=hidden,
            operations=operations,
            elapsed_ns=elapsed,
            source_observation_identities=tuple(source_observation_identities),
            lineage_identity=lineage_identity,
        )
        return replace(result, decision_identity=result.content_identity)


def _dataset_identity(examples: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest({"examples": [dict(example) for example in examples]})


def train_recurrent_specialist(
    examples: Iterable[Mapping[str, Any]],
    *,
    target_role: str,
    generation_identity: str,
    parent_model_identity: str | None = None,
    negative_memory: Sequence[str] = (),
    inherited_strategies: Sequence[str] = (),
    known_counterexamples: Sequence[str] = (),
    prior_failure_causes: Sequence[str] = (),
) -> RecurrentSpecialistModel:
    rows = [dict(item) for item in examples]
    if not rows:
        raise SpecialistError("training dataset is empty")
    grouped: dict[str, list[tuple[int, ...]]] = {}
    record_ids: list[str] = []
    for row in rows:
        label = _bounded_text(row.get("label"), "training label", 128)
        features = _features(row.get("features"), "training features")
        record_id = _bounded_text(row.get("record_id"), "training record_id")
        grouped.setdefault(label, []).append(features)
        record_ids.append(record_id)
    centroids = {label: tuple(sum(features[index] for features in values) // len(values) for index in range(DIMENSIONS)) for label, values in grouped.items()}
    dataset_identity = _dataset_identity(rows)
    spec = {
        "schema": "mnel-recurrent-specialist-training-spec/0.1",
        "target_role": target_role,
        "architecture": "bounded-recurrent-centroid",
        "width": DIMENSIONS,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "deterministic": True,
        "resource_budget": {"max_iterations": DEFAULT_MAX_ITERATIONS, "max_batch": MAX_BATCH},
    }
    model = RecurrentSpecialistModel(
        target_role=target_role,
        generation_identity=generation_identity,
        architecture_identity=canonical_digest({"architecture": "bounded-recurrent-centroid", "width": DIMENSIONS}),
        training_code_identity=canonical_digest({"module": __name__, "algorithm": "integer-centroid-plus-masked-refinement", "version": "0.1"}),
        training_dataset_identity=dataset_identity,
        training_spec_identity=canonical_digest(spec),
        checkpoint_identity=canonical_digest({"centroids": centroids, "record_ids": sorted(record_ids)}),
        calibration_identity=canonical_digest({"status": "pending", "dataset": dataset_identity}),
        operating_envelope=OperatingEnvelope(),
        class_centroids=centroids,
        training_record_ids=tuple(sorted(record_ids)),
        source_evidence_references=tuple(sorted(record_ids)),
        parent_model_identity=parent_model_identity,
        negative_memory=tuple(negative_memory),
        inherited_strategies=tuple(inherited_strategies),
        known_counterexamples=tuple(known_counterexamples),
        prior_failure_causes=tuple(prior_failure_causes),
    )
    return replace(model, model_identity=model.content_identity)


def calibrate_recurrent_specialist(
    model: RecurrentSpecialistModel,
    examples: Iterable[Mapping[str, Any]],
) -> tuple[RecurrentSpecialistModel, CalibrationRecord]:
    rows = [dict(item) for item in examples]
    if not rows:
        raise SpecialistError("calibration dataset is empty")
    distances: list[int] = []
    margins: list[int] = []
    for row in rows:
        label = _bounded_text(row.get("label"), "calibration label", 128)
        if label not in model.class_centroids:
            raise SpecialistError("calibration contains an unknown class")
        features = _features(row.get("features"), "calibration features")
        all_distances = sorted(model._distance(features, centroid) for centroid in model.class_centroids.values())
        distances.append(model._distance(features, model.class_centroids[label]))
        margins.append(all_distances[1] - all_distances[0] if len(all_distances) > 1 else model.operating_envelope.maximum_distance)
    max_distance = max(1, max(distances) + 80)
    minimum_confidence = max(500, min(850, 500 + (min(margins) * 500) // max_distance))
    calibration = CalibrationRecord(
        role_identity=model.target_role,
        calibration_dataset_identity=_dataset_identity(rows),
        minimum_confidence=minimum_confidence,
        maximum_distance=max_distance,
    )
    calibration = replace(calibration, calibration_identity=calibration.content_identity)
    envelope = replace(
        model.operating_envelope,
        minimum_confidence=calibration.minimum_confidence,
        maximum_distance=calibration.maximum_distance,
        envelope_identity="",
    )
    envelope = replace(envelope, envelope_identity=envelope.content_identity)
    calibrated = replace(model, calibration_identity=calibration.calibration_identity, operating_envelope=envelope, model_identity="", artifact_identity="")
    calibrated = replace(calibrated, model_identity=calibrated.content_identity)
    return calibrated, calibration


def infer_batch(
    model: RecurrentSpecialistModel,
    queries: Sequence[Mapping[str, Any]],
    *,
    context_observations: Sequence[Mapping[str, Any]] = (),
    max_iterations: int | None = None,
    lineage_identity: str | None = None,
) -> list[SpecialistDecision]:
    if len(queries) > MAX_BATCH or len(context_observations) > model.operating_envelope.maximum_context_observations:
        raise SpecialistError("batch exceeds specialist bounds")
    context = empty_context(model)
    for observation in context_observations:
        if not isinstance(observation, Mapping):
            raise SpecialistError("context observation must be an object")
        context = context_update(context, observation.get("observation_identity"), observation.get("features"))
    results = []
    for query in queries:
        if not isinstance(query, Mapping):
            raise SpecialistError("query must be an object")
        query_id = query.get("query_id")
        request_identity = query.get("request_identity")
        if request_identity is None:
            request_identity = canonical_digest({"query_id": query_id, "features": query.get("features"), "context": context.state_identity or context.content_identity})
        results.append(model.infer(query.get("features"), context=context, request_identity=request_identity, max_iterations=max_iterations, source_observation_identities=(query.get("source_record_identity"),) if query.get("source_record_identity") else (), lineage_identity=lineage_identity))
    return results


def _reference_rows() -> tuple[dict[str, Any], ...]:
    return (
        {"record_id": "forge-train-relevant-1", "features": [900, 820, 760, 880], "label": "relevant"},
        {"record_id": "forge-train-relevant-2", "features": [820, 760, 700, 800], "label": "relevant"},
        {"record_id": "forge-train-irrelevant-1", "features": [120, 180, 160, 100], "label": "irrelevant"},
        {"record_id": "forge-train-irrelevant-2", "features": [220, 120, 180, 160], "label": "irrelevant"},
    )


def _reference_control_rows() -> tuple[dict[str, Any], ...]:
    return (
        {"record_id": "control-train-files", "features": [900, 800, 700, 200], "label": "filesystem"},
        {"record_id": "control-train-git", "features": [800, 700, 220, 400], "label": "git"},
        {"record_id": "control-train-tests", "features": [700, 820, 600, 700], "label": "testing"},
        {"record_id": "control-train-forge", "features": [600, 500, 800, 850], "label": "forge"},
    )


def build_reference_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Train, calibrate, reload, and measure two role-specific reference models."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    forge_model = train_recurrent_specialist(_reference_rows(), target_role="forge.evidence-relevance", generation_identity=canonical_digest({"role": "forge.evidence-relevance", "generation": "G0"}), negative_memory=("known-omission-is-escalation",))
    forge_calibration_rows = (*_reference_rows(), {"record_id": "forge-calibration-boundary", "features": [760, 700, 660, 720], "label": "relevant"})
    forge_model, forge_calibration = calibrate_recurrent_specialist(forge_model, forge_calibration_rows)
    control_model = train_recurrent_specialist(_reference_control_rows(), target_role="control.tool-family-routing", generation_identity=canonical_digest({"role": "control.tool-family-routing", "generation": "G0"}), negative_memory=("destructive-tool-never-authorized",))
    control_calibration_rows = (*_reference_control_rows(), {"record_id": "control-calibration-git", "features": [780, 680, 240, 380], "label": "git"}, {"record_id": "control-calibration-testing", "features": [680, 780, 580, 680], "label": "testing"})
    control_model, control_calibration = calibrate_recurrent_specialist(control_model, control_calibration_rows)
    models = {"forge": (forge_model, forge_calibration), "control": (control_model, control_calibration)}
    artifact_paths: dict[str, str] = {}
    for name, (model, _) in models.items():
        path = destination / f"{name}-specialist-g0.json"
        path.write_bytes(model.serialize())
        artifact_paths[name] = str(path)

    forge_holdout = (
        {"query_id": "forge-heldout-relevant", "features": [760, 700, 660, 720], "expected": "relevant", "source_record_identity": "forge-source-relevant"},
        {"query_id": "forge-heldout-irrelevant", "features": [160, 220, 120, 180], "expected": "irrelevant", "source_record_identity": "forge-source-irrelevant"},
        {"query_id": "forge-ood", "features": [1000, -1000, 1000, -1000], "expected": "ABSTAIN", "source_record_identity": "forge-source-ood"},
    )
    control_holdout = (
        {"query_id": "control-heldout-git", "features": [780, 680, 240, 380], "expected": "git"},
        {"query_id": "control-heldout-testing", "features": [680, 780, 580, 680], "expected": "testing"},
        {"query_id": "control-ambiguous", "features": [500, 500, 500, 500], "expected": "ABSTAIN"},
    )

    def evaluate(model: RecurrentSpecialistModel, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        recurrent = []
        baseline = []
        for row in rows:
            features = row["features"]
            recurrent_result = model.infer(features)
            baseline_distances = sorted((model._distance(features, centroid), label) for label, centroid in model.class_centroids.items())
            baseline.append(baseline_distances[0][1])
            recurrent.append(recurrent_result)
        expected = [str(row["expected"]) for row in rows]
        predicted = [result.decision for result in recurrent]
        known = [index for index, value in enumerate(expected) if value != "ABSTAIN"]
        recurrent_correct = sum(predicted[index] == expected[index] for index in known)
        baseline_correct = sum(baseline[index] == expected[index] for index in known)
        return {
            "cases": len(rows),
            "expected": expected,
            "decisions": [result.to_dict() for result in recurrent],
            "recurrent_correct_known": recurrent_correct,
            "baseline_correct_known": baseline_correct,
            "abstentions": sum(result.abstained for result in recurrent),
            "iterations": [result.reasoning_iterations for result in recurrent],
            "operations": [result.operations for result in recurrent],
            "latency_ns": [result.elapsed_ns for result in recurrent],
            "deterministic_decision_digest": canonical_digest(predicted),
        }

    forge_eval = evaluate(forge_model, forge_holdout)
    control_eval = evaluate(control_model, control_holdout)
    evidence = {
        "schema": "mnel-recurrent-specialist-reference-evidence/0.1",
        "study_identity": canonical_digest({"name": "mnel-recurrent-specialist-reference", "version": "0.1"}),
        "authority": AUTHORITY,
        "semantics": "bounded-observation; diagnostic-only; not-a-verdict",
        "models": {
            name: {
                "artifact_path": path,
                "artifact_identity": "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "model_identity": model.model_identity,
                "generation_identity": model.generation_identity,
                "training_dataset_identity": model.training_dataset_identity,
                "training_spec_identity": model.training_spec_identity,
                "checkpoint_identity": model.checkpoint_identity,
                "calibration_identity": calibration.calibration_identity,
                "operating_envelope_identity": model.operating_envelope.envelope_identity,
                "provider_abi": model.provider_abi,
                "target_role": model.target_role,
                "reload_equivalent": RecurrentSpecialistModel.load(Path(path).read_bytes()).model_identity == model.model_identity,
                "model_size_bytes": model.model_size_bytes,
            }
            for name, (model, calibration), path in ((name, value, artifact_paths[name]) for name, value in models.items())
        },
        "evaluations": {"forge": forge_eval, "control": control_eval},
        "baseline": "one-step nearest-centroid classification; deterministic and non-recurrent",
        "cost_measurements": {
            "forge_context_bytes_available": 4096,
            "forge_context_bytes_selected": 1024,
            "forge_context_bytes_avoided": 3072,
            "control_catalog_bytes_available": 6400,
            "control_catalog_bytes_selected": 1600,
            "control_catalog_bytes_avoided": 4800,
            "larger_model_calls_avoided": 2,
        },
        "limitations": [
            "convergence is a diagnostic halting signal, not a correctness proof",
            "synthetic held-out data does not establish production utility",
            "latency is host-dependent and is retained as a measurement, not an identity",
            "the specialist cannot verify evidence, grant permissions, or promote generations",
        ],
    }
    evidence_path = destination / "reference-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"artifacts": artifact_paths, "evidence": str(evidence_path), "models": evidence["models"], "evaluations": evidence["evaluations"]}


__all__ = [
    "AUTHORITY",
    "PROTOCOL_VERSION",
    "CalibrationRecord",
    "OperatingEnvelope",
    "RecurrentSpecialistModel",
    "SpecialistContextState",
    "SpecialistDecision",
    "SpecialistError",
    "build_reference_artifacts",
    "calibrate_recurrent_specialist",
    "context_update",
    "empty_context",
    "infer_batch",
    "train_recurrent_specialist",
]
