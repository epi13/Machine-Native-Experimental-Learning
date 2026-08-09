"""Evidence-preserving verified-experience distillation and study controls.

This module compresses retrieval and reuse metadata while leaving source evidence in the
append-only ledger. It intentionally contains no evaluator implementation and never
turns a retrieval score, calibration result, or transfer result into a verdict.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

from .core import (
    Attribution,
    EvidenceLedger,
    TransferStatus,
    Visibility,
    canonical_digest,
    canonical_json,
)


AUTHORITY_DIAGNOSTIC_ONLY = "diagnostic-only"
AUTHORITY_PROPOSAL_ONLY = "proposal-only"
SEMANTICS_NOT_A_VERDICT = "not-a-verdict"
MAX_RECORD_BYTES = 64 * 1024
MAX_GROUPS = 64
MAX_GROUP_RECORDS = 512
MAX_RETRIEVAL_HITS = 128


class DistillationError(ValueError):
    pass


class VisibilityViolation(DistillationError):
    pass


class StudyArmKind(StrEnum):
    RANDOM = "A0-random-bounded"
    AGGREGATE_ONLY = "A1-aggregate-only"
    COMPLETE_EPISODES = "A2-complete-episodes"
    COMPETING_PROBES = "A3-competing-hypotheses-probes"
    ATTRIBUTION = "A4-attribution-principles"
    TRANSFER_GATED = "A5-transfer-gated"
    POLICY_RECURSION = "A6-bounded-policy-recursion"


class AblationKind(StrEnum):
    AGGREGATE_ONLY = "aggregate-only"
    SHUFFLED_ATTRIBUTION = "shuffled-attribution"
    RANDOM_PROPOSAL = "random-proposal"
    SUCCESS_MEMORY = "success-memory-ablation"
    NEGATIVE_MEMORY = "negative-memory-ablation"
    FIXED_POLICY = "fixed-policy"
    EQUAL_BUDGET = "equal-budget"
    HIDDEN_TRANSFER = "hidden-transfer"


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DistillationError(f"{label} is required")
    return value


def _bounded(value: dict[str, Any], label: str, limit: int = MAX_RECORD_BYTES) -> dict[str, Any]:
    _reject_authority(value)
    try:
        encoded = canonical_json(value)
    except (TypeError, ValueError) as error:
        raise DistillationError(f"{label} is not canonical JSON") from error
    if len(encoded) > limit:
        raise DistillationError(f"{label} exceeds its byte ceiling")
    return dict(value)


def _reject_authority(value: Any) -> None:
    forbidden = {
        "promotion_authorized",
        "evaluator_eligible",
        "evaluator_verdict",
        "verdict",
        "mncs_verdict",
        "mncds_verdict",
        "pass_fail",
        "promotion",
        "conformance",
        "future_final",
        "hidden_transfer_result",
        "ravel_promotion",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                raise DistillationError(f"distillation record contains forbidden field: {key}")
            if key == "authority" and child not in {
                AUTHORITY_DIAGNOSTIC_ONLY,
                AUTHORITY_PROPOSAL_ONLY,
                "evidence-record",
            }:
                raise DistillationError("distillation record attempted to expand authority")
            _reject_authority(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_authority(child)


def _identity_body(value: Any) -> str:
    return canonical_digest(value)


@dataclass(frozen=True, slots=True)
class StudyRecord:
    """A source record that may be indexed only through an explicit visibility view."""

    record_type: str
    payload: dict[str, Any]
    visibility: Visibility
    record_identity: str = ""
    epoch: int = 0

    def __post_init__(self) -> None:
        _nonempty(self.record_type, "study record type")
        if not isinstance(self.payload, dict):
            raise DistillationError("study record payload must be an object")
        _reject_authority(self.payload)
        if not isinstance(self.visibility, Visibility) or self.epoch < 0:
            raise DistillationError("study record visibility and epoch are invalid")
        if len(canonical_json(self.payload)) > MAX_RECORD_BYTES:
            raise DistillationError("study record payload is too large")
        if self.record_identity and not isinstance(self.record_identity, str):
            raise DistillationError("study record identity is invalid")

    @property
    def identity(self) -> str:
        return self.record_identity or _identity_body(
            {
                "record_type": self.record_type,
                "payload": self.payload,
                "visibility": self.visibility.value,
                "epoch": self.epoch,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnel-study-source-record/0.4",
            "record_type": self.record_type,
            "record_identity": self.identity,
            "payload": dict(self.payload),
            "visibility": self.visibility.value,
            "epoch": self.epoch,
            "authority": "evidence-record",
            "semantics": "source-evidence; not-a-verdict",
        }


class StudyDataAccess:
    """Explicit, fail-closed view over source records.

    Ordinary development views may read development and selection-observed records only.
    Transfer evaluation gets a separate view that may read transfer-hidden records after
    a prediction is frozen. Future-final records are never available here.
    """

    def __init__(
        self,
        records: Sequence[StudyRecord],
        *,
        allowed_visibility: Sequence[Visibility] = (Visibility.DEVELOPMENT,),
        purpose: str = "development-study",
    ) -> None:
        if Visibility.FUTURE_FINAL in allowed_visibility:
            raise VisibilityViolation("future-final evidence is unavailable to MNEL study code")
        if not allowed_visibility:
            raise VisibilityViolation("study access requires an explicit visibility set")
        if purpose == "development-study" and Visibility.TRANSFER_HIDDEN in allowed_visibility:
            raise VisibilityViolation("development study access may not include transfer-hidden evidence")
        self._records: dict[str, StudyRecord] = {}
        for record in records:
            if record.identity in self._records and self._records[record.identity] != record:
                raise DistillationError("study record identity collision")
            self._records[record.identity] = record
        self.allowed_visibility = tuple(allowed_visibility)
        self.purpose = purpose

    @classmethod
    def development(cls, records: Sequence[StudyRecord]) -> "StudyDataAccess":
        return cls(
            records,
            allowed_visibility=(Visibility.DEVELOPMENT, Visibility.SELECTION_OBSERVED),
            purpose="development-study",
        )

    @classmethod
    def transfer_evaluator(cls, records: Sequence[StudyRecord]) -> "StudyDataAccess":
        return cls(
            records,
            allowed_visibility=(Visibility.TRANSFER_HIDDEN,),
            purpose="transfer-evaluator",
        )

    @property
    def dataset_identity(self) -> str:
        return _identity_body(
            {
                "purpose": self.purpose,
                "allowed_visibility": [item.value for item in self.allowed_visibility],
                "records": [record.to_dict() for record in self.records()],
            }
        )

    def records(self, record_type: str | None = None) -> tuple[StudyRecord, ...]:
        values = [
            record
            for record in self._records.values()
            if record.visibility in self.allowed_visibility
            and (record_type is None or record.record_type == record_type)
        ]
        return tuple(sorted(values, key=lambda item: item.identity))

    def get(self, identity: str) -> StudyRecord:
        try:
            record = self._records[identity]
        except KeyError as error:
            raise VisibilityViolation(f"study record is unavailable: {identity}") from error
        if record.visibility not in self.allowed_visibility:
            raise VisibilityViolation(
                f"study view {self.purpose} may not read {record.visibility.value} evidence"
            )
        return record

    def require(self, identities: Sequence[str]) -> tuple[StudyRecord, ...]:
        return tuple(self.get(identity) for identity in identities)

    def without_types(self, record_types: Sequence[str]) -> "StudyDataAccess":
        blocked = set(record_types)
        return StudyDataAccess(
            tuple(record for record in self.records() if record.record_type not in blocked),
            allowed_visibility=self.allowed_visibility,
            purpose=self.purpose,
        )


@dataclass(frozen=True, slots=True)
class SemanticGroup:
    source_record_ids: tuple[str, ...]
    feature_extractor_identity: str
    clustering_method_identity: str
    parameters: dict[str, Any]
    representative_record_ids: tuple[str, ...]
    scope: dict[str, Any]
    creation_epoch: int
    limitations: tuple[str, ...]
    group_identity: str = ""
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        if not self.source_record_ids or len(set(self.source_record_ids)) != len(self.source_record_ids):
            raise DistillationError("semantic groups require unique source lineage")
        _nonempty(self.feature_extractor_identity, "group feature extractor identity")
        _nonempty(self.clustering_method_identity, "group method identity")
        _bounded(self.parameters, "group parameters")
        _bounded(self.scope, "group scope")
        if not self.limitations or self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise DistillationError("groups require limitations and diagnostic-only authority")
        if self.creation_epoch < 0:
            raise DistillationError("group epoch cannot be negative")
        if self.group_identity and self.group_identity != self.content_identity:
            raise DistillationError("semantic group identity does not match content")

    @property
    def content_identity(self) -> str:
        return _identity_body(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-semantic-group/0.4",
            "source_record_ids": list(self.source_record_ids),
            "feature_extractor_identity": self.feature_extractor_identity,
            "clustering_method_identity": self.clustering_method_identity,
            "parameters": dict(self.parameters),
            "representative_record_ids": list(self.representative_record_ids),
            "scope": dict(self.scope),
            "creation_epoch": self.creation_epoch,
            "limitations": list(self.limitations),
            "authority": self.authority,
            "semantics": "source-preserving-retrieval-group; not-a-verdict",
        }
        if include_identity:
            value["group_identity"] = self.group_identity or self.content_identity
        return value


def _feature_key(record: StudyRecord) -> tuple[str, ...]:
    payload = record.payload
    tags = payload.get("tags", ())
    if isinstance(tags, str):
        tags = (tags,)
    if not isinstance(tags, (list, tuple)):
        tags = ()
    return (
        record.record_type,
        str(payload.get("artifact_type", "")),
        str(payload.get("uncertainty_class", "")),
        str(payload.get("snapshot_type", "")),
        "|".join(sorted(str(item) for item in tags)),
    )


def reference_feature_groups(
    access: StudyDataAccess,
    *,
    feature_extractor_identity: str = "mnel-reference-features/0.4",
    clustering_method_identity: str = "mnel-reference-feature-key-cluster/0.4",
    max_groups: int = MAX_GROUPS,
) -> tuple[SemanticGroup, ...]:
    if max_groups < 1 or max_groups > MAX_GROUPS:
        raise DistillationError("group budget is outside its bounded range")
    buckets: dict[tuple[str, ...], list[StudyRecord]] = {}
    for record in access.records():
        bucket = buckets.setdefault(_feature_key(record), [])
        if len(bucket) < MAX_GROUP_RECORDS:
            bucket.append(record)
    groups: list[SemanticGroup] = []
    for key, values in sorted(buckets.items()):
        source_ids = tuple(item.identity for item in values)
        group_draft = SemanticGroup(
            source_ids,
            feature_extractor_identity,
            clustering_method_identity,
            {"feature_key": list(key), "bounded": True},
            (source_ids[0],),
            {"visibility": [item.value for item in access.allowed_visibility]},
            max(item.epoch for item in values),
            (
                "reference grouping uses explicit record features, not semantic understanding",
                "source records remain independently addressable",
            ),
        )
        groups.append(
            SemanticGroup(
                group_draft.source_record_ids,
                group_draft.feature_extractor_identity,
                group_draft.clustering_method_identity,
                group_draft.parameters,
                group_draft.representative_record_ids,
                group_draft.scope,
                group_draft.creation_epoch,
                group_draft.limitations,
                group_draft.content_identity,
            )
        )
    if len(groups) > max_groups:
        groups = sorted(groups, key=lambda item: item.group_identity)[:max_groups]
    return tuple(sorted(groups, key=lambda item: item.group_identity))


@dataclass(frozen=True, slots=True)
class DistilledPrinciple:
    statement: str
    scope: dict[str, Any]
    attribution_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    counterexample_record_ids: tuple[str, ...]
    falsifier: str
    transfer_status: TransferStatus
    maturity: str
    limitations: tuple[str, ...]
    principle_identity: str = ""
    authority: str = AUTHORITY_PROPOSAL_ONLY

    def __post_init__(self) -> None:
        _nonempty(self.statement, "principle statement")
        _nonempty(self.falsifier, "principle falsifier")
        if not self.attribution_ids or not self.source_record_ids:
            raise DistillationError("principles must preserve attribution and source lineage")
        _bounded(self.scope, "principle scope")
        if not self.limitations or self.authority != AUTHORITY_PROPOSAL_ONLY:
            raise DistillationError("principles require limitations and proposal-only authority")
        if self.principle_identity and self.principle_identity != self.content_identity:
            raise DistillationError("principle identity does not match content")

    @property
    def content_identity(self) -> str:
        return _identity_body(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-distilled-principle/0.4",
            "statement": self.statement,
            "scope": dict(self.scope),
            "attribution_ids": list(self.attribution_ids),
            "source_record_ids": list(self.source_record_ids),
            "counterexample_record_ids": list(self.counterexample_record_ids),
            "falsifier": self.falsifier,
            "transfer_status": self.transfer_status.value,
            "maturity": self.maturity,
            "limitations": list(self.limitations),
            "authority": self.authority,
            "semantics": "provisional-distillation; not-a-verdict",
        }
        if include_identity:
            value["principle_identity"] = self.principle_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class NegativeMemoryEntry:
    memory_type: str
    statement: str
    source_record_ids: tuple[str, ...]
    prohibited_contexts: tuple[str, ...]
    reconsideration_condition: str
    scope: dict[str, Any]
    outcome_identity: str
    memory_identity: str = ""
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        _nonempty(self.memory_type, "negative memory type")
        _nonempty(self.statement, "negative memory statement")
        _nonempty(self.reconsideration_condition, "negative memory reconsideration condition")
        _nonempty(self.outcome_identity, "negative memory outcome identity")
        if not self.source_record_ids or not self.prohibited_contexts:
            raise DistillationError("negative memory requires source lineage and contexts")
        _bounded(self.scope, "negative memory scope")
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise DistillationError("negative memory is diagnostic-only")
        if self.memory_identity and self.memory_identity != self.content_identity:
            raise DistillationError("negative memory identity does not match content")

    @property
    def content_identity(self) -> str:
        return _identity_body(self.to_dict(include_identity=False))

    def matches(self, context: dict[str, Any]) -> bool:
        terms: set[str] = set()
        for value in context.values():
            if isinstance(value, str):
                terms.add(value)
            elif isinstance(value, (list, tuple, set)):
                terms.update(str(item) for item in value)
        return bool(terms.intersection(self.prohibited_contexts))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-negative-memory/0.4",
            "memory_type": self.memory_type,
            "statement": self.statement,
            "source_record_ids": list(self.source_record_ids),
            "prohibited_contexts": list(self.prohibited_contexts),
            "reconsideration_condition": self.reconsideration_condition,
            "scope": dict(self.scope),
            "outcome_identity": self.outcome_identity,
            "authority": self.authority,
            "semantics": "negative-evidence; not-a-verdict",
        }
        if include_identity:
            value["memory_identity"] = self.memory_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class DistilledStrategy:
    trigger_conditions: tuple[str, ...]
    preconditions: tuple[str, ...]
    expected_effect: dict[str, Any]
    known_failure_modes: tuple[str, ...]
    negative_memory_ids: tuple[str, ...]
    counterexample_record_ids: tuple[str, ...]
    causal_attribution_ids: tuple[str, ...]
    supporting_source_record_ids: tuple[str, ...]
    transfer_evidence_ids: tuple[str, ...]
    scope: dict[str, Any]
    calibration_metadata: dict[str, Any]
    rollback_lineage: tuple[str, ...]
    transfer_status: TransferStatus
    strategy_identity: str = ""
    authority: str = AUTHORITY_PROPOSAL_ONLY

    def __post_init__(self) -> None:
        if not self.trigger_conditions or not self.preconditions:
            raise DistillationError("strategies require trigger conditions and preconditions")
        if not self.known_failure_modes or not self.supporting_source_record_ids:
            raise DistillationError("strategies require failure modes and source lineage")
        if not self.causal_attribution_ids:
            raise DistillationError("strategies require causal attribution lineage")
        _bounded(self.expected_effect, "strategy expected effect")
        _bounded(self.scope, "strategy scope")
        _bounded(self.calibration_metadata, "strategy calibration metadata")
        if self.authority != AUTHORITY_PROPOSAL_ONLY:
            raise DistillationError("strategies are proposal-only")
        if self.strategy_identity and self.strategy_identity != self.content_identity:
            raise DistillationError("strategy identity does not match content")

    @property
    def content_identity(self) -> str:
        return _identity_body(self.to_dict(include_identity=False))

    def conflicts(self, context: dict[str, Any], memory: Sequence[NegativeMemoryEntry]) -> tuple[str, ...]:
        return tuple(
            item.memory_identity or item.content_identity
            for item in memory
            if item.matches(context)
            and (not self.negative_memory_ids or (item.memory_identity or item.content_identity) in self.negative_memory_ids)
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-distilled-strategy/0.4",
            "trigger_conditions": list(self.trigger_conditions),
            "preconditions": list(self.preconditions),
            "expected_effect": dict(self.expected_effect),
            "known_failure_modes": list(self.known_failure_modes),
            "negative_memory_ids": list(self.negative_memory_ids),
            "counterexample_record_ids": list(self.counterexample_record_ids),
            "causal_attribution_ids": list(self.causal_attribution_ids),
            "supporting_source_record_ids": list(self.supporting_source_record_ids),
            "transfer_evidence_ids": list(self.transfer_evidence_ids),
            "scope": dict(self.scope),
            "calibration_metadata": dict(self.calibration_metadata),
            "rollback_lineage": list(self.rollback_lineage),
            "transfer_status": self.transfer_status.value,
            "authority": self.authority,
            "semantics": "reusable-proposal; not-a-verdict",
        }
        if include_identity:
            value["strategy_identity"] = self.strategy_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class TransferPrediction:
    strategy_identity: str
    source_development_identities: tuple[str, ...]
    transfer_environment_identity: str
    predicted_effect: dict[str, Any]
    prediction_identity: str = ""
    authority: str = AUTHORITY_PROPOSAL_ONLY

    def __post_init__(self) -> None:
        if not self.source_development_identities:
            raise DistillationError("transfer prediction requires development lineage")
        _nonempty(self.strategy_identity, "transfer strategy identity")
        _nonempty(self.transfer_environment_identity, "transfer environment identity")
        _bounded(self.predicted_effect, "transfer predicted effect")
        if self.authority != AUTHORITY_PROPOSAL_ONLY:
            raise DistillationError("transfer predictions are proposal-only")
        if self.prediction_identity and self.prediction_identity != self.content_identity:
            raise DistillationError("transfer prediction identity does not match content")

    @property
    def content_identity(self) -> str:
        return _identity_body(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-transfer-prediction/0.4",
            "strategy_identity": self.strategy_identity,
            "source_development_identities": list(self.source_development_identities),
            "transfer_environment_identity": self.transfer_environment_identity,
            "predicted_effect": dict(self.predicted_effect),
            "authority": self.authority,
            "semantics": "frozen-before-transfer-observation; not-a-verdict",
        }
        if include_identity:
            value["prediction_identity"] = self.prediction_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class TransferEvaluation:
    prediction_identity: str
    observed_outcome_identity: str
    evaluator_evidence_identity: str
    status: TransferStatus
    rollback_lineage: tuple[str, ...]
    evaluation_identity: str = ""
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        for value, label in (
            (self.prediction_identity, "prediction identity"),
            (self.observed_outcome_identity, "observed outcome identity"),
            (self.evaluator_evidence_identity, "evaluator evidence identity"),
        ):
            _nonempty(value, label)
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise DistillationError("transfer evaluations are diagnostic evidence")
        if self.evaluation_identity and self.evaluation_identity != self.content_identity:
            raise DistillationError("transfer evaluation identity does not match content")

    @property
    def content_identity(self) -> str:
        return _identity_body(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-transfer-evaluation/0.4",
            "prediction_identity": self.prediction_identity,
            "observed_outcome_identity": self.observed_outcome_identity,
            "evaluator_evidence_identity": self.evaluator_evidence_identity,
            "status": self.status.value,
            "rollback_lineage": list(self.rollback_lineage),
            "authority": self.authority,
            "semantics": "transfer-evidence; not-a-verdict",
        }
        if include_identity:
            value["evaluation_identity"] = self.evaluation_identity or self.content_identity
        return value


class TransferWorkflow:
    def __init__(self) -> None:
        self._predictions: dict[str, TransferPrediction] = {}
        self._evaluations: dict[str, TransferEvaluation] = {}

    def freeze_prediction(
        self,
        strategy: DistilledStrategy,
        *,
        transfer_environment_identity: str,
        predicted_effect: dict[str, Any],
    ) -> TransferPrediction:
        prediction = TransferPrediction(
            strategy.strategy_identity or strategy.content_identity,
            tuple(strategy.supporting_source_record_ids),
            transfer_environment_identity,
            dict(predicted_effect),
        )
        prediction = TransferPrediction(
            prediction.strategy_identity,
            prediction.source_development_identities,
            prediction.transfer_environment_identity,
            prediction.predicted_effect,
            prediction.content_identity,
        )
        identity = prediction.prediction_identity
        if identity in self._predictions:
            raise DistillationError("duplicate transfer prediction")
        self._predictions[identity] = prediction
        return prediction

    def finalize(
        self,
        prediction: TransferPrediction,
        *,
        observed_outcome_identity: str,
        evaluator_evidence_identity: str,
        status: TransferStatus,
        rollback_lineage: Sequence[str] = (),
    ) -> TransferEvaluation:
        identity = prediction.prediction_identity or prediction.content_identity
        if identity not in self._predictions:
            raise DistillationError("transfer prediction was not frozen by this workflow")
        if identity in self._evaluations:
            raise DistillationError("transfer prediction is already finalized")
        evaluation = TransferEvaluation(
            identity,
            observed_outcome_identity,
            evaluator_evidence_identity,
            status,
            tuple(rollback_lineage),
        )
        evaluation = TransferEvaluation(
            evaluation.prediction_identity,
            evaluation.observed_outcome_identity,
            evaluation.evaluator_evidence_identity,
            evaluation.status,
            evaluation.rollback_lineage,
            evaluation.content_identity,
        )
        self._evaluations[identity] = evaluation
        return evaluation

    def reject_same_candidate_repair(self, strategy_identity: str, evaluation: TransferEvaluation) -> None:
        prediction = self._predictions.get(evaluation.prediction_identity)
        if prediction is not None and prediction.strategy_identity == strategy_identity:
            raise VisibilityViolation("a strategy may not be repaired from its own hidden transfer result")


@dataclass(frozen=True, slots=True)
class StudyArm:
    arm_id: str
    kind: StudyArmKind
    allowed_information: tuple[str, ...]
    forbidden_information: tuple[str, ...]
    retrieval_mode: str
    memory_availability: tuple[str, ...]
    attribution_available: bool
    strategy_available: bool
    recursion_allowance: int
    budget: dict[str, int]
    authority: str = AUTHORITY_PROPOSAL_ONLY

    def __post_init__(self) -> None:
        _nonempty(self.arm_id, "study arm id")
        if self.recursion_allowance < 0 or self.authority != AUTHORITY_PROPOSAL_ONLY:
            raise DistillationError("study arms require bounded recursion and proposal authority")
        if not set(self.allowed_information).isdisjoint(self.forbidden_information):
            raise DistillationError("study arm information cannot be both allowed and forbidden")
        if set(self.budget) != {"operations", "wall_seconds", "candidates"} or any(
            not isinstance(value, int) or value < 1 for value in self.budget.values()
        ):
            raise DistillationError("study arm budgets are malformed")

    @property
    def arm_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-study-arm/0.4",
            "arm_id": self.arm_id,
            "kind": self.kind.value,
            "allowed_information": list(self.allowed_information),
            "forbidden_information": list(self.forbidden_information),
            "retrieval_mode": self.retrieval_mode,
            "memory_availability": list(self.memory_availability),
            "attribution_available": self.attribution_available,
            "strategy_available": self.strategy_available,
            "recursion_allowance": self.recursion_allowance,
            "budget": dict(self.budget),
            "authority": self.authority,
            "semantics": "study-arm-contract; not-a-verdict",
        }
        if include_identity:
            value["arm_identity"] = self.arm_identity
        return value


@dataclass(frozen=True, slots=True)
class AblationSpec:
    ablation_id: str
    kind: AblationKind
    source_arm_identity: str
    seed: int
    parameters: dict[str, Any]
    authority: str = AUTHORITY_PROPOSAL_ONLY

    def __post_init__(self) -> None:
        _nonempty(self.ablation_id, "ablation id")
        _nonempty(self.source_arm_identity, "ablation source arm identity")
        if self.seed < 0 or self.authority != AUTHORITY_PROPOSAL_ONLY:
            raise DistillationError("ablation seed or authority is invalid")
        _bounded(self.parameters, "ablation parameters")

    @property
    def ablation_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-ablation-spec/0.4",
            "ablation_id": self.ablation_id,
            "kind": self.kind.value,
            "source_arm_identity": self.source_arm_identity,
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "authority": self.authority,
            "semantics": "controlled-study-transformation; not-a-verdict",
        }
        if include_identity:
            value["ablation_identity"] = self.ablation_identity
        return value


@dataclass(frozen=True, slots=True)
class ShuffledAttributionControl:
    seed: int
    original_attribution_ids: tuple[str, ...]
    shuffled_attribution_ids: tuple[str, ...]
    source_study_identity: str
    control_identity: str = ""

    def __post_init__(self) -> None:
        if self.seed < 0 or not self.original_attribution_ids:
            raise DistillationError("shuffled attribution requires a seed and source ids")
        if len(self.original_attribution_ids) != len(self.shuffled_attribution_ids):
            raise DistillationError("shuffled attribution lengths differ")
        if sorted(self.original_attribution_ids) != sorted(self.shuffled_attribution_ids):
            raise DistillationError("shuffled attribution must preserve the id multiset")
        _nonempty(self.source_study_identity, "shuffled attribution source study")
        if self.control_identity and self.control_identity != self.content_identity:
            raise DistillationError("shuffled attribution identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-shuffled-attribution-control/0.4",
            "seed": self.seed,
            "original_attribution_ids": list(self.original_attribution_ids),
            "shuffled_attribution_ids": list(self.shuffled_attribution_ids),
            "source_study_identity": self.source_study_identity,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "control-only; original attribution remains immutable",
        }
        if include_identity:
            value["control_identity"] = self.control_identity or self.content_identity
        return value


def shuffle_attributions(
    attribution_ids: Sequence[str], *, seed: int, source_study_identity: str
) -> ShuffledAttributionControl:
    values = list(attribution_ids)
    if not values or seed < 0:
        raise DistillationError("shuffled attribution inputs are invalid")
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return ShuffledAttributionControl(seed, tuple(values), tuple(shuffled), source_study_identity)


@dataclass(frozen=True, slots=True)
class StudySpecification:
    study_id: str
    development_dataset_identity: str
    hidden_transfer_dataset_identity: str
    arms: tuple[StudyArm, ...]
    ablations: tuple[AblationSpec, ...]
    equal_budget: dict[str, int]
    study_identity: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.study_id, "study id")
        _nonempty(self.development_dataset_identity, "development dataset identity")
        _nonempty(self.hidden_transfer_dataset_identity, "hidden transfer dataset identity")
        if not self.arms:
            raise DistillationError("study requires at least one arm")
        if len({item.arm_id for item in self.arms}) != len(self.arms):
            raise DistillationError("study arm ids must be unique")
        if set(self.equal_budget) != {"operations", "wall_seconds", "candidates"} or any(
            not isinstance(value, int) or value < 1 for value in self.equal_budget.values()
        ):
            raise DistillationError("study equal-budget declaration is malformed")
        if any(item.budget != self.equal_budget for item in self.arms):
            raise DistillationError("study arms must use the declared equal budget")
        if self.study_identity and self.study_identity != self.content_identity:
            raise DistillationError("study identity does not match content")
        if not self.study_identity:
            object.__setattr__(self, "study_identity", self.content_identity)

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-distillation-study-spec/0.4",
            "study_id": self.study_id,
            "development_dataset_identity": self.development_dataset_identity,
            "hidden_transfer_dataset_identity": self.hidden_transfer_dataset_identity,
            "arms": [item.to_dict() for item in self.arms],
            "ablations": [item.to_dict() for item in self.ablations],
            "equal_budget": dict(self.equal_budget),
            "authority": AUTHORITY_PROPOSAL_ONLY,
            "semantics": "study-specification; not-a-verdict",
        }
        if include_identity:
            value["study_identity"] = self.study_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    trigger: str
    context: dict[str, Any]
    limit: int = 8
    include_negative: bool = True
    query_identity: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.trigger, "retrieval trigger")
        _bounded(self.context, "retrieval context")
        if self.limit < 1 or self.limit > MAX_RETRIEVAL_HITS:
            raise DistillationError("retrieval limit is outside its bounded range")
        if self.query_identity and self.query_identity != self.content_identity:
            raise DistillationError("retrieval query identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-retrieval-query/0.4",
            "trigger": self.trigger,
            "context": dict(self.context),
            "limit": self.limit,
            "include_negative": self.include_negative,
        }
        if include_identity:
            value["query_identity"] = self.query_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    record_class: str
    record_identity: str
    score: float
    reasons: tuple[str, ...]
    source_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not self.record_class or not self.record_identity:
            raise DistillationError("retrieval hit is malformed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_class": self.record_class,
            "record_identity": self.record_identity,
            "score": self.score,
            "reasons": list(self.reasons),
            "source_record_ids": list(self.source_record_ids),
        }


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query_identity: str
    hits: tuple[RetrievalHit, ...]
    negative_conflict_ids: tuple[str, ...]
    method_identity: str

    @property
    def result_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-retrieval-result/0.4",
            "query_identity": self.query_identity,
            "hits": [item.to_dict() for item in self.hits],
            "negative_conflict_ids": list(self.negative_conflict_ids),
            "method_identity": self.method_identity,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "heuristic-retrieval; not-a-verdict",
        }
        if include_identity:
            value["result_identity"] = self.result_identity
        return value


class RetrievalIndex:
    METHOD_IDENTITY = "mnel-reference-retrieval/0.4"

    def __init__(self) -> None:
        self._records: dict[str, StudyRecord] = {}
        self._groups: dict[str, SemanticGroup] = {}
        self._principles: dict[str, DistilledPrinciple] = {}
        self._strategies: dict[str, DistilledStrategy] = {}
        self._negative: dict[str, NegativeMemoryEntry] = {}

    def add_source_records(self, records: Sequence[StudyRecord]) -> None:
        for record in records:
            if record.visibility in {Visibility.TRANSFER_HIDDEN, Visibility.FUTURE_FINAL}:
                raise VisibilityViolation("retrieval indexes may not ingest hidden or future-final records")
            if record.identity in self._records and self._records[record.identity] != record:
                raise DistillationError("retrieval source identity collision")
            self._records[record.identity] = record

    def add_groups(self, groups: Sequence[SemanticGroup]) -> None:
        for item in groups:
            self._groups[item.group_identity or item.content_identity] = item

    def add_principles(self, principles: Sequence[DistilledPrinciple]) -> None:
        for item in principles:
            self._principles[item.principle_identity or item.content_identity] = item

    def add_strategies(self, strategies: Sequence[DistilledStrategy]) -> None:
        for item in strategies:
            self._strategies[item.strategy_identity or item.content_identity] = item

    def add_negative_memory(self, entries: Sequence[NegativeMemoryEntry]) -> None:
        for item in entries:
            self._negative[item.memory_identity or item.content_identity] = item

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        memory = tuple(self._negative.values()) if query.include_negative else ()
        conflicts = tuple(sorted(item.memory_identity or item.content_identity for item in memory if item.matches(query.context)))
        hits: list[RetrievalHit] = []
        for identity, strategy in self._strategies.items():
            score = 0.0
            reasons: list[str] = []
            if query.trigger in strategy.trigger_conditions:
                score += 4.0
                reasons.append("trigger-compatible")
            if any(str(value) in strategy.scope.values() for value in query.context.values()):
                score += 1.0
                reasons.append("scope-compatible")
            if strategy.transfer_status is TransferStatus.SUPPORTED:
                score += 2.0
                reasons.append("transfer-supported")
            strategy_conflicts = strategy.conflicts(query.context, memory)
            if strategy_conflicts:
                score -= 6.0
                reasons.append("negative-memory-conflict")
            if score > 0 or strategy_conflicts:
                hits.append(RetrievalHit("strategy", identity, score, tuple(reasons), strategy.supporting_source_record_ids))
        for identity, principle in self._principles.items():
            score = 1.0 if any(str(value) in principle.scope.values() for value in query.context.values()) else 0.0
            if score:
                hits.append(RetrievalHit("principle", identity, score, ("scope-compatible",), principle.source_record_ids))
        for identity, entry in self._negative.items():
            if query.include_negative and entry.matches(query.context):
                hits.append(RetrievalHit("negative-memory", identity, 3.0, ("prohibited-context-match",), entry.source_record_ids))
        for identity, record in self._records.items():
            values = {str(item) for item in record.payload.values() if isinstance(item, (str, int, float, bool))}
            if query.trigger in values or any(str(value) in values for value in query.context.values()):
                hits.append(RetrievalHit("source-record", identity, 0.5, ("explicit-feature-match",), (identity,)))
        hits.sort(key=lambda item: (-item.score, item.record_class, item.record_identity))
        return RetrievalResult(query.query_identity or query.content_identity, tuple(hits[: query.limit]), conflicts, self.METHOD_IDENTITY)


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    query_identity: str
    k: int
    relevant_count: int
    precision_at_k: float
    recall_at_k: float
    hit_rate: float
    reciprocal_rank: float
    duplicate_retrieval_rate: float
    evidence_reuse_depth: int
    source_diversity: int
    positive_negative_balance: float
    transfer_supported_strategy_rate: float
    unsupported_strategy_rate: float
    metric_identity: str = ""

    def __post_init__(self) -> None:
        values = (
            self.precision_at_k,
            self.recall_at_k,
            self.hit_rate,
            self.reciprocal_rank,
            self.duplicate_retrieval_rate,
            self.positive_negative_balance,
            self.transfer_supported_strategy_rate,
            self.unsupported_strategy_rate,
        )
        if any(not math.isfinite(value) for value in values):
            raise DistillationError("retrieval metrics must be finite")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-retrieval-metrics/0.4",
            "query_identity": self.query_identity,
            "k": self.k,
            "relevant_count": self.relevant_count,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "hit_rate": self.hit_rate,
            "reciprocal_rank": self.reciprocal_rank,
            "duplicate_retrieval_rate": self.duplicate_retrieval_rate,
            "evidence_reuse_depth": self.evidence_reuse_depth,
            "source_diversity": self.source_diversity,
            "positive_negative_balance": self.positive_negative_balance,
            "transfer_supported_strategy_rate": self.transfer_supported_strategy_rate,
            "unsupported_strategy_rate": self.unsupported_strategy_rate,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "retrieval-measurement; not-a-verdict",
        }
        if include_identity:
            value["metric_identity"] = self.metric_identity or self.content_identity
        return value


def evaluate_retrieval(
    result: RetrievalResult,
    relevant_ids: Sequence[str],
    *,
    strategy_status: dict[str, TransferStatus] | None = None,
) -> RetrievalMetrics:
    relevant = set(relevant_ids)
    hits = result.hits
    top_ids = [item.record_identity for item in hits]
    unique = set(top_ids)
    true_hits = [index for index, identity in enumerate(top_ids, 1) if identity in relevant]
    k = len(hits)
    relevant_count = len(relevant)
    precision = len(true_hits) / k if k else 0.0
    recall = len(true_hits) / relevant_count if relevant_count else 0.0
    reciprocal = 1.0 / true_hits[0] if true_hits else 0.0
    duplicate_rate = (len(top_ids) - len(unique)) / len(top_ids) if top_ids else 0.0
    strategy_hits = [item for item in hits if item.record_class == "strategy"]
    supported = sum(
        1 for item in strategy_hits if strategy_status and strategy_status.get(item.record_identity) is TransferStatus.SUPPORTED
    )
    unsupported = sum(
        1 for item in strategy_hits if not strategy_status or strategy_status.get(item.record_identity) is not TransferStatus.SUPPORTED
    )
    negative = sum(1 for item in hits if item.record_class == "negative-memory")
    positive_negative_balance = negative / (negative + len(strategy_hits)) if (negative + len(strategy_hits)) else 0.0
    depth = max((len(item.source_record_ids) for item in hits), default=0)
    diversity = len({source for item in hits for source in item.source_record_ids})
    return RetrievalMetrics(
        result.query_identity,
        k,
        relevant_count,
        precision,
        recall,
        float(bool(true_hits)),
        reciprocal,
        duplicate_rate,
        depth,
        diversity,
        positive_negative_balance,
        supported / len(strategy_hits) if strategy_hits else 0.0,
        unsupported / len(strategy_hits) if strategy_hits else 0.0,
    )


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    dataset_identity: str
    model_identity: str
    calibration_method_identity: str
    count: int
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    coverage: float
    abstention_rate: float
    out_of_distribution_rate: float
    metric_identity: str = ""

    def __post_init__(self) -> None:
        if self.count < 1 or any(
            not math.isfinite(value)
            for value in (
                self.brier_score,
                self.log_loss,
                self.expected_calibration_error,
                self.coverage,
                self.abstention_rate,
                self.out_of_distribution_rate,
            )
        ):
            raise DistillationError("calibration metrics are invalid")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-calibration-metrics/0.4",
            "dataset_identity": self.dataset_identity,
            "model_identity": self.model_identity,
            "calibration_method_identity": self.calibration_method_identity,
            "count": self.count,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "expected_calibration_error": self.expected_calibration_error,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "out_of_distribution_rate": self.out_of_distribution_rate,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "calibration-measurement; not-a-verdict",
        }
        if include_identity:
            value["metric_identity"] = self.metric_identity or self.content_identity
        return value


def calculate_calibration(
    predictions: Sequence[float | None],
    labels: Sequence[int | bool],
    *,
    dataset_identity: str,
    model_identity: str,
    calibration_method_identity: str = "mnel-reference-bins/0.4",
    out_of_distribution: Sequence[bool] = (),
    bins: int = 10,
) -> CalibrationMetrics:
    if len(predictions) != len(labels) or not predictions or bins < 1:
        raise DistillationError("calibration inputs are malformed")
    if out_of_distribution and len(out_of_distribution) != len(predictions):
        raise DistillationError("OOD flags must match calibration inputs")
    valid: list[tuple[float, float]] = []
    abstained = 0
    for prediction, label in zip(predictions, labels):
        if prediction is None:
            abstained += 1
            continue
        if not isinstance(prediction, (int, float)) or isinstance(prediction, bool) or not math.isfinite(prediction):
            raise DistillationError("calibration predictions must be finite numbers or None")
        if prediction < 0.0 or prediction > 1.0 or label not in (0, 1, False, True):
            raise DistillationError("calibration values are outside their bounds")
        valid.append((float(prediction), float(bool(label))))
    if not valid:
        raise DistillationError("calibration requires at least one non-abstained prediction")
    brier = sum((prediction - label) ** 2 for prediction, label in valid) / len(valid)
    log_loss = sum(
        -(label * math.log(max(prediction, 1e-12)) + (1 - label) * math.log(max(1 - prediction, 1e-12)))
        for prediction, label in valid
    ) / len(valid)
    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [item for item in valid if low <= item[0] < high or (index == bins - 1 and item[0] == high)]
        if members:
            confidence = sum(item[0] for item in members) / len(members)
            frequency = sum(item[1] for item in members) / len(members)
            ece += len(members) / len(valid) * abs(confidence - frequency)
    ood_rate = sum(bool(item) for item in out_of_distribution) / len(predictions) if out_of_distribution else 0.0
    return CalibrationMetrics(
        dataset_identity,
        model_identity,
        calibration_method_identity,
        len(predictions),
        brier,
        log_loss,
        ece,
        len(valid) / len(predictions),
        abstained / len(predictions),
        ood_rate,
    )


def validate_equal_budget(arms: Sequence[StudyArm]) -> dict[str, int]:
    if not arms:
        raise DistillationError("equal-budget validation requires study arms")
    budget = dict(arms[0].budget)
    if any(item.budget != budget for item in arms[1:]):
        raise DistillationError("study arms do not satisfy equal-budget control")
    return budget


def make_study_record(
    record_type: str,
    payload: dict[str, Any],
    *,
    visibility: Visibility = Visibility.DEVELOPMENT,
    epoch: int = 0,
    record_identity: str = "",
) -> StudyRecord:
    _reject_authority(payload)
    return StudyRecord(record_type, dict(payload), visibility, record_identity, epoch)


def build_distilled_strategy(
    *,
    trigger_conditions: Sequence[str],
    preconditions: Sequence[str],
    expected_effect: dict[str, Any],
    known_failure_modes: Sequence[str],
    negative_memory_ids: Sequence[str],
    counterexample_record_ids: Sequence[str],
    causal_attribution_ids: Sequence[str],
    supporting_source_record_ids: Sequence[str],
    transfer_evidence_ids: Sequence[str],
    scope: dict[str, Any],
    calibration_metadata: dict[str, Any] | None = None,
    rollback_lineage: Sequence[str] = (),
    transfer_status: TransferStatus = TransferStatus.UNTESTED,
) -> DistilledStrategy:
    draft = DistilledStrategy(
        tuple(trigger_conditions),
        tuple(preconditions),
        dict(expected_effect),
        tuple(known_failure_modes),
        tuple(negative_memory_ids),
        tuple(counterexample_record_ids),
        tuple(causal_attribution_ids),
        tuple(dict.fromkeys(supporting_source_record_ids)),
        tuple(transfer_evidence_ids),
        dict(scope),
        dict(calibration_metadata or {"method": "unspecified"}),
        tuple(rollback_lineage),
        transfer_status,
    )
    return DistilledStrategy(
        draft.trigger_conditions,
        draft.preconditions,
        draft.expected_effect,
        draft.known_failure_modes,
        draft.negative_memory_ids,
        draft.counterexample_record_ids,
        draft.causal_attribution_ids,
        draft.supporting_source_record_ids,
        draft.transfer_evidence_ids,
        draft.scope,
        draft.calibration_metadata,
        draft.rollback_lineage,
        draft.transfer_status,
        draft.content_identity,
    )


def build_distilled_principle(
    *,
    statement: str,
    scope: dict[str, Any],
    attribution_ids: Sequence[str],
    source_record_ids: Sequence[str],
    counterexample_record_ids: Sequence[str],
    falsifier: str,
    transfer_status: TransferStatus = TransferStatus.UNTESTED,
    maturity: str = "provisional",
    limitations: Sequence[str] = ("source-bound; not independently validated",),
) -> DistilledPrinciple:
    draft = DistilledPrinciple(
        statement,
        dict(scope),
        tuple(attribution_ids),
        tuple(dict.fromkeys(source_record_ids)),
        tuple(counterexample_record_ids),
        falsifier,
        transfer_status,
        maturity,
        tuple(limitations),
    )
    return DistilledPrinciple(
        draft.statement,
        draft.scope,
        draft.attribution_ids,
        draft.source_record_ids,
        draft.counterexample_record_ids,
        draft.falsifier,
        draft.transfer_status,
        draft.maturity,
        draft.limitations,
        draft.content_identity,
    )


def run_reference_distill_study(workspace: str | Path | None = None) -> dict[str, Any]:
    """Run a deterministic, held-out transfer study with controls and measurements."""

    from .reference_provider import train_transition_frequency
    from .snapshots import SnapshotStore, decode_snapshot, transition_snapshot

    snapshot_store = SnapshotStore()
    snapshot_identities = {
        "transition-a": transition_snapshot(
            b"cold",
            b"warm",
            producer_identity="mnel-distill-reference-producer/0.4",
            source_identity="sha256:development-source-a",
            dependency_identity="sha256:development-dependency",
            feature_extractor_identity="sha256:transition-feature-extractor",
        ),
        "transition-b": transition_snapshot(
            b"hot",
            b"warm",
            producer_identity="mnel-distill-reference-producer/0.4",
            source_identity="sha256:development-source-b",
            dependency_identity="sha256:development-dependency",
            feature_extractor_identity="sha256:transition-feature-extractor",
        ),
    }
    for snapshot in snapshot_identities.values():
        snapshot_store.register(snapshot)
    development_snapshots = {
        key: value.snapshot_identity for key, value in snapshot_identities.items()
    }
    development_records = (
        make_study_record(
            "experience-episode",
            {
                "artifact_type": "routing",
                "uncertainty_class": "unexpected-transition",
                "snapshot_type": "transition",
                "tags": ["transition", "routing"],
                "snapshot_identity": development_snapshots["transition-a"],
                "outcome": "success",
            },
            record_identity="sha256:development-episode-a",
            epoch=1,
        ),
        make_study_record(
            "experience-episode",
            {
                "artifact_type": "routing",
                "uncertainty_class": "unexpected-transition",
                "snapshot_type": "transition",
                "tags": ["transition", "routing"],
                "snapshot_identity": development_snapshots["transition-b"],
                "outcome": "success",
            },
            record_identity="sha256:development-episode-b",
            epoch=1,
        ),
        make_study_record(
            "counterexample",
            {
                "artifact_type": "routing",
                "uncertainty_class": "unexpected-transition",
                "snapshot_type": "transition",
                "tags": ["transition", "unsupported-provider"],
                "outcome": "regression",
            },
            record_identity="sha256:development-counterexample",
            epoch=2,
        ),
        make_study_record(
            "causal-attribution",
            {"attribution_id": "sha256:attribution-a", "supporting_episode_ids": ["sha256:development-episode-a"]},
            record_identity="sha256:attribution-record-a",
            epoch=2,
        ),
    )
    hidden_records = (
        make_study_record(
            "transfer-outcome",
            {"strategy_identity": "pending", "outcome": "supported", "metric": 0.92},
            visibility=Visibility.TRANSFER_HIDDEN,
            record_identity="sha256:hidden-transfer-outcome",
            epoch=3,
        ),
        make_study_record(
            "future-final-reference",
            {"outcome": "future-final-only"},
            visibility=Visibility.FUTURE_FINAL,
            record_identity="sha256:future-final-record",
            epoch=4,
        ),
    )
    all_records = development_records + hidden_records
    development = StudyDataAccess.development(all_records)
    try:
        development.get("sha256:hidden-transfer-outcome")
        raise DistillationError("development access unexpectedly read hidden transfer")
    except VisibilityViolation:
        pass
    try:
        development.get("sha256:future-final-record")
        raise DistillationError("development access unexpectedly read future-final evidence")
    except VisibilityViolation:
        pass
    groups = reference_feature_groups(development)
    attribution = Attribution(
        "sha256:attribution-a",
        "reference-distill-study",
        "intervention-routing",
        "sha256:evaluation-a",
        "supported-with-alternatives",
        ("immediate",),
        ("sha256:development-episode-a", "sha256:development-episode-b"),
        ("unsupported-provider effect",),
        ("sha256:development-episode-a", "sha256:development-episode-b", "sha256:attribution-record-a"),
    )
    principle = build_distilled_principle(
        statement="Transition-aware routing may improve the declared routing fixture when its provider assumptions hold.",
        scope={"artifact_type": "routing", "snapshot_type": "transition"},
        attribution_ids=(attribution.attribution_id,),
        source_record_ids=attribution.source_record_ids,
        counterexample_record_ids=("sha256:development-counterexample",),
        falsifier="held-out transfer does not reproduce the predicted bounded effect",
    )
    negative = NegativeMemoryEntry(
        "counterexample",
        "Do not apply transition-aware routing in the unsupported-provider context without a new probe.",
        ("sha256:development-counterexample",),
        ("unsupported-provider",),
        "a new independent probe supports the provider assumption",
        {"artifact_type": "routing", "snapshot_type": "transition"},
        "sha256:development-counterexample",
    )
    negative = NegativeMemoryEntry(
        negative.memory_type,
        negative.statement,
        negative.source_record_ids,
        negative.prohibited_contexts,
        negative.reconsideration_condition,
        negative.scope,
        negative.outcome_identity,
        negative.content_identity,
    )
    strategy = build_distilled_strategy(
        trigger_conditions=("unexpected-transition",),
        preconditions=("artifact_type=routing", "snapshot_type=transition", "provider-assumption-supported"),
        expected_effect={"exact_target_rate": "increase", "retention": "no-regression"},
        known_failure_modes=("unsupported-provider", "unprobed-transfer-context"),
        negative_memory_ids=(negative.memory_identity,),
        counterexample_record_ids=negative.source_record_ids,
        causal_attribution_ids=(attribution.attribution_id,),
        supporting_source_record_ids=attribution.source_record_ids,
        transfer_evidence_ids=(),
        scope={"artifact_type": "routing", "snapshot_type": "transition"},
        calibration_metadata={"method": "reference-fixed-calibration", "dataset": "development"},
    )
    index = RetrievalIndex()
    index.add_source_records(development.records())
    index.add_groups(groups)
    index.add_principles((principle,))
    index.add_strategies((strategy,))
    index.add_negative_memory((negative,))
    query = RetrievalQuery(
        "unexpected-transition",
        {"artifact_type": "routing", "snapshot_type": "transition"},
        limit=8,
    )
    retrieval = index.retrieve(query)
    risky_query = RetrievalQuery(
        "unexpected-transition",
        {"artifact_type": "routing", "snapshot_type": "transition", "provider": "unsupported-provider"},
        limit=8,
    )
    risky_retrieval = index.retrieve(risky_query)
    ablated_index = RetrievalIndex()
    ablated_index.add_source_records(development.records())
    ablated_index.add_groups(groups)
    ablated_index.add_principles((principle,))
    ablated_index.add_strategies((strategy,))
    ablated_retrieval = ablated_index.retrieve(risky_query)
    strategy_status = {strategy.strategy_identity: strategy.transfer_status}
    retrieval_metrics = evaluate_retrieval(retrieval, (strategy.strategy_identity,), strategy_status=strategy_status)
    calibration = calculate_calibration(
        (0.8, 0.2, 0.6),
        (True, False, True),
        dataset_identity=development.dataset_identity,
        model_identity="mnel-reference-distilled-strategy/0.4",
        out_of_distribution=(False, False, True),
    )
    provider_access = StudyDataAccess.development(development_records[:2])
    provider_model = train_transition_frequency(provider_access, snapshot_store)
    provider_observation = provider_model.infer(snapshot_identities["transition-a"])
    reloaded_provider = type(provider_model).load(provider_model.serialize())
    reloaded_observation = reloaded_provider.infer(snapshot_identities["transition-a"])
    reference_view = decode_snapshot(snapshot_identities["transition-a"])
    reference_changed = bool(
        hasattr(reference_view, "previous_state")
        and reference_view.previous_state != reference_view.next_state
    )
    provider_baseline_comparison = {
        "provider_observation_identity": provider_observation.observation_identity
        or provider_observation.content_identity,
        "reloaded_observation_identity": reloaded_observation.observation_identity
        or reloaded_observation.content_identity,
        "deterministic_reference": {
            "condition_observed": reference_changed,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
        },
        "heuristic_nonempty_transition": {
            "score": float(bool(reference_view.previous_state or reference_view.next_state)),
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
        },
        "random_control": {
            "seed": 17,
            "score": 0.5,
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
        },
        "classical_hmm_baseline": {
            "provider_id": "state.hidden-markov-model",
            "status": "not-input-compatible",
            "reason": "native HMM consumes bounded numeric state sequences; this fixture uses byte transition views",
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
        },
        "provider_is_verifier": False,
        "authority": AUTHORITY_DIAGNOSTIC_ONLY,
        "semantics": "diagnostic-comparison; not-a-verdict",
    }
    arms = (
        StudyArm("A0", StudyArmKind.RANDOM, ("development",), ("attribution", "transfer-hidden"), "random", (), False, False, 0, {"operations": 100, "wall_seconds": 30, "candidates": 4}),
        StudyArm("A4", StudyArmKind.ATTRIBUTION, ("development", "attribution"), ("transfer-hidden",), "lineage-aware", ("positive", "negative"), True, True, 0, {"operations": 100, "wall_seconds": 30, "candidates": 4}),
        StudyArm("A5", StudyArmKind.TRANSFER_GATED, ("development", "attribution"), ("transfer-hidden",), "lineage-aware", ("positive", "negative"), True, True, 0, {"operations": 100, "wall_seconds": 30, "candidates": 4}),
    )
    shuffled = shuffle_attributions(
        (attribution.attribution_id, "sha256:attribution-b"),
        seed=17,
        source_study_identity="sha256:reference-distill-study",
    )
    ablations = tuple(
        AblationSpec(f"ablation-{kind.value}", kind, arms[1].arm_identity, 17, {})
        for kind in (
            AblationKind.AGGREGATE_ONLY,
            AblationKind.SHUFFLED_ATTRIBUTION,
            AblationKind.RANDOM_PROPOSAL,
            AblationKind.SUCCESS_MEMORY,
            AblationKind.NEGATIVE_MEMORY,
            AblationKind.FIXED_POLICY,
            AblationKind.EQUAL_BUDGET,
            AblationKind.HIDDEN_TRANSFER,
        )
    )
    specification = StudySpecification(
        "reference-distill-study",
        development.dataset_identity,
        StudyDataAccess.transfer_evaluator(all_records).dataset_identity,
        arms,
        ablations,
        validate_equal_budget(arms),
    )
    specification = StudySpecification(
        specification.study_id,
        specification.development_dataset_identity,
        specification.hidden_transfer_dataset_identity,
        specification.arms,
        specification.ablations,
        specification.equal_budget,
        specification.content_identity,
    )
    workflow = TransferWorkflow()
    prediction = workflow.freeze_prediction(
        strategy,
        transfer_environment_identity="sha256:held-out-transfer-environment",
        predicted_effect={"exact_target_rate": {"direction": "increase", "minimum": 0.9}},
    )
    hidden_access = StudyDataAccess.transfer_evaluator(all_records)
    hidden = hidden_access.get("sha256:hidden-transfer-outcome")
    evaluation = workflow.finalize(
        prediction,
        observed_outcome_identity=hidden.identity,
        evaluator_evidence_identity="sha256:external-evaluator-evidence",
        status=TransferStatus.SUPPORTED,
    )
    report_identity_body = {
        "specification": specification.to_dict(),
        "groups": [item.to_dict() for item in groups],
        "principle": principle.to_dict(),
        "strategy": strategy.to_dict(),
        "negative_memory": negative.to_dict(),
        "retrieval": retrieval.to_dict(),
        "risky_retrieval": risky_retrieval.to_dict(),
        "ablated_retrieval": ablated_retrieval.to_dict(),
        "retrieval_metrics": retrieval_metrics.to_dict(),
        "calibration": calibration.to_dict(),
        "shuffled_attribution": shuffled.to_dict(),
        "prediction": prediction.to_dict(),
        "evaluation": evaluation.to_dict(),
        "provider_model": provider_model.to_dict(),
        "provider_observation": provider_observation.to_dict(),
        "provider_baseline_comparison": provider_baseline_comparison,
    }
    study_identity = canonical_digest(report_identity_body)
    report = {
        "schema": "mnel-distillation-study-report/0.4",
        "study_identity": study_identity,
        "study_specification_identity": specification.study_identity or specification.content_identity,
        "development_dataset_identity": development.dataset_identity,
        "hidden_transfer_dataset_identity": StudyDataAccess.transfer_evaluator(all_records).dataset_identity,
        "group_count": len(groups),
        "candidate_count": 1,
        "retrieval_metrics": retrieval_metrics.to_dict(),
        "calibration_metrics": calibration.to_dict(),
        "transfer_status": evaluation.status.value,
        "transfer_prediction_frozen_before_evaluation": True,
        "negative_memory_conflicts": list(risky_retrieval.negative_conflict_ids),
        "negative_memory_demoted_strategy": any(
            item.record_identity == strategy.strategy_identity and item.score < 0 for item in risky_retrieval.hits
        ),
        "negative_memory_ablation_strategy_score": next(
            (item.score for item in ablated_retrieval.hits if item.record_identity == strategy.strategy_identity),
            None,
        ),
        "shuffled_attribution_control_identity": shuffled.control_identity or shuffled.content_identity,
        "arm_identities": [item.arm_identity for item in specification.arms],
        "ablation_identities": [item.ablation_identity for item in specification.ablations],
        "learned_provider_model_identity": provider_model.model_identity or provider_model.content_identity,
        "learned_provider_artifact_identity": provider_model.artifact_identity,
        "learned_provider_model_size_bytes": provider_model.model_size_bytes,
        "learned_provider_reload_reproduced": provider_observation.to_dict() == reloaded_observation.to_dict(),
        "provider_baseline_comparison": provider_baseline_comparison,
        "source_record_count": len(development.records()),
        "authority_violation_attempts": 2,
        "limitations": [
            "reference feature grouping is not semantic understanding",
            "synthetic transfer outcome is a fixture observation, not a universal claim",
            "the native HMM baseline is retained separately because its numeric state-sequence input is not compatible with this byte-transition fixture",
            "no external Forge, Fabric, RAVEL, or MNCS authority is implemented",
        ],
        "authority": AUTHORITY_DIAGNOSTIC_ONLY,
        "semantics": "measurement-report; not-a-verdict",
    }
    if workspace is not None:
        root = Path(workspace)
        ledger = EvidenceLedger(root / "distill-evidence.jsonl")
        for record in development_records:
            ledger.append("study-source-record", record.to_dict(), actor="mnel-distillation-study")
        for record in hidden_records:
            ledger.append("study-hidden-source-metadata", record.to_dict(), actor="mnel-transfer-evaluator")
        for record_type, value in (
            ("semantic-group", groups),
            ("distilled-principle", (principle,)),
            ("negative-memory", (negative,)),
            ("distilled-strategy", (strategy,)),
            ("study-arm", specification.arms),
            ("ablation-spec", specification.ablations),
            ("retrieval-result", (retrieval, risky_retrieval, ablated_retrieval)),
            ("retrieval-metrics", (retrieval_metrics,)),
            ("calibration-metrics", (calibration,)),
            ("shuffled-attribution-control", (shuffled,)),
            ("transfer-prediction", (prediction,)),
            ("transfer-evaluation", (evaluation,)),
        ):
            for item in value:
                payload = item.to_dict() if hasattr(item, "to_dict") else item
                ledger.append(record_type, payload, actor="mnel-distillation-study")
        ledger.append(
            "learned-provider-artifact",
            provider_model.to_dict(),
            actor="mnel-distillation-study",
        )
        ledger.append(
            "learned-provider-observation",
            provider_observation.to_dict(),
            actor="mnel-reference-provider",
        )
        ledger.append(
            "provider-baseline-comparison",
            provider_baseline_comparison,
            actor="mnel-distillation-study",
        )
        ledger.append("distillation-study-report", report, actor="mnel-distillation-study")
        report["ledger"] = ledger.summarize()
    return {
        "report": report,
        "study_identity": study_identity,
        "strategy": strategy.to_dict(),
        "negative_memory": negative.to_dict(),
        "groups": [item.to_dict() for item in groups],
        "retrieval": retrieval.to_dict(),
        "risky_retrieval": risky_retrieval.to_dict(),
        "ablated_retrieval": ablated_retrieval.to_dict(),
        "shuffled_attribution": shuffled.to_dict(),
        "prediction": prediction.to_dict(),
        "evaluation": evaluation.to_dict(),
        "specification": specification.to_dict(),
        "provider_model": provider_model.to_dict(),
        "provider_observation": provider_observation.to_dict(),
        "provider_baseline_comparison": provider_baseline_comparison,
    }
