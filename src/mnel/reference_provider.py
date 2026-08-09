"""Tiny deterministic learned-provider study implementation.

The transition-frequency model is intentionally small and inspectable. It is a learned
diagnostic provider, not a verifier and not an evaluator. Training is restricted to an
explicit development access view; hidden transfer records cannot enter the model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .core import Visibility, canonical_digest, canonical_json
from .distillation import (
    AUTHORITY_DIAGNOSTIC_ONLY,
    DistillationError,
    StudyDataAccess,
    _reject_authority,
)
from .snapshots import DiagnosticSnapshot, SnapshotStore, SnapshotError, TransitionView, decode_snapshot


@dataclass(frozen=True, slots=True)
class LearnedProviderObservation:
    provider_id: str
    model_identity: str
    snapshot_identity: str
    score: float
    abstained: bool
    out_of_distribution: bool
    calibration_identity: str
    observation_identity: str = ""
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        if not self.provider_id or not self.model_identity or not self.snapshot_identity:
            raise DistillationError("provider observations require identities")
        if not 0.0 <= self.score <= 1.0:
            raise DistillationError("provider score must be within [0, 1]")
        if not self.calibration_identity or self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise DistillationError("provider observations must remain diagnostic-only")
        if self.observation_identity and self.observation_identity != self.content_identity:
            raise DistillationError("provider observation identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-learned-provider-observation/0.4",
            "record_type": "learned-provider-observation",
            "provider_id": self.provider_id,
            "model_identity": self.model_identity,
            "snapshot_identity": self.snapshot_identity,
            "score": self.score,
            "abstained": self.abstained,
            "out_of_distribution": self.out_of_distribution,
            "calibration_identity": self.calibration_identity,
            "authority": self.authority,
            "semantics": "learned-diagnostic-observation; not-a-verdict",
        }
        if include_identity:
            value["observation_identity"] = self.observation_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class TransitionFrequencyModel:
    training_dataset_identity: str
    training_record_ids: tuple[str, ...]
    feature_extractor_identity: str
    training_code_identity: str
    calibration_identity: str
    transition_counts: dict[str, int]
    total_count: int
    model_identity: str = ""
    artifact_identity: str = ""
    provider_id: str = "mnel-reference-transition-frequency/0.4"
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        if not self.provider_id or not self.training_record_ids or self.total_count < 1:
            raise DistillationError("trained provider requires non-empty development data")
        if sum(self.transition_counts.values()) != self.total_count:
            raise DistillationError("transition counts do not match total count")
        if any(not isinstance(key, str) or not isinstance(value, int) or value < 1 for key, value in self.transition_counts.items()):
            raise DistillationError("transition counts are malformed")
        if not all((self.training_dataset_identity, self.feature_extractor_identity, self.training_code_identity, self.calibration_identity)):
            raise DistillationError("provider training identities are required")
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise DistillationError("learned providers are diagnostic-only")
        if self.model_identity and self.model_identity != self.content_identity:
            raise DistillationError("provider model identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    @property
    def model_size_bytes(self) -> int:
        return len(canonical_json(self.to_dict()))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-learned-provider-artifact/0.4",
            "provider_id": self.provider_id,
            "training_dataset_identity": self.training_dataset_identity,
            "training_record_ids": list(self.training_record_ids),
            "feature_extractor_identity": self.feature_extractor_identity,
            "training_code_identity": self.training_code_identity,
            "calibration_identity": self.calibration_identity,
            "transition_counts": dict(sorted(self.transition_counts.items())),
            "total_count": self.total_count,
            "authority": self.authority,
            "semantics": "learned-provider-artifact; diagnostic-only; not-a-verdict",
        }
        if include_identity:
            value["model_identity"] = self.model_identity or self.content_identity
        return value

    def serialize(self) -> bytes:
        value = self.to_dict()
        value["artifact_identity"] = "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()
        return canonical_json(value)

    @classmethod
    def load(cls, payload: bytes) -> "TransitionFrequencyModel":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise DistillationError("provider artifact is not valid JSON") from error
        if not isinstance(value, dict) or value.get("schema") != "mnel-learned-provider-artifact/0.4":
            raise DistillationError("unsupported provider artifact schema")
        _reject_authority(value)
        supplied_artifact = value.pop("artifact_identity", None)
        if not isinstance(supplied_artifact, str):
            raise DistillationError("provider artifact identity is missing")
        canonical_payload = canonical_json(value)
        expected_artifact = "sha256:" + hashlib.sha256(canonical_payload).hexdigest()
        if supplied_artifact != expected_artifact:
            raise DistillationError("provider artifact bytes do not match artifact identity")
        model = cls(
            training_dataset_identity=value.get("training_dataset_identity"),
            training_record_ids=tuple(value.get("training_record_ids", ())),
            feature_extractor_identity=value.get("feature_extractor_identity"),
            training_code_identity=value.get("training_code_identity"),
            calibration_identity=value.get("calibration_identity"),
            transition_counts=dict(value.get("transition_counts", {})),
            total_count=value.get("total_count"),
            model_identity=value.get("model_identity", ""),
            provider_id=value.get("provider_id", ""),
            authority=value.get("authority", ""),
        )
        if model.model_identity != model.content_identity:
            raise DistillationError("provider artifact model identity is invalid")
        object.__setattr__(model, "artifact_identity", supplied_artifact)
        return model

    def infer(self, snapshot: DiagnosticSnapshot) -> LearnedProviderObservation:
        try:
            view = decode_snapshot(snapshot)
        except SnapshotError as error:
            raise DistillationError("provider cannot decode snapshot") from error
        if not isinstance(view, TransitionView):
            raise DistillationError("transition-frequency provider requires a transition snapshot")
        key = _transition_key(view)
        count = self.transition_counts.get(key, 0)
        if count == 0:
            return LearnedProviderObservation(
                self.provider_id,
                self.model_identity or self.content_identity,
                snapshot.snapshot_identity,
                0.0,
                True,
                True,
                self.calibration_identity,
            )
        frequency = count / self.total_count
        return LearnedProviderObservation(
            self.provider_id,
            self.model_identity or self.content_identity,
            snapshot.snapshot_identity,
            max(0.0, min(1.0, frequency)),
            False,
            False,
            self.calibration_identity,
        )


def _transition_key(view: TransitionView) -> str:
    return hashlib.sha256(view.previous_state + b"\x00" + view.next_state).hexdigest()


def train_transition_frequency(
    access: StudyDataAccess,
    snapshots: SnapshotStore,
    *,
    record_type: str = "experience-episode",
    feature_extractor_identity: str = "mnel-transition-bytes/0.4",
    training_code_identity: str = "mnel-reference-frequency-training/0.4",
    calibration_identity: str = "mnel-reference-calibration/0.4",
) -> TransitionFrequencyModel:
    if access.purpose != "development-study" or any(
        item.visibility not in {Visibility.DEVELOPMENT, Visibility.SELECTION_OBSERVED}
        for item in access.records()
    ):
        raise DistillationError("provider training requires a development-only access view")
    counts: dict[str, int] = {}
    record_ids: list[str] = []
    for record in access.records(record_type):
        snapshot_identity = record.payload.get("snapshot_identity")
        if not isinstance(snapshot_identity, str):
            raise DistillationError("training record lacks a snapshot identity")
        snapshot = snapshots.get(snapshot_identity)
        view = decode_snapshot(snapshot)
        if not isinstance(view, TransitionView):
            raise DistillationError("training dataset contains a non-transition snapshot")
        key = _transition_key(view)
        counts[key] = counts.get(key, 0) + 1
        record_ids.append(record.identity)
    if not record_ids:
        raise DistillationError("training dataset contains no eligible records")
    draft = TransitionFrequencyModel(
        access.dataset_identity,
        tuple(sorted(record_ids)),
        feature_extractor_identity,
        training_code_identity,
        calibration_identity,
        counts,
        len(record_ids),
    )
    model = TransitionFrequencyModel(
        draft.training_dataset_identity,
        draft.training_record_ids,
        draft.feature_extractor_identity,
        draft.training_code_identity,
        draft.calibration_identity,
        draft.transition_counts,
        draft.total_count,
        draft.content_identity,
    )
    artifact_payload = canonical_json(model.to_dict())
    object.__setattr__(model, "artifact_identity", "sha256:" + hashlib.sha256(artifact_payload).hexdigest())
    return model
