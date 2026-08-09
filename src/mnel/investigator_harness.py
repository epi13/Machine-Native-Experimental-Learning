"""Bounded local investigator context, workspace, and quarantine contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from .core import Visibility, canonical_digest, canonical_json


class WorkspaceAccess(StrEnum):
    READ_ONLY = "read-only"
    PROPOSAL = "proposal"


class ContextPackingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InvestigatorWorkspace:
    path: str
    access: WorkspaceAccess

    @classmethod
    def read_only(cls, path: str | Path) -> "InvestigatorWorkspace":
        return cls(str(Path(path).resolve()), WorkspaceAccess.READ_ONLY)

    @classmethod
    def proposal(cls, path: str | Path) -> "InvestigatorWorkspace":
        return cls(str(Path(path).resolve()), WorkspaceAccess.PROPOSAL)

    def assert_write_allowed(self) -> None:
        if self.access is WorkspaceAccess.READ_ONLY:
            raise PermissionError("read-only investigator workspace cannot be mutated")


@dataclass(frozen=True, slots=True)
class RuntimeIdentityEnvelope:
    model_identity: str
    quantization_identity: str
    runtime_identity: str
    prompt_identity: str
    tool_schema_identity: str

    def __post_init__(self) -> None:
        if not all(
            item.strip()
            for item in (
                self.model_identity,
                self.quantization_identity,
                self.runtime_identity,
                self.prompt_identity,
                self.tool_schema_identity,
            )
        ):
            raise ValueError("model, quantization, runtime, prompt, and tool identities are required")

    def to_dict(self) -> dict[str, str]:
        return {
            "model_identity": self.model_identity,
            "quantization_identity": self.quantization_identity,
            "runtime_identity": self.runtime_identity,
            "prompt_identity": self.prompt_identity,
            "tool_schema_identity": self.tool_schema_identity,
        }


@dataclass(frozen=True, slots=True)
class PackedContext:
    snapshot_identity: str
    record_ids: tuple[str, ...]
    records: tuple[dict[str, Any], ...]
    encoded_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnel-eligible-context/0.2",
            "snapshot_identity": self.snapshot_identity,
            "record_ids": list(self.record_ids),
            "records": list(self.records),
            "encoded_bytes": self.encoded_bytes,
            "authority": "proposal-only",
        }


def pack_eligible_context(
    records: Iterable[dict[str, Any]],
    *,
    max_records: int = 32,
    max_bytes: int = 256 * 1024,
) -> PackedContext:
    """Pack visible records in stable identity order under an explicit byte ceiling."""

    if max_records < 1 or max_bytes < 1:
        raise ContextPackingError("context limits must be positive")
    normalized: list[dict[str, Any]] = []
    for record in records:
        record_id = record.get("record_id") or record.get("observation_id") or record.get("id")
        visibility = record.get("visibility", Visibility.DEVELOPMENT.value)
        if not isinstance(record_id, str) or not record_id.strip():
            raise ContextPackingError("every context record requires an identity")
        if visibility in {Visibility.TRANSFER_HIDDEN.value, Visibility.FUTURE_FINAL.value}:
            raise ContextPackingError(f"hidden record cannot enter investigator context: {record_id}")
        normalized.append(dict(record))
    normalized.sort(key=lambda record: str(record.get("record_id") or record.get("observation_id") or record["id"]))
    selected = normalized[:max_records]
    encoded = canonical_json(selected)
    if len(encoded) > max_bytes:
        raise ContextPackingError("eligible context exceeds its explicit byte ceiling")
    record_ids = tuple(
        str(record.get("record_id") or record.get("observation_id") or record["id"])
        for record in selected
    )
    return PackedContext(
        snapshot_identity=canonical_digest(
            {"record_ids": record_ids, "encoded": encoded.decode("utf-8")}
        ),
        record_ids=record_ids,
        records=tuple(selected),
        encoded_bytes=len(encoded),
    )


@dataclass(frozen=True, slots=True)
class CandidateTransaction:
    transaction_identity: str
    parent_candidate_id: str
    context_snapshot_identity: str
    workspace: str
    access: WorkspaceAccess = WorkspaceAccess.PROPOSAL
    proposal_only: bool = True

    @classmethod
    def create(
        cls,
        *,
        parent_candidate_id: str,
        context_snapshot_identity: str,
        workspace: str | Path,
    ) -> "CandidateTransaction":
        if not parent_candidate_id.strip() or not context_snapshot_identity.strip():
            raise ValueError("candidate transactions require parent and context identities")
        value = {
            "parent_candidate_id": parent_candidate_id,
            "context_snapshot_identity": context_snapshot_identity,
            "workspace": str(Path(workspace).resolve()),
            "access": WorkspaceAccess.PROPOSAL.value,
            "proposal_only": True,
        }
        return cls(
            transaction_identity=canonical_digest(value),
            parent_candidate_id=parent_candidate_id,
            context_snapshot_identity=context_snapshot_identity,
            workspace=value["workspace"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnel-candidate-transaction/0.2",
            "transaction_identity": self.transaction_identity,
            "parent_candidate_id": self.parent_candidate_id,
            "context_snapshot_identity": self.context_snapshot_identity,
            "workspace": self.workspace,
            "access": self.access.value,
            "proposal_only": self.proposal_only,
        }


@dataclass(frozen=True, slots=True)
class QuarantineEntry:
    provider_id: str
    reason: str
    evidence_identity: str


class QuarantineQueue:
    """Explicit queue for failures; enqueueing never changes evaluator state."""

    def __init__(self) -> None:
        self._entries: list[QuarantineEntry] = []

    def enqueue(self, entry: QuarantineEntry) -> None:
        if not entry.provider_id.strip() or not entry.reason.strip() or not entry.evidence_identity.strip():
            raise ValueError("quarantine entries require provider, reason, and evidence identities")
        self._entries.append(entry)

    def list(self) -> tuple[QuarantineEntry, ...]:
        return tuple(sorted(self._entries, key=lambda entry: (entry.provider_id, entry.evidence_identity)))


@dataclass(frozen=True, slots=True)
class MorningReport:
    context_snapshot_identity: str
    packed_record_count: int
    provider_observation_count: int
    quarantined_provider_count: int
    proposal_count: int
    authority: str = "proposal-only"

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": "mnel-morning-report/0.2",
            "context_snapshot_identity": self.context_snapshot_identity,
            "packed_record_count": self.packed_record_count,
            "provider_observation_count": self.provider_observation_count,
            "quarantined_provider_count": self.quarantined_provider_count,
            "proposal_count": self.proposal_count,
            "authority": self.authority,
        }
        value["report_identity"] = canonical_digest(value)
        return value
