"""Compact identity-bound diagnostic snapshot producers.

Snapshots are immutable transport objects, not verifier results. Their content identity
includes source, dependency, extractor, schema, and payload identities so reuse is
invalidated when any material producer dependency changes.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Sequence

from .core import canonical_digest


class SnapshotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DiagnosticSnapshot:
    snapshot_type: str
    schema_version: int
    producer_identity: str
    source_identity: str
    dependency_identity: str
    feature_extractor_identity: str
    payload: bytes
    payload_identity: str
    snapshot_identity: str

    @classmethod
    def build(
        cls,
        *,
        snapshot_type: str,
        schema_version: int,
        producer_identity: str,
        source_identity: str,
        dependency_identity: str,
        feature_extractor_identity: str,
        payload: bytes,
    ) -> "DiagnosticSnapshot":
        if not snapshot_type.strip() or schema_version < 1:
            raise SnapshotError("snapshot type and positive schema version are required")
        identities = (
            producer_identity,
            source_identity,
            dependency_identity,
            feature_extractor_identity,
        )
        if any(not identity.strip() for identity in identities):
            raise SnapshotError("snapshot producer and dependency identities are required")
        if not payload or len(payload) > 1024 * 1024:
            raise SnapshotError("snapshot payload must be non-empty and bounded")
        payload_identity = "sha256:" + hashlib.sha256(payload).hexdigest()
        identity_body = {
            "snapshot_type": snapshot_type,
            "schema_version": schema_version,
            "producer_identity": producer_identity,
            "source_identity": source_identity,
            "dependency_identity": dependency_identity,
            "feature_extractor_identity": feature_extractor_identity,
            "payload_identity": payload_identity,
        }
        return cls(
            snapshot_type,
            schema_version,
            producer_identity,
            source_identity,
            dependency_identity,
            feature_extractor_identity,
            bytes(payload),
            payload_identity,
            canonical_digest(identity_body),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "mnel-diagnostic-snapshot/0.3",
            "snapshot_type": self.snapshot_type,
            "schema_version": self.schema_version,
            "producer_identity": self.producer_identity,
            "source_identity": self.source_identity,
            "dependency_identity": self.dependency_identity,
            "feature_extractor_identity": self.feature_extractor_identity,
            "payload_identity": self.payload_identity,
            "payload_bytes": len(self.payload),
            "snapshot_identity": self.snapshot_identity,
            "authority": "diagnostic-only",
            "semantics": "not-a-verdict",
        }


def transition_snapshot(
    previous_state: bytes,
    next_state: bytes,
    *,
    producer_identity: str,
    source_identity: str,
    dependency_identity: str,
    feature_extractor_identity: str,
    schema_version: int = 1,
) -> DiagnosticSnapshot:
    return DiagnosticSnapshot.build(
        snapshot_type="transition",
        schema_version=schema_version,
        producer_identity=producer_identity,
        source_identity=source_identity,
        dependency_identity=dependency_identity,
        feature_extractor_identity=feature_extractor_identity,
        payload=_pair_payload(b"MNEL-T1", previous_state, next_state),
    )


def pair_snapshot(
    left: bytes,
    right: bytes,
    *,
    producer_identity: str,
    source_identity: str,
    dependency_identity: str,
    feature_extractor_identity: str,
    schema_version: int = 1,
) -> DiagnosticSnapshot:
    return DiagnosticSnapshot.build(
        snapshot_type="pair",
        schema_version=schema_version,
        producer_identity=producer_identity,
        source_identity=source_identity,
        dependency_identity=dependency_identity,
        feature_extractor_identity=feature_extractor_identity,
        payload=_pair_payload(b"MNEL-P1", left, right),
    )


def tabular_snapshot(
    rows: Sequence[Sequence[float]],
    *,
    producer_identity: str,
    source_identity: str,
    dependency_identity: str,
    feature_extractor_identity: str,
    schema_version: int = 1,
) -> DiagnosticSnapshot:
    if not rows or not rows[0]:
        raise SnapshotError("tabular snapshot requires non-empty rows and columns")
    column_count = len(rows[0])
    if len(rows) > 65535 or column_count > 65535 or any(len(row) != column_count for row in rows):
        raise SnapshotError("tabular shape is invalid or exceeds the bounded format")
    values: list[float] = []
    for row in rows:
        for value in row:
            if not math.isfinite(value):
                raise SnapshotError("tabular values must be finite")
            values.append(float(value))
    payload = struct.pack(">4sHH", b"MNET", len(rows), column_count) + struct.pack(
        f">{len(values)}d", *values
    )
    return DiagnosticSnapshot.build(
        snapshot_type="tabular",
        schema_version=schema_version,
        producer_identity=producer_identity,
        source_identity=source_identity,
        dependency_identity=dependency_identity,
        feature_extractor_identity=feature_extractor_identity,
        payload=payload,
    )


def _pair_payload(header: bytes, left: bytes, right: bytes) -> bytes:
    if not left or not right or len(left) > 65535 or len(right) > 65535:
        raise SnapshotError("pair members must be non-empty and bounded")
    return header + struct.pack(">HH", len(left), len(right)) + left + right
