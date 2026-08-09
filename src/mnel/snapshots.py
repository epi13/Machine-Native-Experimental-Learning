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
from typing import Sequence, TypeAlias

from .core import canonical_digest


class SnapshotError(ValueError):
    pass


MAX_SNAPSHOT_BYTES = 1024 * 1024
MAX_TRANSITION_MEMBER_BYTES = 65535
MAX_TRACE_EVENTS = 256
MAX_TRACE_LABEL_BYTES = 64
MAX_TRACE_PAYLOAD_BYTES = 4096
MAX_GRAPH_NODES = 4096
MAX_GRAPH_EDGES = 8192
MAX_GRAPH_LABEL_BYTES = 64
MAX_COMPOSITE_COMPONENTS = 64


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
    schema_identity: str = "mnel-diagnostic-snapshot"

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
        schema_identity: str = "mnel-diagnostic-snapshot",
    ) -> "DiagnosticSnapshot":
        if (
            not isinstance(snapshot_type, str)
            or not snapshot_type.strip()
            or not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version < 1
            or not isinstance(schema_identity, str)
            or not schema_identity.strip()
        ):
            raise SnapshotError("snapshot type, schema identity, and positive schema version are required")
        identities = (
            producer_identity,
            source_identity,
            dependency_identity,
            feature_extractor_identity,
        )
        if any(not isinstance(identity, str) or not identity.strip() for identity in identities):
            raise SnapshotError("snapshot producer and dependency identities are required")
        if not payload or len(payload) > MAX_SNAPSHOT_BYTES:
            raise SnapshotError("snapshot payload must be non-empty and bounded")
        payload_identity = "sha256:" + hashlib.sha256(payload).hexdigest()
        identity_body = {
            "snapshot_type": snapshot_type,
            "schema_version": schema_version,
            "schema_identity": schema_identity,
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
            schema_identity,
        )

    def validate_integrity(self) -> None:
        """Reject tampered metadata or payload before a consumer receives a view."""

        rebuilt = DiagnosticSnapshot.build(
            snapshot_type=self.snapshot_type,
            schema_version=self.schema_version,
            schema_identity=self.schema_identity,
            producer_identity=self.producer_identity,
            source_identity=self.source_identity,
            dependency_identity=self.dependency_identity,
            feature_extractor_identity=self.feature_extractor_identity,
            payload=self.payload,
        )
        if rebuilt.payload_identity != self.payload_identity or rebuilt.snapshot_identity != self.snapshot_identity:
            raise SnapshotError("snapshot identity or payload identity does not match its content")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "mnel-diagnostic-snapshot/0.3",
            "snapshot_type": self.snapshot_type,
            "schema_version": self.schema_version,
            "schema_identity": self.schema_identity,
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
    schema_identity: str = "mnel-diagnostic-snapshot",
) -> DiagnosticSnapshot:
    return DiagnosticSnapshot.build(
        snapshot_type="transition",
        schema_version=schema_version,
        producer_identity=producer_identity,
        source_identity=source_identity,
        dependency_identity=dependency_identity,
        feature_extractor_identity=feature_extractor_identity,
        payload=_pair_payload(b"MNEL-T1", previous_state, next_state),
        schema_identity=schema_identity,
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
    schema_identity: str = "mnel-diagnostic-snapshot",
) -> DiagnosticSnapshot:
    return DiagnosticSnapshot.build(
        snapshot_type="pair",
        schema_version=schema_version,
        producer_identity=producer_identity,
        source_identity=source_identity,
        dependency_identity=dependency_identity,
        feature_extractor_identity=feature_extractor_identity,
        payload=_pair_payload(b"MNEL-P1", left, right),
        schema_identity=schema_identity,
    )


def tabular_snapshot(
    rows: Sequence[Sequence[float]],
    *,
    producer_identity: str,
    source_identity: str,
    dependency_identity: str,
    feature_extractor_identity: str,
    schema_version: int = 1,
    schema_identity: str = "mnel-diagnostic-snapshot",
) -> DiagnosticSnapshot:
    if not rows or not rows[0]:
        raise SnapshotError("tabular snapshot requires non-empty rows and columns")
    if any(not isinstance(row, Sequence) for row in rows):
        raise SnapshotError("tabular rows must be sequences")
    column_count = len(rows[0])
    if len(rows) > 65535 or column_count > 65535 or any(len(row) != column_count for row in rows):
        raise SnapshotError("tabular shape is invalid or exceeds the bounded format")
    values: list[float] = []
    for row in rows:
        for value in row:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
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
        schema_identity=schema_identity,
    )


def _pair_payload(header: bytes, left: bytes, right: bytes) -> bytes:
    if (
        not isinstance(left, bytes)
        or not isinstance(right, bytes)
        or not left
        or not right
        or len(left) > MAX_TRANSITION_MEMBER_BYTES
        or len(right) > MAX_TRANSITION_MEMBER_BYTES
    ):
        raise SnapshotError("pair members must be non-empty and bounded")
    return header + struct.pack(">HH", len(left), len(right)) + left + right


@dataclass(frozen=True, slots=True)
class TransitionView:
    previous_state: bytes
    next_state: bytes


@dataclass(frozen=True, slots=True)
class PairView:
    left: bytes
    right: bytes


@dataclass(frozen=True, slots=True)
class TabularView:
    rows: tuple[tuple[float, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.rows[0]) if self.rows else 0


@dataclass(frozen=True, slots=True)
class TraceEvent:
    event_type: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class TraceView:
    events: tuple[TraceEvent, ...]


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: int
    label: str


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: int
    target: int
    edge_type: str


@dataclass(frozen=True, slots=True)
class GraphView:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


@dataclass(frozen=True, slots=True)
class CompositeView:
    component_snapshot_identities: tuple[str, ...]


SnapshotView: TypeAlias = TransitionView | PairView | TabularView | TraceView | GraphView | CompositeView


def decode_transition(payload: bytes) -> TransitionView:
    left, right = _decode_pair_payload(payload, b"MNEL-T1")
    return TransitionView(left, right)


def decode_pair(payload: bytes) -> PairView:
    left, right = _decode_pair_payload(payload, b"MNEL-P1")
    return PairView(left, right)


def decode_tabular(payload: bytes) -> TabularView:
    if len(payload) > MAX_SNAPSHOT_BYTES or len(payload) < 8 or payload[:4] != b"MNET":
        raise SnapshotError("invalid tabular magic or truncated header")
    rows, columns = struct.unpack(">HH", payload[4:8])
    if rows == 0 or columns == 0 or rows > 65535 or columns > 65535:
        raise SnapshotError("invalid tabular dimensions")
    expected = 8 + rows * columns * 8
    if len(payload) != expected:
        raise SnapshotError("tabular payload is truncated or has trailing bytes")
    values = struct.unpack(f">{rows * columns}d", payload[8:])
    if any(not math.isfinite(value) for value in values):
        raise SnapshotError("tabular payload contains a non-finite value")
    return TabularView(tuple(tuple(values[row * columns : (row + 1) * columns]) for row in range(rows)))


def decode_trace(payload: bytes) -> TraceView:
    if len(payload) > MAX_SNAPSHOT_BYTES or len(payload) < 9 or payload[:7] != b"MNEL-R1":
        raise SnapshotError("invalid trace magic or truncated header")
    count = struct.unpack(">H", payload[7:9])[0]
    if count == 0 or count > MAX_TRACE_EVENTS:
        raise SnapshotError("invalid trace event count")
    offset = 9
    events: list[TraceEvent] = []
    for _ in range(count):
        if offset + 3 > len(payload):
            raise SnapshotError("truncated trace event header")
        label_length, body_length = struct.unpack(">BH", payload[offset : offset + 3])
        offset += 3
        if label_length == 0 or label_length > MAX_TRACE_LABEL_BYTES or body_length > MAX_TRACE_PAYLOAD_BYTES:
            raise SnapshotError("trace event exceeds its label or payload ceiling")
        end = offset + label_length + body_length
        if end > len(payload):
            raise SnapshotError("truncated trace event payload")
        try:
            event_type = payload[offset : offset + label_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise SnapshotError("trace event type is not UTF-8") from error
        offset += label_length
        events.append(TraceEvent(event_type, bytes(payload[offset : offset + body_length])))
        offset = end
    if offset != len(payload):
        raise SnapshotError("trace payload has trailing bytes")
    return TraceView(tuple(events))


def decode_graph(payload: bytes) -> GraphView:
    if len(payload) > MAX_SNAPSHOT_BYTES or len(payload) < 11 or payload[:7] != b"MNEL-G1":
        raise SnapshotError("invalid graph magic or truncated header")
    node_count, edge_count = struct.unpack(">HH", payload[7:11])
    if node_count == 0 or node_count > MAX_GRAPH_NODES or edge_count > MAX_GRAPH_EDGES:
        raise SnapshotError("invalid graph dimensions")
    offset = 11
    nodes: list[GraphNode] = []
    seen_nodes: set[int] = set()
    for _ in range(node_count):
        if offset + 5 > len(payload):
            raise SnapshotError("truncated graph node")
        node_id, label_length = struct.unpack(">IB", payload[offset : offset + 5])
        offset += 5
        if label_length == 0 or label_length > MAX_GRAPH_LABEL_BYTES or node_id in seen_nodes:
            raise SnapshotError("invalid or duplicate graph node")
        if offset + label_length > len(payload):
            raise SnapshotError("truncated graph node label")
        try:
            label = payload[offset : offset + label_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise SnapshotError("graph node label is not UTF-8") from error
        offset += label_length
        seen_nodes.add(node_id)
        nodes.append(GraphNode(node_id, label))
    edges: list[GraphEdge] = []
    seen_edges: set[tuple[int, int, str]] = set()
    for _ in range(edge_count):
        if offset + 9 > len(payload):
            raise SnapshotError("truncated graph edge")
        source, target, label_length = struct.unpack(">IIB", payload[offset : offset + 9])
        offset += 9
        if source not in seen_nodes or target not in seen_nodes or label_length == 0 or label_length > MAX_GRAPH_LABEL_BYTES:
            raise SnapshotError("graph edge references an unknown node or invalid type")
        if offset + label_length > len(payload):
            raise SnapshotError("truncated graph edge type")
        try:
            edge_type = payload[offset : offset + label_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise SnapshotError("graph edge type is not UTF-8") from error
        offset += label_length
        edge = (source, target, edge_type)
        if edge in seen_edges:
            raise SnapshotError("duplicate graph edge")
        seen_edges.add(edge)
        edges.append(GraphEdge(*edge))
    if offset != len(payload):
        raise SnapshotError("graph payload has trailing bytes")
    return GraphView(tuple(nodes), tuple(edges))


def decode_composite(payload: bytes) -> CompositeView:
    if len(payload) > MAX_SNAPSHOT_BYTES or len(payload) < 9 or payload[:7] != b"MNEL-C1":
        raise SnapshotError("invalid composite magic or truncated header")
    count = struct.unpack(">H", payload[7:9])[0]
    if count == 0 or count > MAX_COMPOSITE_COMPONENTS:
        raise SnapshotError("invalid composite component count")
    offset = 9
    identities: list[str] = []
    for _ in range(count):
        if offset + 2 > len(payload):
            raise SnapshotError("truncated composite component")
        length = struct.unpack(">H", payload[offset : offset + 2])[0]
        offset += 2
        if length == 0 or length > 128 or offset + length > len(payload):
            raise SnapshotError("invalid composite component identity")
        try:
            identity = payload[offset : offset + length].decode("ascii")
        except UnicodeDecodeError as error:
            raise SnapshotError("composite component identity is not ASCII") from error
        if (
            len(identity) != 71
            or not identity.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in identity[7:])
            or identity in identities
        ):
            raise SnapshotError("composite component identity is invalid or duplicated")
        identities.append(identity)
        offset += length
    if offset != len(payload):
        raise SnapshotError("composite payload has trailing bytes")
    return CompositeView(tuple(identities))


def decode_snapshot(snapshot: DiagnosticSnapshot) -> SnapshotView:
    snapshot.validate_integrity()
    decoders = {
        "transition": decode_transition,
        "pair": decode_pair,
        "tabular": decode_tabular,
        "trace": decode_trace,
        "graph": decode_graph,
        "composite": decode_composite,
    }
    try:
        decoder = decoders[snapshot.snapshot_type]
    except KeyError as error:
        raise SnapshotError(f"unsupported snapshot type: {snapshot.snapshot_type}") from error
    return decoder(snapshot.payload)


class SnapshotStore:
    """Identity-keyed immutable snapshot store shared by probes and providers."""

    def __init__(self) -> None:
        self._snapshots: dict[str, DiagnosticSnapshot] = {}

    def register(self, snapshot: DiagnosticSnapshot) -> str:
        snapshot.validate_integrity()
        view = decode_snapshot(snapshot)
        if isinstance(view, CompositeView):
            for identity in view.component_snapshot_identities:
                if identity not in self._snapshots:
                    raise SnapshotError("composite references an unregistered snapshot")
        existing = self._snapshots.get(snapshot.snapshot_identity)
        if existing is not None and existing != snapshot:
            raise SnapshotError("snapshot identity collision")
        self._snapshots[snapshot.snapshot_identity] = snapshot
        return snapshot.snapshot_identity

    def get(self, identity: str) -> DiagnosticSnapshot:
        try:
            return self._snapshots[identity]
        except KeyError as error:
            raise SnapshotError(f"unknown snapshot identity: {identity}") from error

    def view(self, identity: str, *, accepted_types: Sequence[str] = (), schema_versions: Sequence[int] = ()) -> SnapshotView:
        snapshot = self.get(identity)
        if accepted_types and snapshot.snapshot_type not in accepted_types:
            raise SnapshotError("snapshot type is incompatible with the requested consumer")
        if schema_versions and snapshot.schema_version not in schema_versions:
            raise SnapshotError("snapshot schema version is incompatible with the requested consumer")
        return decode_snapshot(snapshot)

    def identities(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshots))


def trace_snapshot(
    events: Sequence[tuple[str, bytes]],
    *,
    producer_identity: str,
    source_identity: str,
    dependency_identity: str,
    feature_extractor_identity: str,
    schema_version: int = 1,
    schema_identity: str = "mnel-diagnostic-snapshot",
) -> DiagnosticSnapshot:
    if not events or len(events) > MAX_TRACE_EVENTS:
        raise SnapshotError("trace requires a bounded non-empty event sequence")
    encoded = bytearray(b"MNEL-R1" + struct.pack(">H", len(events)))
    for event_type, payload in events:
        if not isinstance(event_type, str) or not isinstance(payload, bytes):
            raise SnapshotError("trace events require a string type and byte payload")
        label = event_type.encode("utf-8")
        if not label or len(label) > MAX_TRACE_LABEL_BYTES or len(payload) > MAX_TRACE_PAYLOAD_BYTES:
            raise SnapshotError("trace event exceeds its label or payload ceiling")
        encoded.extend(struct.pack(">BH", len(label), len(payload)))
        encoded.extend(label)
        encoded.extend(payload)
    return DiagnosticSnapshot.build(
        snapshot_type="trace",
        schema_version=schema_version,
        producer_identity=producer_identity,
        source_identity=source_identity,
        dependency_identity=dependency_identity,
        feature_extractor_identity=feature_extractor_identity,
        payload=bytes(encoded),
        schema_identity=schema_identity,
    )


def graph_snapshot(
    nodes: Sequence[tuple[int, str]],
    edges: Sequence[tuple[int, int, str]],
    *,
    producer_identity: str,
    source_identity: str,
    dependency_identity: str,
    feature_extractor_identity: str,
    schema_version: int = 1,
    schema_identity: str = "mnel-diagnostic-snapshot",
) -> DiagnosticSnapshot:
    if not nodes or len(nodes) > MAX_GRAPH_NODES or len(edges) > MAX_GRAPH_EDGES:
        raise SnapshotError("graph exceeds its node or edge ceiling")
    if any(
        not isinstance(item, (tuple, list)) or len(item) != 2 for item in nodes
    ) or any(not isinstance(item, (tuple, list)) or len(item) != 3 for item in edges):
        raise SnapshotError("graph nodes and edges have invalid shapes")
    normalized_nodes = list(nodes)
    if any(
        not isinstance(node_id, int)
        or isinstance(node_id, bool)
        or node_id < 0
        or node_id > 0xFFFFFFFF
        or not isinstance(label, str)
        or not label
        for node_id, label in normalized_nodes
    ):
        raise SnapshotError("graph nodes require non-negative ids and labels")
    if len({node_id for node_id, _ in normalized_nodes}) != len(normalized_nodes):
        raise SnapshotError("graph node identities must be unique")
    node_ids = {node_id for node_id, _ in normalized_nodes}
    normalized_edges = list(edges)
    if any(
        not isinstance(source, int)
        or not isinstance(target, int)
        or isinstance(source, bool)
        or isinstance(target, bool)
        or source < 0
        or target < 0
        or source > 0xFFFFFFFF
        or target > 0xFFFFFFFF
        or source not in node_ids
        or target not in node_ids
        or not isinstance(edge_type, str)
        or not edge_type
        for source, target, edge_type in normalized_edges
    ):
        raise SnapshotError("graph edges must reference declared nodes")
    if len(set(normalized_edges)) != len(normalized_edges):
        raise SnapshotError("graph edges must be unique")
    normalized_nodes.sort(key=lambda item: item[0])
    normalized_edges.sort(key=lambda item: (item[0], item[1], item[2]))
    encoded = bytearray(b"MNEL-G1" + struct.pack(">HH", len(normalized_nodes), len(normalized_edges)))
    for node_id, label in normalized_nodes:
        label_bytes = label.encode("utf-8")
        if len(label_bytes) > MAX_GRAPH_LABEL_BYTES:
            raise SnapshotError("graph node label exceeds its ceiling")
        encoded.extend(struct.pack(">IB", node_id, len(label_bytes)))
        encoded.extend(label_bytes)
    for source, target, edge_type in normalized_edges:
        label_bytes = edge_type.encode("utf-8")
        if len(label_bytes) > MAX_GRAPH_LABEL_BYTES:
            raise SnapshotError("graph edge type exceeds its ceiling")
        encoded.extend(struct.pack(">IIB", source, target, len(label_bytes)))
        encoded.extend(label_bytes)
    return DiagnosticSnapshot.build(
        snapshot_type="graph",
        schema_version=schema_version,
        producer_identity=producer_identity,
        source_identity=source_identity,
        dependency_identity=dependency_identity,
        feature_extractor_identity=feature_extractor_identity,
        payload=bytes(encoded),
        schema_identity=schema_identity,
    )


def composite_snapshot(
    components: Sequence[DiagnosticSnapshot],
    *,
    producer_identity: str,
    source_identity: str,
    dependency_identity: str,
    feature_extractor_identity: str,
    schema_version: int = 1,
    schema_identity: str = "mnel-diagnostic-snapshot",
) -> DiagnosticSnapshot:
    if not components or len(components) > MAX_COMPOSITE_COMPONENTS:
        raise SnapshotError("composite requires a bounded non-empty component set")
    if any(not isinstance(component, DiagnosticSnapshot) for component in components):
        raise SnapshotError("composite components must be diagnostic snapshots")
    for component in components:
        component.validate_integrity()
    identities = [component.snapshot_identity for component in components]
    if len(set(identities)) != len(identities):
        raise SnapshotError("composite component identities must be unique")
    encoded = bytearray(b"MNEL-C1" + struct.pack(">H", len(identities)))
    for identity in identities:
        value = identity.encode("ascii")
        if len(value) > 128:
            raise SnapshotError("composite component identity exceeds its ceiling")
        encoded.extend(struct.pack(">H", len(value)))
        encoded.extend(value)
    return DiagnosticSnapshot.build(
        snapshot_type="composite",
        schema_version=schema_version,
        producer_identity=producer_identity,
        source_identity=source_identity,
        dependency_identity=dependency_identity,
        feature_extractor_identity=feature_extractor_identity,
        payload=bytes(encoded),
        schema_identity=schema_identity,
    )


def _decode_pair_payload(payload: bytes, magic: bytes) -> tuple[bytes, bytes]:
    if len(payload) > MAX_SNAPSHOT_BYTES or len(payload) < 11 or payload[:7] != magic:
        raise SnapshotError("invalid pair magic or truncated header")
    left_length, right_length = struct.unpack(">HH", payload[7:11])
    if left_length == 0 or right_length == 0:
        raise SnapshotError("pair members must be non-empty")
    end = 11 + left_length + right_length
    if end != len(payload):
        raise SnapshotError("pair payload is truncated or has trailing bytes")
    return bytes(payload[11 : 11 + left_length]), bytes(payload[11 + left_length : end])
