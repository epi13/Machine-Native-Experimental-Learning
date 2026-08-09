"""Bounded MNEL-side Forge diagnostic lifecycle.

This module provides explicit registry, probe, witness, mutation, comparison, health,
coverage, and question-candidate contracts. It is a deterministic reference surface, not
an implementation of the external MNCS Forge authority.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Sequence

from .core import EvidenceLedger, canonical_digest, canonical_json
from .snapshots import (
    DiagnosticSnapshot,
    GraphView,
    PairView,
    SnapshotError,
    SnapshotStore,
    SnapshotView,
    TabularView,
    TraceView,
    TransitionView,
    decode_snapshot,
    graph_snapshot,
    pair_snapshot,
    tabular_snapshot,
    trace_snapshot,
    transition_snapshot,
)


AUTHORITY_DIAGNOSTIC_ONLY = "diagnostic-only"
AUTHORITY_PROPOSAL_ONLY = "proposal-only"
SEMANTICS_NOT_A_VERDICT = "not-a-verdict"
FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "verdict",
        "conformance",
        "promotion_authorized",
        "promotion",
        "evaluator_verdict",
        "evaluator_eligible",
        "future_final",
        "hidden_transfer",
        "ravel_promotion",
        "verifier_authority",
        "evaluator_authority",
        "mncs_verdict",
        "mncds_verdict",
    }
)
MAX_QUESTION_CANDIDATES = 32


class ForgeLifecycleError(ValueError):
    pass


class ProbeExecutionStatus(StrEnum):
    COMPLETED = "completed"
    INELIGIBLE = "ineligible"
    NOT_APPLICABLE = "not-applicable"
    ABSTAINED = "abstained"
    ERROR = "error"
    BUDGET_EXCEEDED = "budget-exceeded"
    UNAVAILABLE = "unavailable"
    QUARANTINED = "quarantined"


class MutationPolicy(StrEnum):
    FORBIDDEN = "forbidden"
    REGISTERED_ONLY = "registered-only"


class VerifierState(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    QUARANTINED = "quarantined"


def _reject_authority(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_AUTHORITY_KEYS:
                raise ForgeLifecycleError(f"diagnostic record contains forbidden field: {key}")
            if key == "authority" and child not in {AUTHORITY_DIAGNOSTIC_ONLY, AUTHORITY_PROPOSAL_ONLY}:
                raise ForgeLifecycleError("diagnostic record attempted to expand authority")
            _reject_authority(child)
    elif isinstance(value, list):
        for child in value:
            _reject_authority(child)


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForgeLifecycleError(f"{label} is required")
    return value


def _bounded_dict(value: dict[str, Any], label: str, max_bytes: int = 16 * 1024) -> dict[str, Any]:
    _reject_authority(value)
    try:
        encoded = canonical_json(value)
    except (TypeError, ValueError) as error:
        raise ForgeLifecycleError(f"{label} is not canonical JSON") from error
    if len(encoded) > max_bytes:
        raise ForgeLifecycleError(f"{label} exceeds its byte ceiling")
    return dict(value)


@dataclass(frozen=True, slots=True)
class Precondition:
    kind: str
    value: str | int | bool

    def __post_init__(self) -> None:
        _nonempty(self.kind, "precondition kind")
        if not isinstance(self.value, (str, int, bool)):
            raise ForgeLifecycleError("precondition values must be scalar")

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True, slots=True)
class PreconditionOutcome:
    kind: str
    expected: str | int | bool
    actual: object
    satisfied: bool
    availability: str = "available"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "expected": self.expected,
            "actual": self.actual,
            "satisfied": self.satisfied,
            "availability": self.availability,
        }


@dataclass(frozen=True, slots=True)
class PreconditionReport:
    status: str
    outcomes: tuple[PreconditionOutcome, ...]

    @property
    def satisfied(self) -> bool:
        return self.status == "satisfied"

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "outcomes": [item.to_dict() for item in self.outcomes]}


@dataclass(frozen=True, slots=True)
class VerifierDeclaration:
    verifier_id: str
    verifier_version: str
    implementation_identity: str
    accepted_snapshot_types: tuple[str, ...]
    accepted_schema_versions: tuple[int, ...]
    required_preconditions: tuple[Precondition, ...]
    expected_witness_type: str
    resource_limits: dict[str, int]
    deterministic: bool
    mutation_capability: bool
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        for value, label in (
            (self.verifier_id, "verifier id"),
            (self.verifier_version, "verifier version"),
            (self.implementation_identity, "implementation identity"),
            (self.expected_witness_type, "expected witness type"),
        ):
            _nonempty(value, label)
        if not self.accepted_snapshot_types or not self.accepted_schema_versions:
            raise ForgeLifecycleError("verifiers must declare accepted snapshot types and schema versions")
        if any(not item.strip() for item in self.accepted_snapshot_types):
            raise ForgeLifecycleError("verifier snapshot types must be non-empty")
        if any(not isinstance(item, int) or item < 1 for item in self.accepted_schema_versions):
            raise ForgeLifecycleError("verifier schema versions must be positive integers")
        required = {"operation_limit", "wall_time_ms", "output_bytes"}
        if set(self.resource_limits) != required or any(
            not isinstance(value, int) or value < 1 for value in self.resource_limits.values()
        ):
            raise ForgeLifecycleError("verifier resource limits must declare positive operation, wall, and output bounds")
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise ForgeLifecycleError("micro-verifiers are diagnostic-only")

    @property
    def declaration_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VerifierDeclaration":
        if not isinstance(value, dict):
            raise ForgeLifecycleError("verifier declaration must be an object")
        if value.get("schema") != "mnel-verifier-declaration/0.3":
            raise ForgeLifecycleError("unsupported verifier declaration schema")
        raw_preconditions = value.get("required_preconditions", ())
        if not isinstance(raw_preconditions, (list, tuple)) or any(
            not isinstance(item, dict) or "kind" not in item or "value" not in item
            for item in raw_preconditions
        ):
            raise ForgeLifecycleError("verifier preconditions are malformed")
        declaration = cls(
            verifier_id=value.get("verifier_id"),
            verifier_version=value.get("verifier_version"),
            implementation_identity=value.get("implementation_identity"),
            accepted_snapshot_types=tuple(value.get("accepted_snapshot_types", ())),
            accepted_schema_versions=tuple(value.get("accepted_schema_versions", ())),
            required_preconditions=tuple(Precondition(item["kind"], item["value"]) for item in raw_preconditions),
            expected_witness_type=value.get("expected_witness_type"),
            resource_limits=dict(value.get("resource_limits", {})),
            deterministic=value.get("deterministic"),
            mutation_capability=value.get("mutation_capability"),
            authority=value.get("authority", ""),
        )
        supplied_identity = value.get("declaration_identity")
        if supplied_identity is not None and supplied_identity != declaration.declaration_identity:
            raise ForgeLifecycleError("verifier declaration identity does not match its content")
        return declaration

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-verifier-declaration/0.3",
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "implementation_identity": self.implementation_identity,
            "accepted_snapshot_types": list(self.accepted_snapshot_types),
            "accepted_schema_versions": list(self.accepted_schema_versions),
            "required_preconditions": [item.to_dict() for item in self.required_preconditions],
            "expected_witness_type": self.expected_witness_type,
            "resource_limits": dict(self.resource_limits),
            "deterministic": self.deterministic,
            "mutation_capability": self.mutation_capability,
            "authority": self.authority,
        }
        if include_identity:
            value["declaration_identity"] = self.declaration_identity
        return value


class ReferenceVerifier(Protocol):
    def run(self, view: SnapshotView, parameters: dict[str, Any], operation_limit: int) -> dict[str, Any]: ...


class VerifierRegistry:
    def __init__(self) -> None:
        self._declarations: dict[str, VerifierDeclaration] = {}
        self._implementations: dict[str, ReferenceVerifier] = {}
        self._states: dict[str, VerifierState] = {}
        self._state_reasons: dict[str, str] = {}

    def register(self, declaration: VerifierDeclaration, implementation: ReferenceVerifier) -> str:
        identity = declaration.declaration_identity
        existing = self._declarations.get(declaration.verifier_id)
        if existing is not None:
            if existing.declaration_identity != identity:
                raise ForgeLifecycleError("verifier id collision with different declaration identity")
            raise ForgeLifecycleError("duplicate verifier registration")
        if not callable(getattr(implementation, "run", None)):
            raise ForgeLifecycleError("verifier implementation must expose run")
        self._declarations[declaration.verifier_id] = declaration
        self._implementations[declaration.verifier_id] = implementation
        self._states[declaration.verifier_id] = VerifierState.ENABLED
        return identity

    def register_declaration(self, declaration: VerifierDeclaration) -> str:
        """Register a declaration without an executable implementation."""

        identity = declaration.declaration_identity
        if declaration.verifier_id in self._declarations:
            raise ForgeLifecycleError("duplicate verifier registration")
        self._declarations[declaration.verifier_id] = declaration
        self._states[declaration.verifier_id] = VerifierState.DISABLED
        self._state_reasons[declaration.verifier_id] = "no local implementation attached"
        return identity

    def load(self, declarations: Sequence[dict[str, Any]]) -> tuple[str, ...]:
        """Load explicit declaration objects; executable implementations remain separate."""

        identities = []
        for value in declarations:
            declaration = VerifierDeclaration.from_dict(value)
            identities.append(self.register_declaration(declaration))
        return tuple(identities)

    def lookup(self, verifier_id: str) -> VerifierDeclaration:
        try:
            return self._declarations[verifier_id]
        except KeyError as error:
            raise ForgeLifecycleError(f"unknown verifier: {verifier_id}") from error

    def implementation(self, verifier_id: str) -> ReferenceVerifier:
        try:
            return self._implementations[verifier_id]
        except KeyError as error:
            raise ForgeLifecycleError(f"verifier has no executable implementation: {verifier_id}") from error

    def set_state(self, verifier_id: str, state: VerifierState, reason: str = "") -> None:
        self.lookup(verifier_id)
        if state is VerifierState.QUARANTINED and not reason.strip():
            raise ForgeLifecycleError("quarantine requires a reason")
        self._states[verifier_id] = state
        if reason:
            self._state_reasons[verifier_id] = reason

    def state(self, verifier_id: str) -> VerifierState:
        self.lookup(verifier_id)
        return self._states[verifier_id]

    def declarations(self) -> tuple[VerifierDeclaration, ...]:
        return tuple(self._declarations[key] for key in sorted(self._declarations))

    def match(self, snapshot: DiagnosticSnapshot, context: dict[str, Any] | None = None) -> tuple[VerifierDeclaration, ...]:
        snapshot.validate_integrity()
        decode_snapshot(snapshot)
        matches: list[VerifierDeclaration] = []
        for declaration in self.declarations():
            if self._states[declaration.verifier_id] is not VerifierState.ENABLED:
                continue
            if snapshot.snapshot_type not in declaration.accepted_snapshot_types:
                continue
            if snapshot.schema_version not in declaration.accepted_schema_versions:
                continue
            report = evaluate_preconditions(
                declaration.required_preconditions,
                snapshot,
                decode_snapshot(snapshot),
                context or {},
            )
            if report.satisfied:
                matches.append(declaration)
        return tuple(matches)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "mnel-verifier-registry/0.3",
            "verifiers": [
                {
                    **declaration.to_dict(),
                    "state": self._states[declaration.verifier_id].value,
                    "state_reason": self._state_reasons.get(declaration.verifier_id),
                }
                for declaration in self.declarations()
            ],
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "registry_identity": canonical_digest(
                [declaration.to_dict() for declaration in self.declarations()]
            ),
        }


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    probe_id: str
    question: str
    subject_identities: dict[str, str]
    verifier_id: str
    snapshot_identities: tuple[str, ...]
    expected_witness_type: str
    preconditions: tuple[Precondition, ...]
    resource_budget: dict[str, int]
    mutation_policy: MutationPolicy
    runtime_identity: dict[str, str]
    lineage: dict[str, str]
    parameters: dict[str, Any] = field(default_factory=dict)
    authority: str = AUTHORITY_PROPOSAL_ONLY

    def __post_init__(self) -> None:
        for value, label in (
            (self.probe_id, "probe id"),
            (self.question, "probe question"),
            (self.verifier_id, "verifier id"),
            (self.expected_witness_type, "expected witness type"),
        ):
            _nonempty(value, label)
        if not self.snapshot_identities or any(
            not isinstance(item, str) or not item.strip() for item in self.snapshot_identities
        ):
            raise ForgeLifecycleError("probe requests require snapshot identities")
        _bounded_dict(self.subject_identities, "subject identities")
        _bounded_dict(self.runtime_identity, "runtime identity")
        _bounded_dict(self.lineage, "probe lineage")
        _bounded_dict(self.parameters, "probe parameters")
        required = {"operation_limit", "wall_time_ms", "output_bytes"}
        if set(self.resource_budget) != required or any(
            not isinstance(value, int) or value < 1 for value in self.resource_budget.values()
        ):
            raise ForgeLifecycleError("probe budget must declare positive operation, wall, and output bounds")
        if self.authority != AUTHORITY_PROPOSAL_ONLY:
            raise ForgeLifecycleError("investigator probe requests are proposal-only")

    @property
    def question_identity(self) -> str:
        return canonical_digest(
            {
                "question": self.question,
                "subject_identities": self.subject_identities,
                "snapshot_identities": self.snapshot_identities,
                "parameters": self.parameters,
            }
        )

    @property
    def request_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-forge-probe-request/0.3",
            "probe_id": self.probe_id,
            "question": self.question,
            "question_identity": self.question_identity,
            "subject_identities": dict(self.subject_identities),
            "verifier_id": self.verifier_id,
            "snapshot_identities": list(self.snapshot_identities),
            "expected_witness_type": self.expected_witness_type,
            "preconditions": [item.to_dict() for item in self.preconditions],
            "resource_budget": dict(self.resource_budget),
            "mutation_policy": self.mutation_policy.value,
            "runtime_identity": dict(self.runtime_identity),
            "lineage": dict(self.lineage),
            "parameters": dict(self.parameters),
            "authority": self.authority,
            "semantics": "diagnostic-request; not-a-verdict",
        }
        if include_identity:
            value["request_identity"] = self.request_identity
        return value


def evaluate_preconditions(
    preconditions: Sequence[Precondition],
    snapshot: DiagnosticSnapshot,
    view: SnapshotView,
    context: dict[str, Any],
) -> PreconditionReport:
    outcomes: list[PreconditionOutcome] = []
    for condition in preconditions:
        actual: object
        available = "available"
        if condition.kind == "required_snapshot_type":
            actual = snapshot.snapshot_type
        elif condition.kind == "required_schema_version":
            actual = snapshot.schema_version
        elif condition.kind == "dependency_identity":
            actual = snapshot.dependency_identity
        elif condition.kind == "source_language":
            actual = context.get("source_language")
        elif condition.kind == "feature_available":
            features = context.get("features", ())
            actual = condition.value in features
        elif condition.kind == "tool_available":
            tools = context.get("tools", ())
            actual = condition.value in tools
        elif condition.kind == "mutation_allowed":
            actual = context.get("mutation_allowed")
        elif condition.kind == "minimum_rows":
            actual = len(view.rows) if isinstance(view, TabularView) else None
        elif condition.kind == "minimum_nodes":
            actual = len(view.nodes) if isinstance(view, GraphView) else None
        elif condition.kind == "minimum_events":
            actual = len(view.events) if isinstance(view, TraceView) else None
        else:
            actual = None
            available = "unavailable"
        satisfied = _precondition_matches(condition, actual, available)
        outcomes.append(PreconditionOutcome(condition.kind, condition.value, actual, satisfied, available))
    if any(item.availability == "unavailable" for item in outcomes):
        status = "unavailable"
    elif all(item.satisfied for item in outcomes):
        status = "satisfied"
    else:
        status = "failed"
    return PreconditionReport(status, tuple(outcomes))


def _precondition_matches(condition: Precondition, actual: object, availability: str) -> bool:
    if availability != "available":
        return False
    if condition.kind.startswith("minimum_"):
        return isinstance(actual, int) and isinstance(condition.value, int) and actual >= condition.value
    return actual == condition.value


@dataclass(frozen=True, slots=True)
class Witness:
    probe_identity: str
    question_identity: str
    verifier_id: str
    verifier_version: str
    implementation_identity: str
    snapshot_identities: tuple[str, ...]
    expected_witness_type: str
    precondition_report: PreconditionReport
    execution_status: ProbeExecutionStatus
    diagnostic_output: dict[str, Any]
    resource_usage: dict[str, int]
    mutation_identity: str | None = None
    error: str | None = None
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY
    semantics: str = SEMANTICS_NOT_A_VERDICT

    def __post_init__(self) -> None:
        _bounded_dict(self.diagnostic_output, "witness diagnostic output", 32 * 1024)
        _bounded_dict(self.resource_usage, "witness resource usage", 4096)
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY or self.semantics != SEMANTICS_NOT_A_VERDICT:
            raise ForgeLifecycleError("witness authority is fixed to diagnostic-only")

    @property
    def witness_identity(self) -> str:
        identity_body = self.to_dict(include_identity=False)
        # Timing is evidence, but wall-clock timing is not a stable content identity.
        identity_body["resource_usage"] = {
            key: value for key, value in self.resource_usage.items() if key != "elapsed_ns"
        }
        return canonical_digest(identity_body)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-diagnostic-witness/0.3",
            "probe_identity": self.probe_identity,
            "question_identity": self.question_identity,
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "implementation_identity": self.implementation_identity,
            "snapshot_identities": list(self.snapshot_identities),
            "expected_witness_type": self.expected_witness_type,
            "precondition_report": self.precondition_report.to_dict(),
            "execution_status": self.execution_status.value,
            "diagnostic_output": dict(self.diagnostic_output),
            "resource_usage": dict(self.resource_usage),
            "mutation_identity": self.mutation_identity,
            "error": self.error,
            "authority": self.authority,
            "semantics": self.semantics,
        }
        if include_identity:
            value["witness_identity"] = self.witness_identity
        return value


class TransitionChangeVerifier:
    def run(self, view: SnapshotView, parameters: dict[str, Any], operation_limit: int) -> dict[str, Any]:
        if not isinstance(view, TransitionView):
            raise ForgeLifecycleError("transition verifier received an incompatible view")
        if operation_limit < 1:
            raise ForgeLifecycleError("operation budget exhausted")
        changed = view.previous_state != view.next_state
        return {"condition_observed": changed, "changed": changed, "member_bytes": len(view.previous_state) + len(view.next_state)}


class TabularBoundsVerifier:
    def run(self, view: SnapshotView, parameters: dict[str, Any], operation_limit: int) -> dict[str, Any]:
        if not isinstance(view, TabularView):
            raise ForgeLifecycleError("tabular verifier received an incompatible view")
        minimum = parameters.get("minimum")
        maximum = parameters.get("maximum")
        if (
            not isinstance(minimum, (int, float))
            or isinstance(minimum, bool)
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum > maximum
        ):
            raise ForgeLifecycleError("tabular verifier requires finite minimum and maximum parameters")
        values = [value for row in view.rows for value in row]
        if len(values) > operation_limit:
            raise ForgeLifecycleError("tabular verifier operation budget exceeded")
        outliers = [value for value in values if value < minimum or value > maximum]
        return {"condition_observed": not outliers, "outlier_count": len(outliers), "minimum": minimum, "maximum": maximum}


class PairRelationVerifier:
    def run(self, view: SnapshotView, parameters: dict[str, Any], operation_limit: int) -> dict[str, Any]:
        if not isinstance(view, PairView):
            raise ForgeLifecycleError("pair verifier received an incompatible view")
        relation = parameters.get("relation", "equal")
        if relation not in {"equal", "different"}:
            raise ForgeLifecycleError("pair relation must be equal or different")
        equal = view.left == view.right
        observed = equal if relation == "equal" else not equal
        return {"condition_observed": observed, "equal": equal, "relation": relation}


class TraceOrderVerifier:
    def run(self, view: SnapshotView, parameters: dict[str, Any], operation_limit: int) -> dict[str, Any]:
        if not isinstance(view, TraceView):
            raise ForgeLifecycleError("trace verifier received an incompatible view")
        before = parameters.get("before")
        after = parameters.get("after")
        if not isinstance(before, str) or not isinstance(after, str):
            raise ForgeLifecycleError("trace verifier requires before and after event types")
        if len(view.events) > operation_limit:
            raise ForgeLifecycleError("trace verifier operation budget exceeded")
        before_positions = [index for index, event in enumerate(view.events) if event.event_type == before]
        after_positions = [index for index, event in enumerate(view.events) if event.event_type == after]
        observed = bool(before_positions and after_positions and min(before_positions) < max(after_positions))
        return {"condition_observed": observed, "before": before, "after": after, "event_count": len(view.events)}


class GraphEdgeVerifier:
    def run(self, view: SnapshotView, parameters: dict[str, Any], operation_limit: int) -> dict[str, Any]:
        if not isinstance(view, GraphView):
            raise ForgeLifecycleError("graph verifier received an incompatible view")
        source, target, edge_type = parameters.get("source"), parameters.get("target"), parameters.get("edge_type")
        if not isinstance(source, int) or not isinstance(target, int) or not isinstance(edge_type, str):
            raise ForgeLifecycleError("graph verifier requires source, target, and edge_type")
        if len(view.edges) > operation_limit:
            raise ForgeLifecycleError("graph verifier operation budget exceeded")
        observed = any(edge.source == source and edge.target == target and edge.edge_type == edge_type for edge in view.edges)
        return {"condition_observed": observed, "source": source, "target": target, "edge_type": edge_type}


def reference_verifier_registry() -> VerifierRegistry:
    registry = VerifierRegistry()
    common = {"operation_limit": 4096, "wall_time_ms": 100, "output_bytes": 8192}
    registry.register(
        VerifierDeclaration("transition-change", "0.3.0", "mnel-reference-transition/1", ("transition",), (1,), (), "transition-witness", common, True, False),
        TransitionChangeVerifier(),
    )
    registry.register(
        VerifierDeclaration("transition-change-independent", "0.3.0", "mnel-reference-transition-independent/1", ("transition",), (1,), (), "transition-witness", common, True, False),
        TransitionChangeVerifier(),
    )
    registry.register(
        VerifierDeclaration("tabular-bounds", "0.3.0", "mnel-reference-tabular-bounds/1", ("tabular",), (1,), (), "tabular-witness", common, True, False),
        TabularBoundsVerifier(),
    )
    registry.register(
        VerifierDeclaration("pair-relation", "0.3.0", "mnel-reference-pair-relation/1", ("pair",), (1,), (), "pair-witness", common, True, False),
        PairRelationVerifier(),
    )
    registry.register(
        VerifierDeclaration("trace-order", "0.3.0", "mnel-reference-trace-order/1", ("trace",), (1,), (), "trace-witness", common, True, False),
        TraceOrderVerifier(),
    )
    registry.register(
        VerifierDeclaration("graph-edge", "0.3.0", "mnel-reference-graph-edge/1", ("graph",), (1,), (), "graph-witness", common, True, False),
        GraphEdgeVerifier(),
    )
    return registry


class VerifierHealthStore:
    def __init__(self, quarantine_after_errors: int = 3) -> None:
        if quarantine_after_errors < 1:
            raise ForgeLifecycleError("quarantine_after_errors must be positive")
        self.quarantine_after_errors = quarantine_after_errors
        self._records: dict[str, dict[str, Any]] = {}

    def _record(self, verifier_id: str) -> dict[str, Any]:
        return self._records.setdefault(
            verifier_id,
            {
                "successful_executions": 0,
                "execution_errors": 0,
                "precondition_exclusions": 0,
                "abstentions": 0,
                "budget_violations": 0,
                "malformed_outputs": 0,
                "latency_ns_total": 0,
                "latency_samples": 0,
                "snapshot_types": set(),
                "quarantined": False,
                "quarantine_reason": None,
            },
        )

    def observe(self, verifier_id: str, snapshot_type: str, status: ProbeExecutionStatus, elapsed_ns: int) -> None:
        record = self._record(verifier_id)
        record["latency_ns_total"] += max(0, elapsed_ns)
        record["latency_samples"] += 1
        record["snapshot_types"].add(snapshot_type)
        if status is ProbeExecutionStatus.COMPLETED:
            record["successful_executions"] += 1
        elif status in {ProbeExecutionStatus.INELIGIBLE, ProbeExecutionStatus.NOT_APPLICABLE}:
            record["precondition_exclusions"] += 1
        elif status is ProbeExecutionStatus.ABSTAINED:
            record["abstentions"] += 1
        elif status is ProbeExecutionStatus.BUDGET_EXCEEDED:
            record["budget_violations"] += 1
        elif status is ProbeExecutionStatus.ERROR:
            record["execution_errors"] += 1
            if record["execution_errors"] >= self.quarantine_after_errors:
                record["quarantined"] = True
                record["quarantine_reason"] = "repeated verifier execution errors"

    def quarantine(self, verifier_id: str, reason: str) -> None:
        _nonempty(reason, "quarantine reason")
        record = self._record(verifier_id)
        record["quarantined"] = True
        record["quarantine_reason"] = reason

    def is_quarantined(self, verifier_id: str) -> bool:
        return bool(self._record(verifier_id)["quarantined"])

    def malformed_output(self, verifier_id: str, reason: str) -> None:
        _nonempty(reason, "malformed output reason")
        record = self._record(verifier_id)
        record["malformed_outputs"] += 1

    def to_dict(self, verifier_id: str) -> dict[str, object]:
        record = self._record(verifier_id)
        value = {
            "schema": "mnel-verifier-health/0.3",
            "verifier_id": verifier_id,
            **{key: (sorted(item) if isinstance(item, set) else item) for key, item in record.items()},
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "health-is-not-truth",
        }
        value["health_identity"] = canonical_digest(value)
        return value


class ReferenceForgeRuntime:
    def __init__(self, snapshots: SnapshotStore, verifiers: VerifierRegistry, health: VerifierHealthStore | None = None) -> None:
        self.snapshots = snapshots
        self.verifiers = verifiers
        self.health = health or VerifierHealthStore()

    def execute(self, request: ProbeRequest, *, mutation_identity: str | None = None) -> Witness:
        started = time.monotonic_ns()
        if mutation_identity is not None and request.mutation_policy is MutationPolicy.FORBIDDEN:
            return self._witness(
                request,
                None,
                ProbeExecutionStatus.NOT_APPLICABLE,
                {},
                {},
                "probe request forbids mutation execution",
                started,
                mutation_identity,
            )
        try:
            declaration = self.verifiers.lookup(request.verifier_id)
        except ForgeLifecycleError as error:
            return self._witness(request, None, ProbeExecutionStatus.UNAVAILABLE, {}, {}, str(error), started, mutation_identity)
        state = self.verifiers.state(request.verifier_id)
        if state is not VerifierState.ENABLED:
            status = ProbeExecutionStatus.QUARANTINED if state is VerifierState.QUARANTINED else ProbeExecutionStatus.UNAVAILABLE
            return self._witness(request, declaration, status, {}, {}, f"verifier state is {state.value}", started, mutation_identity)
        if declaration.expected_witness_type != request.expected_witness_type:
            return self._witness(request, declaration, ProbeExecutionStatus.NOT_APPLICABLE, {}, {}, "witness type mismatch", started, mutation_identity)
        try:
            snapshot = self.snapshots.get(request.snapshot_identities[0])
            if len(request.snapshot_identities) != 1:
                raise ForgeLifecycleError("reference verifiers require exactly one snapshot")
            view = self.snapshots.view(
                snapshot.snapshot_identity,
                accepted_types=declaration.accepted_snapshot_types,
                schema_versions=declaration.accepted_schema_versions,
            )
            report = evaluate_preconditions(request.preconditions + declaration.required_preconditions, snapshot, view, request.parameters)
            if report.status != "satisfied":
                status = ProbeExecutionStatus.INELIGIBLE if report.status == "failed" else ProbeExecutionStatus.NOT_APPLICABLE
                return self._witness(request, declaration, status, {}, {"precondition_report": report.to_dict()}, None, started, mutation_identity, report)
            operation_limit = min(
                request.resource_budget["operation_limit"], declaration.resource_limits["operation_limit"]
            )
            output = self.verifiers.implementation(request.verifier_id).run(
                view, request.parameters, operation_limit
            )
            if not isinstance(output, dict):
                self.health.malformed_output(request.verifier_id, "verifier output is not an object")
                raise ForgeLifecycleError("verifier output must be an object")
            operation_count = output.get("operation_count", 0)
            if (
                not isinstance(operation_count, int)
                or isinstance(operation_count, bool)
                or operation_count < 0
                or operation_count > operation_limit
            ):
                self.health.malformed_output(request.verifier_id, "verifier operation count is invalid")
                raise ForgeLifecycleError("verifier operation count exceeds its budget")
            encoded = canonical_json(output)
            if len(encoded) > min(
                request.resource_budget["output_bytes"], declaration.resource_limits["output_bytes"]
            ):
                self.health.malformed_output(request.verifier_id, "verifier output exceeds its byte budget")
                raise ForgeLifecycleError("verifier output exceeds its byte budget")
            elapsed = time.monotonic_ns() - started
            wall_limit_ms = min(request.resource_budget["wall_time_ms"], declaration.resource_limits["wall_time_ms"])
            if elapsed > wall_limit_ms * 1_000_000:
                status = ProbeExecutionStatus.BUDGET_EXCEEDED
            else:
                status = ProbeExecutionStatus.COMPLETED
            witness = self._witness(
                request,
                declaration,
                status,
                output,
                {"operation_count": operation_count, "output_bytes": len(encoded)},
                None,
                started,
                mutation_identity,
                report,
            )
            self.health.observe(request.verifier_id, snapshot.snapshot_type, status, elapsed)
            if self.health.is_quarantined(request.verifier_id):
                self.verifiers.set_state(
                    request.verifier_id,
                    VerifierState.QUARANTINED,
                    "repeated verifier execution errors",
                )
            return witness
        except (ForgeLifecycleError, SnapshotError, TypeError, ValueError) as error:
            snapshot_type = snapshot.snapshot_type if "snapshot" in locals() else "unknown"
            status = ProbeExecutionStatus.BUDGET_EXCEEDED if "budget" in str(error).lower() else ProbeExecutionStatus.ERROR
            self.health.observe(request.verifier_id, snapshot_type, status, time.monotonic_ns() - started)
            if self.health.is_quarantined(request.verifier_id):
                self.verifiers.set_state(
                    request.verifier_id,
                    VerifierState.QUARANTINED,
                    "repeated verifier execution errors",
                )
            return self._witness(request, declaration, status, {}, {}, str(error), started, mutation_identity)

    def _witness(
        self,
        request: ProbeRequest,
        declaration: VerifierDeclaration | None,
        status: ProbeExecutionStatus,
        output: dict[str, Any],
        usage: dict[str, Any],
        error: str | None,
        started: int,
        mutation_identity: str | None,
        report: PreconditionReport | None = None,
    ) -> Witness:
        report = report or PreconditionReport("unavailable", ())
        return Witness(
            probe_identity=request.request_identity,
            question_identity=request.question_identity,
            verifier_id=request.verifier_id,
            verifier_version=declaration.verifier_version if declaration else "unknown",
            implementation_identity=declaration.implementation_identity if declaration else "unknown",
            snapshot_identities=request.snapshot_identities,
            expected_witness_type=request.expected_witness_type,
            precondition_report=report,
            execution_status=status,
            diagnostic_output=output,
            resource_usage={"elapsed_ns": time.monotonic_ns() - started, **usage},
            mutation_identity=mutation_identity,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class WitnessComparison:
    question_identity: str
    subject_identities: dict[str, str]
    witness_identities: tuple[str, ...]
    verifier_ids: tuple[str, ...]
    comparison_status: str
    disagreement_fields: tuple[str, ...]
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY
    semantics: str = "evidence-characterization; not-a-verdict"

    def __post_init__(self) -> None:
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise ForgeLifecycleError("witness comparisons are diagnostic-only")
        if self.semantics != "evidence-characterization; not-a-verdict":
            raise ForgeLifecycleError("witness comparison semantics are fixed")

    @property
    def comparison_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-witness-comparison/0.3",
            "question_identity": self.question_identity,
            "subject_identities": dict(self.subject_identities),
            "witness_identities": list(self.witness_identities),
            "verifier_ids": list(self.verifier_ids),
            "comparison_status": self.comparison_status,
            "disagreement_fields": list(self.disagreement_fields),
            "authority": self.authority,
            "semantics": self.semantics,
        }
        if include_identity:
            value["comparison_identity"] = self.comparison_identity
        return value


def compare_witnesses(witnesses: Sequence[Witness], subject_identities: dict[str, str]) -> WitnessComparison:
    if len(witnesses) < 2:
        raise ForgeLifecycleError("independent comparison requires at least two witnesses")
    if len({item.question_identity for item in witnesses}) != 1:
        raise ForgeLifecycleError("witnesses address different identified questions")
    completed = [item for item in witnesses if item.execution_status is ProbeExecutionStatus.COMPLETED]
    signatures = [
        canonical_json({key: item.diagnostic_output.get(key) for key in ("condition_observed", "relation", "equal")})
        for item in completed
    ]
    if len(completed) != len(witnesses):
        status = "incomplete"
        fields = ("execution_status",)
    elif len(set(signatures)) == 1:
        status = "agreement"
        fields = ()
    else:
        status = "disagreement"
        fields = ("diagnostic_output",)
    return WitnessComparison(
        witnesses[0].question_identity,
        dict(subject_identities),
        tuple(item.witness_identity for item in witnesses),
        tuple(sorted(item.verifier_id for item in witnesses)),
        status,
        fields,
    )


@dataclass(frozen=True, slots=True)
class MutationRecord:
    original_snapshot_identity: str
    mutation_operator_identity: str
    parameters: dict[str, Any]
    resulting_snapshot_identity: str
    scope: str
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY

    def __post_init__(self) -> None:
        for value, label in (
            (self.original_snapshot_identity, "original snapshot identity"),
            (self.mutation_operator_identity, "mutation operator identity"),
            (self.resulting_snapshot_identity, "resulting snapshot identity"),
            (self.scope, "mutation scope"),
        ):
            _nonempty(value, label)
        _bounded_dict(self.parameters, "mutation parameters")
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY:
            raise ForgeLifecycleError("mutations are diagnostic-only")

    @property
    def mutation_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-mutation-record/0.3",
            "original_snapshot_identity": self.original_snapshot_identity,
            "mutation_operator_identity": self.mutation_operator_identity,
            "parameters": dict(self.parameters),
            "resulting_snapshot_identity": self.resulting_snapshot_identity,
            "scope": self.scope,
            "authority": self.authority,
            "semantics": "diagnostic-experiment; not-a-verdict",
        }
        if include_identity:
            value["mutation_identity"] = self.mutation_identity
        return value


class MutationRegistry:
    OPERATORS = frozenset({"transition.swap", "trace.remove-event", "trace.swap-events", "tabular.perturb", "graph.remove-edge", "pair.swap"})

    def apply(self, operator_id: str, snapshot: DiagnosticSnapshot, parameters: dict[str, Any], store: SnapshotStore) -> MutationRecord:
        if operator_id not in self.OPERATORS:
            raise ForgeLifecycleError("mutation operator is not registered")
        snapshot.validate_integrity()
        view = decode_snapshot(snapshot)
        kwargs = {
            "producer_identity": f"{snapshot.producer_identity}|mutation:{operator_id}",
            "source_identity": snapshot.source_identity,
            "dependency_identity": snapshot.dependency_identity,
            "feature_extractor_identity": snapshot.feature_extractor_identity,
            "schema_version": snapshot.schema_version,
            "schema_identity": snapshot.schema_identity,
        }
        if operator_id == "transition.swap" and isinstance(view, TransitionView):
            mutated = transition_snapshot(view.next_state, view.previous_state, **kwargs)
        elif operator_id == "pair.swap" and isinstance(view, PairView):
            mutated = pair_snapshot(view.right, view.left, **kwargs)
        elif operator_id == "trace.remove-event" and isinstance(view, TraceView):
            index = _parameter_index(parameters, len(view.events))
            events = view.events[:index] + view.events[index + 1 :]
            if not events:
                raise ForgeLifecycleError("trace mutation may not remove its final event")
            mutated = trace_snapshot(tuple((event.event_type, event.payload) for event in events), **kwargs)
        elif operator_id == "trace.swap-events" and isinstance(view, TraceView):
            first = _parameter_index(parameters, len(view.events))
            second = _parameter_index(parameters, len(view.events), "second_index")
            if first == second:
                raise ForgeLifecycleError("trace swap requires two distinct event indexes")
            events = list(view.events)
            events[first], events[second] = events[second], events[first]
            mutated = trace_snapshot(tuple((event.event_type, event.payload) for event in events), **kwargs)
        elif operator_id == "tabular.perturb" and isinstance(view, TabularView):
            row = _parameter_index(parameters, len(view.rows), "row")
            column = _parameter_index(parameters, view.column_count, "column")
            delta = parameters.get("delta")
            if not isinstance(delta, (int, float)) or not math.isfinite(delta):
                raise ForgeLifecycleError("tabular perturbation delta must be finite")
            rows = [list(values) for values in view.rows]
            rows[row][column] += float(delta)
            mutated = tabular_snapshot(tuple(tuple(values) for values in rows), **kwargs)
        elif operator_id == "graph.remove-edge" and isinstance(view, GraphView):
            index = _parameter_index(parameters, len(view.edges))
            edges = view.edges[:index] + view.edges[index + 1 :]
            mutated = graph_snapshot(tuple((node.node_id, node.label) for node in view.nodes), tuple((edge.source, edge.target, edge.edge_type) for edge in edges), **kwargs)
        else:
            raise ForgeLifecycleError("mutation operator is incompatible with the snapshot type")
        store.register(mutated)
        return MutationRecord(snapshot.snapshot_identity, f"mnel-mutation/{operator_id}/1", dict(parameters), mutated.snapshot_identity, "bounded registered mutation")


def _parameter_index(parameters: dict[str, Any], length: int, name: str = "index") -> int:
    value = parameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value >= length:
        raise ForgeLifecycleError(f"mutation {name} is outside its bounded range")
    return value


@dataclass(frozen=True, slots=True)
class LearnedDiagnosticEvent:
    provider_id: str
    provider_observation_identity: str
    snapshot_identities: tuple[str, ...]
    declaration_identity: str
    source_record_ids: tuple[str, ...]
    payload: dict[str, Any]
    authority: str = AUTHORITY_DIAGNOSTIC_ONLY
    semantics: str = SEMANTICS_NOT_A_VERDICT

    def __post_init__(self) -> None:
        for value, label in (
            (self.provider_id, "provider id"),
            (self.provider_observation_identity, "provider observation identity"),
            (self.declaration_identity, "provider declaration identity"),
        ):
            _nonempty(value, label)
        if not self.snapshot_identities or any(
            not isinstance(item, str) or not item.strip() for item in self.snapshot_identities
        ):
            raise ForgeLifecycleError("learned observations require snapshot identities")
        _bounded_dict(self.payload, "learned diagnostic payload")
        if self.authority != AUTHORITY_DIAGNOSTIC_ONLY or self.semantics != SEMANTICS_NOT_A_VERDICT:
            raise ForgeLifecycleError("learned provider observations are diagnostic-only")

    @property
    def event_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    @classmethod
    def from_observation(cls, observation: dict[str, Any]) -> "LearnedDiagnosticEvent":
        _reject_authority(observation)
        required = ("provider_id", "observation_identity", "snapshot_ids", "declaration_identity")
        if any(key not in observation for key in required):
            raise ForgeLifecycleError("learned observation lacks identity-bound fields")
        provider_id = observation["provider_id"]
        observation_identity = observation["observation_identity"]
        snapshot_ids = observation["snapshot_ids"]
        declaration_identity = observation["declaration_identity"]
        if not isinstance(snapshot_ids, (list, tuple)) or not snapshot_ids:
            raise ForgeLifecycleError("learned observation requires snapshot identities")
        for value, label in (
            (provider_id, "provider id"),
            (observation_identity, "provider observation identity"),
            (declaration_identity, "provider declaration identity"),
        ):
            _nonempty(value, label)
        if any(not isinstance(item, str) or not item.strip() for item in snapshot_ids):
            raise ForgeLifecycleError("learned observation snapshot identities must be non-empty strings")
        payload = {key: value for key, value in observation.items() if key not in required}
        _bounded_dict(payload, "learned diagnostic payload")
        return cls(
            provider_id,
            observation_identity,
            tuple(snapshot_ids),
            declaration_identity,
            tuple(str(item) for item in observation.get("source_record_ids", ())),
            payload,
        )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-learned-provider-observation-event/0.3",
            "record_type": "learned-provider-observation",
            "provider_id": self.provider_id,
            "provider_observation_identity": self.provider_observation_identity,
            "snapshot_identities": list(self.snapshot_identities),
            "declaration_identity": self.declaration_identity,
            "source_record_ids": list(self.source_record_ids),
            "payload": dict(self.payload),
            "authority": self.authority,
            "semantics": self.semantics,
        }
        if include_identity:
            value["event_identity"] = self.event_identity
        return value


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    registered_snapshot_types: tuple[str, ...]
    exercised_snapshot_types: tuple[str, ...]
    exercised_verifier_ids: tuple[str, ...]
    uncovered_snapshot_types: tuple[str, ...]
    single_source_question_identities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.registered_snapshot_types and self.exercised_snapshot_types:
            raise ForgeLifecycleError("exercised coverage cannot exceed registered coverage")

    @property
    def coverage_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-verifier-coverage/0.3",
            "registered_snapshot_types": list(self.registered_snapshot_types),
            "exercised_snapshot_types": list(self.exercised_snapshot_types),
            "exercised_verifier_ids": list(self.exercised_verifier_ids),
            "uncovered_snapshot_types": list(self.uncovered_snapshot_types),
            "single_source_question_identities": list(self.single_source_question_identities),
            "authority": AUTHORITY_DIAGNOSTIC_ONLY,
            "semantics": "coverage-is-not-truth",
        }
        if include_identity:
            value["coverage_identity"] = self.coverage_identity
        return value


def build_coverage(registry: VerifierRegistry, snapshots: SnapshotStore, witnesses: Sequence[Witness]) -> CoverageRecord:
    registered = sorted(
        {
            item
            for declaration in registry.declarations()
            if registry.state(declaration.verifier_id) is VerifierState.ENABLED
            for item in declaration.accepted_snapshot_types
        }
    )
    exercised = sorted({snapshots.get(identity).snapshot_type for witness in witnesses for identity in witness.snapshot_identities if witness.execution_status is ProbeExecutionStatus.COMPLETED})
    verifier_ids = sorted({witness.verifier_id for witness in witnesses if witness.execution_status is ProbeExecutionStatus.COMPLETED})
    question_verifiers: dict[str, set[str]] = {}
    for witness in witnesses:
        if witness.execution_status is ProbeExecutionStatus.COMPLETED:
            question_verifiers.setdefault(witness.question_identity, set()).add(witness.verifier_id)
    single = sorted(question for question, verifier_ids_for_question in question_verifiers.items() if len(verifier_ids_for_question) < 2)
    return CoverageRecord(tuple(registered), tuple(exercised), tuple(verifier_ids), tuple(item for item in registered if item not in exercised), tuple(single))


@dataclass(frozen=True, slots=True)
class QuestionCandidate:
    subject_identity: str
    reason: str
    target_snapshot_type: str | None
    supporting_record_ids: tuple[str, ...]
    authority: str = AUTHORITY_PROPOSAL_ONLY
    candidate_kind: str = "coverage-gap"
    priority: int = 0
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.subject_identity, "candidate subject identity")
        _nonempty(self.reason, "candidate reason")
        _nonempty(self.candidate_kind, "candidate kind")
        if self.authority != AUTHORITY_PROPOSAL_ONLY or not 0 <= self.priority <= 100:
            raise ForgeLifecycleError("question candidates are proposal-only")
        if not self.supporting_record_ids:
            raise ForgeLifecycleError("question candidates require evidence lineage")

    @property
    def candidate_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "mnel-omitted-question-candidate/0.3",
            "subject_identity": self.subject_identity,
            "reason": self.reason,
            "target_snapshot_type": self.target_snapshot_type,
            "supporting_record_ids": list(self.supporting_record_ids),
            "authority": self.authority,
            "candidate_kind": self.candidate_kind,
            "priority": self.priority,
            "lineage": list(self.lineage or self.supporting_record_ids),
            "semantics": "proposal-only; not-a-verdict",
        }
        if include_identity:
            value["candidate_identity"] = self.candidate_identity
        return value


def discover_question_candidates(
    subject_identity: str,
    snapshots: SnapshotStore,
    coverage: CoverageRecord,
    comparisons: Sequence[WitnessComparison] = (),
    witnesses: Sequence[Witness] = (),
    mutations: Sequence[MutationRecord] = (),
    learned_observations: Sequence[LearnedDiagnosticEvent] = (),
    registry: VerifierRegistry | None = None,
    visible_lineage: frozenset[str] | None = None,
    max_candidates: int = MAX_QUESTION_CANDIDATES,
) -> tuple[QuestionCandidate, ...]:
    if max_candidates < 1 or max_candidates > MAX_QUESTION_CANDIDATES:
        raise ForgeLifecycleError("question candidate budget is outside its bounded range")
    candidates: dict[str, QuestionCandidate] = {}

    def add(
        reason: str,
        target_snapshot_type: str | None,
        supporting: Sequence[str],
        *,
        candidate_kind: str,
        priority: int,
    ) -> None:
        lineage = tuple(dict.fromkeys(str(item) for item in supporting if str(item).strip()))
        if not lineage:
            raise ForgeLifecycleError("skeptic candidate has no evidence lineage")
        if visible_lineage is not None and not set(lineage).issubset(visible_lineage):
            raise ForgeLifecycleError("skeptic candidate attempted to use unavailable evidence")
        candidate = QuestionCandidate(
            subject_identity,
            reason,
            target_snapshot_type,
            lineage,
            candidate_kind=candidate_kind,
            priority=priority,
            lineage=lineage,
        )
        candidates[candidate.candidate_identity] = candidate

    for snapshot_type in coverage.uncovered_snapshot_types:
        add(
            "no compatible verifier exercised this snapshot type",
            snapshot_type,
            (coverage.coverage_identity,),
            candidate_kind="missing-verifier-coverage",
            priority=90,
        )
    for question_identity in coverage.single_source_question_identities:
        add(
            "identified question has only one diagnostic verifier",
            None,
            (question_identity,),
            candidate_kind="single-diagnostic-source",
            priority=70,
        )
    for comparison in comparisons:
        if comparison.comparison_status == "disagreement":
            add(
                "independent witnesses disagree",
                None,
                comparison.witness_identities,
                candidate_kind="witness-disagreement",
                priority=100,
            )
    status_groups: dict[str, list[Witness]] = {}
    for witness in witnesses:
        if witness.execution_status in {
            ProbeExecutionStatus.INELIGIBLE,
            ProbeExecutionStatus.NOT_APPLICABLE,
            ProbeExecutionStatus.ABSTAINED,
            ProbeExecutionStatus.UNAVAILABLE,
            ProbeExecutionStatus.ERROR,
        }:
            status_groups.setdefault(witness.question_identity, []).append(witness)
            add(
                f"verifier execution was {witness.execution_status.value}",
                None,
                (witness.witness_identity,),
                candidate_kind="verifier-abstention-or-error",
                priority=80,
            )
    for failed in status_groups.values():
        if len(failed) >= 2:
            add(
                "question has repeated unavailable or ineligible diagnostic outcomes",
                None,
                tuple(item.witness_identity for item in failed),
                candidate_kind="repeated-unknown",
                priority=85,
            )
    witness_snapshot_ids = {
        identity for witness in witnesses for identity in witness.snapshot_identities
    }
    for snapshot_identity in snapshots.identities():
        if snapshot_identity not in witness_snapshot_ids:
            snapshot = snapshots.get(snapshot_identity)
            add(
                "identified snapshot has no compatible verifier execution",
                snapshot.snapshot_type,
                (snapshot_identity,),
                candidate_kind="snapshot-unprobed",
                priority=78,
            )
    for mutation in mutations:
        result_identity = mutation.resulting_snapshot_identity
        if not any(result_identity in witness.snapshot_identities for witness in witnesses):
            add(
                "registered mutation has no corresponding counterfactual probe",
                None,
                (mutation.mutation_identity, mutation.original_snapshot_identity),
                candidate_kind="missing-counterfactual-probe",
                priority=88,
            )
        if mutation.original_snapshot_identity not in witness_snapshot_ids:
            add(
                "registered mutation has no identified original probe",
                None,
                (mutation.mutation_identity,),
                candidate_kind="missing-original-probe",
                priority=82,
            )
    mutated_originals = {item.original_snapshot_identity for item in mutations}
    for witness in witnesses:
        if witness.execution_status is ProbeExecutionStatus.COMPLETED and not any(
            identity in mutated_originals for identity in witness.snapshot_identities
        ):
            add(
                "completed original probe has no registered mutation counterpart",
                None,
                (witness.witness_identity,),
                candidate_kind="missing-mutation-counterpart",
                priority=60,
            )
    for event in learned_observations:
        learned_value = event.payload.get("condition_observed")
        for witness in witnesses:
            if (
                witness.execution_status is ProbeExecutionStatus.COMPLETED
                and any(identity in event.snapshot_identities for identity in witness.snapshot_identities)
                and isinstance(learned_value, bool)
                and learned_value != witness.diagnostic_output.get("condition_observed")
            ):
                add(
                    "learned-provider observation disagrees with a deterministic witness",
                    None,
                    (event.event_identity, witness.witness_identity),
                    candidate_kind="learned-diagnostic-disagreement",
                    priority=95,
                )
    if registry is not None:
        for declaration in registry.declarations():
            if registry.state(declaration.verifier_id) is VerifierState.QUARANTINED:
                add(
                    "quarantined verifier creates a diagnostic coverage hole",
                    declaration.accepted_snapshot_types[0],
                    (declaration.declaration_identity,),
                    candidate_kind="verifier-health-hole",
                    priority=75,
                )
    values = sorted(candidates.values(), key=lambda item: (-item.priority, item.candidate_identity))
    return tuple(values[:max_candidates])


def run_reference_forge_study(workspace: str | Path | None = None) -> dict[str, Any]:
    identities = {
        "producer_identity": "mnel-reference-producer/0.3",
        "source_identity": "sha256:reference-source",
        "dependency_identity": "sha256:reference-dependency",
        "feature_extractor_identity": "sha256:reference-extractor",
    }
    store = SnapshotStore()
    transition = transition_snapshot(b"cold", b"warm", **identities)
    table = tabular_snapshot(((0.2, 0.4), (0.3, 0.5)), **identities)
    trace = trace_snapshot((("start", b""), ("finish", b"")), **identities)
    graph = graph_snapshot(((1, "source"), (2, "target")), ((1, 2, "calls"),), **identities)
    for snapshot in (transition, table, trace, graph):
        store.register(snapshot)
    registry = reference_verifier_registry()
    health = VerifierHealthStore()
    runtime = ReferenceForgeRuntime(store, registry, health)
    base = {
        "subject_identities": {"source": identities["source_identity"]},
        "snapshot_identities": (transition.snapshot_identity,),
        "preconditions": (),
        "resource_budget": {"operation_limit": 100, "wall_time_ms": 1000, "output_bytes": 4096},
        "mutation_policy": MutationPolicy.REGISTERED_ONLY,
        "runtime_identity": {"runtime": "mnel-reference-runtime/0.3"},
        "lineage": {"investigator_request": "sha256:reference-investigator-request"},
        "parameters": {},
    }
    witness_a = runtime.execute(ProbeRequest("probe-transition-a", "did the identified transition change state?", verifier_id="transition-change", expected_witness_type="transition-witness", **base))
    witness_b = runtime.execute(ProbeRequest("probe-transition-b", "did the identified transition change state?", verifier_id="transition-change-independent", expected_witness_type="transition-witness", **base))
    table_witness = runtime.execute(ProbeRequest("probe-table", "are all table values bounded?", subject_identities=base["subject_identities"], verifier_id="tabular-bounds", snapshot_identities=(table.snapshot_identity,), expected_witness_type="tabular-witness", preconditions=(Precondition("minimum_rows", 2),), resource_budget=base["resource_budget"], mutation_policy=base["mutation_policy"], runtime_identity=base["runtime_identity"], lineage=base["lineage"], parameters={"minimum": 0.0, "maximum": 1.0}))
    comparison = compare_witnesses((witness_a, witness_b), base["subject_identities"])
    mutation = MutationRegistry().apply("trace.swap-events", trace, {"index": 0, "second_index": 1}, store)
    mutation_request = ProbeRequest(
        "probe-trace-mutation",
        "did the identified trace preserve the requested event order after mutation?",
        subject_identities=base["subject_identities"],
        verifier_id="trace-order",
        snapshot_identities=(mutation.resulting_snapshot_identity,),
        expected_witness_type="trace-witness",
        preconditions=(),
        resource_budget=base["resource_budget"],
        mutation_policy=base["mutation_policy"],
        runtime_identity=base["runtime_identity"],
        lineage=base["lineage"],
        parameters={"before": "start", "after": "finish"},
    )
    mutation_witness = runtime.execute(mutation_request, mutation_identity=mutation.mutation_identity)
    learned = LearnedDiagnosticEvent.from_observation({"provider_id": "state.hidden-markov-model", "observation_identity": "sha256:learned-observation", "snapshot_ids": [transition.snapshot_identity], "declaration_identity": "sha256:learned-declaration", "value": 0.8, "out_of_distribution": False})
    witnesses = (witness_a, witness_b, table_witness, mutation_witness)
    coverage = build_coverage(registry, store, witnesses)
    candidates = discover_question_candidates(base["subject_identities"]["source"], store, coverage, (comparison,))
    records: list[dict[str, Any]] = [
        {"record_type": "diagnostic-snapshot", **snapshot.to_dict()} for snapshot in (transition, table, trace, graph, mutation_record_snapshot(store, mutation.resulting_snapshot_identity))
    ]
    records.extend({"record_type": "verifier-witness", **witness.to_dict()} for witness in witnesses)
    records.extend(({"record_type": "witness-comparison", **comparison.to_dict()}, {"record_type": "mutation", **mutation.to_dict()}, {"record_type": "learned-provider-observation", **learned.to_dict()}, {"record_type": "verifier-coverage", **coverage.to_dict()}))
    records.extend({"record_type": "omitted-question-candidate", **candidate.to_dict()} for candidate in candidates)
    result: dict[str, Any] = {"schema": "mnel-forge-reference-study/0.3", "records": records, "comparison": comparison.to_dict(), "mutation": mutation.to_dict(), "coverage": coverage.to_dict(), "health": [health.to_dict(item.verifier_id) for item in registry.declarations()], "question_candidates": [item.to_dict() for item in candidates], "authority": AUTHORITY_DIAGNOSTIC_ONLY, "study_identity": canonical_digest(records)}
    if workspace is not None:
        root = Path(workspace)
        ledger = EvidenceLedger(root / "forge-evidence.jsonl")
        for record in records:
            ledger.append(record["record_type"], record, actor="mnel-reference-verifier")
        result["ledger"] = ledger.summarize()
    return result


def mutation_record_snapshot(store: SnapshotStore, identity: str) -> DiagnosticSnapshot:
    return store.get(identity)
