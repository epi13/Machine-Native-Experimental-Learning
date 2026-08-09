"""Bounded distributed MNEL workloads on the public MNCS Fabric boundary.

This module deliberately models coarse-grained, identified work.  Fabric chooses
an execution node; the provider runtime still chooses CPU/CUDA/offload policy on
that node.  The in-process reference path uses the current public Fabric
controller/worker API when the sibling package is installed, while the network
adapter only accepts operator-supplied TLS configuration and pre-staged bundles.

Execution observations remain evidence.  They do not become evaluator verdicts,
conformance claims, or promotion decisions merely because Fabric completed them.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import tomllib
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Sequence

from .core import EvidenceLedger, Visibility, canonical_digest, canonical_json
from .distillation import StudyDataAccess, StudyRecord, VisibilityViolation
from .provider_study import PortfolioCase
from .reference_provider import TabularCentroidModel, TransitionFrequencyModel
from .snapshots import TabularView, TransitionView, decode_snapshot


AUTHORITY = "diagnostic-only"
MAX_WORKLOADS = 512
MAX_DEPENDENCIES = 64
MAX_SHARDS = 256
MAX_OUTPUT_BYTES = 256 * 1024


class DistributedExecutionError(ValueError):
    pass


class WorkloadClass(StrEnum):
    EXPERT_INFERENCE = "expert-inference"
    MICRO_PROVIDER_RUN = "micro-provider-run"
    STUDY_CASE = "study-case"
    CALIBRATION = "calibration"
    TRAINING_SHARD = "training-shard"
    EVALUATION_SUPPORT = "evaluation-support"
    COUNTERFACTUAL_PROBE = "counterfactual-probe"
    REPLICATION = "replication"


class ExecutionDisposition(StrEnum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class ExpertPlacementRequest:
    """Capability request for an expert, separate from its worker placement."""

    provider_id: str
    provider_artifact_identity: str
    model_identity: str
    calibration_identity: str
    architecture_family: str
    supported_snapshot_types: tuple[str, ...]
    model_size_bytes: int
    required_capabilities: tuple[str, ...] = ("python",)
    execution_device: str = "auto"
    offload_policy: str = "auto"
    minimum_ram_bytes: int = 0
    maximum_vram_bytes: int | None = None
    estimated_operations: int = 0
    runtime_class: str = "cpu-reference"
    locality_hint: str | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.provider_id, "provider_id")
        _identity(self.provider_artifact_identity, "provider_artifact_identity")
        _identity(self.model_identity, "model_identity")
        _identity(self.calibration_identity, "calibration_identity")
        if (
            not self.supported_snapshot_types
            or self.model_size_bytes < 1
            or self.minimum_ram_bytes < 0
            or self.estimated_operations < 0
        ):
            raise DistributedExecutionError("expert placement request has invalid bounds")
        if self.maximum_vram_bytes is not None and self.maximum_vram_bytes < 1:
            raise DistributedExecutionError("maximum_vram_bytes must be positive when supplied")
        if self.execution_device not in {"auto", "cpu", "cuda"} or self.offload_policy not in {
            "auto",
            "none",
            "sequential-cpu",
        }:
            raise DistributedExecutionError("expert execution device or offload policy is invalid")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            raise DistributedExecutionError("expert placement capabilities must be unique")

    @property
    def placement_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-expert-placement-request/0.5",
            "provider_id": self.provider_id,
            "provider_artifact_identity": self.provider_artifact_identity,
            "model_identity": self.model_identity,
            "calibration_identity": self.calibration_identity,
            "architecture_family": self.architecture_family,
            "supported_snapshot_types": list(self.supported_snapshot_types),
            "model_size_bytes": self.model_size_bytes,
            "required_capabilities": list(self.required_capabilities),
            "execution_device": self.execution_device,
            "offload_policy": self.offload_policy,
            "minimum_ram_bytes": self.minimum_ram_bytes,
            "maximum_vram_bytes": self.maximum_vram_bytes,
            "estimated_operations": self.estimated_operations,
            "runtime_class": self.runtime_class,
            "locality_hint": self.locality_hint,
            "authority": AUTHORITY,
            "semantics": "worker capability request; not-a-verdict",
        }
        if include_identity:
            value["placement_identity"] = self.placement_identity
        return value


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise DistributedExecutionError(f"{label} must be a sha256 identity")
    return value


def _external_identity(value: object, label: str) -> str:
    if isinstance(value, str) and (
        (value.startswith("sha256:") and len(value) == 71)
        or (len(value) == 64 and all(char in "0123456789abcdef" for char in value))
    ):
        return value
    raise DistributedExecutionError(f"{label} must be a supported external identity")


def _bounded_text(value: object, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise DistributedExecutionError(f"{label} must be bounded text")
    return value


def _reject_authority(value: Any) -> None:
    forbidden = {
        "verdict",
        "evaluator_verdict",
        "conformance",
        "mncs_conformance",
        "mncds_conformance",
        "promotion",
        "promotion_authorized",
        "evaluator_authority",
        "ravel_promotion",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden and child not in (
                None,
                False,
                "UNKNOWN",
                "not-asserted",
            ):
                raise DistributedExecutionError(
                    f"distributed record expands authority through {key}"
                )
            _reject_authority(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_authority(child)


@dataclass(frozen=True, slots=True)
class DistributedWorkload:
    study_identity: str
    experiment_identity: str
    workload_class: WorkloadClass
    provider_id: str | None = None
    expert_placement: ExpertPlacementRequest | None = None
    provider_artifact_identity: str | None = None
    model_identity: str | None = None
    calibration_identity: str | None = None
    snapshot_identities: tuple[str, ...] = ()
    dataset_identity: str | None = None
    required_capabilities: tuple[str, ...] = ("python",)
    resource_budget: dict[str, int | float] = field(
        default_factory=lambda: {"operations": 1000, "wall_seconds": 30}
    )
    replication_count: int = 1
    shard_index: int | None = None
    shard_count: int | None = None
    visibility: Visibility = Visibility.DEVELOPMENT
    seed: int = 0
    forge_workflow_identity: str = "mnel-fabric-study"
    expected_output_kind: str = "diagnostic-observation"
    authority: str = AUTHORITY

    def __post_init__(self) -> None:
        _identity(self.study_identity, "study_identity")
        _identity(self.experiment_identity, "experiment_identity")
        if self.provider_artifact_identity is not None:
            _identity(self.provider_artifact_identity, "provider_artifact_identity")
        if (
            self.expert_placement is not None
            and self.provider_artifact_identity != self.expert_placement.provider_artifact_identity
        ):
            raise DistributedExecutionError(
                "workload and expert placement artifact identities differ"
            )
        if self.model_identity is not None:
            _identity(self.model_identity, "model_identity")
        if self.calibration_identity is not None:
            _identity(self.calibration_identity, "calibration_identity")
        if not self.snapshot_identities and self.dataset_identity is None:
            raise DistributedExecutionError("workload requires snapshot or dataset identity")
        for value in self.snapshot_identities:
            _identity(value, "snapshot_identity")
        if self.dataset_identity is not None:
            _identity(self.dataset_identity, "dataset_identity")
        if len(set(self.required_capabilities)) != len(self.required_capabilities) or not all(
            self.required_capabilities
        ):
            raise DistributedExecutionError(
                "workload capabilities must be unique non-empty strings"
            )
        if self.replication_count < 1 or self.replication_count > 64 or self.seed < 0:
            raise DistributedExecutionError("workload replication/seed bounds are invalid")
        if (self.shard_index is None) != (self.shard_count is None):
            raise DistributedExecutionError("shard index and count must be supplied together")
        if self.shard_count is not None and not (
            0 <= self.shard_index < self.shard_count <= MAX_SHARDS
        ):
            raise DistributedExecutionError("training shard bounds are invalid")
        if self.visibility in {Visibility.TRANSFER_HIDDEN, Visibility.FUTURE_FINAL}:
            raise VisibilityViolation(
                "distributed development execution cannot receive hidden or future-final data"
            )
        if self.authority != AUTHORITY:
            raise DistributedExecutionError("distributed workloads are diagnostic-only")
        _reject_authority(self.resource_budget)

    @property
    def workload_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-distributed-workload/0.5",
            "study_identity": self.study_identity,
            "experiment_identity": self.experiment_identity,
            "workload_class": self.workload_class.value,
            "provider_id": self.provider_id,
            "expert_placement": self.expert_placement.to_dict() if self.expert_placement else None,
            "provider_artifact_identity": self.provider_artifact_identity,
            "model_identity": self.model_identity,
            "calibration_identity": self.calibration_identity,
            "snapshot_identities": list(self.snapshot_identities),
            "dataset_identity": self.dataset_identity,
            "required_capabilities": list(self.required_capabilities),
            "resource_budget": dict(self.resource_budget),
            "replication_count": self.replication_count,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "visibility": self.visibility.value,
            "seed": self.seed,
            "forge_workflow_identity": self.forge_workflow_identity,
            "expected_output_kind": self.expected_output_kind,
            "authority": self.authority,
            "semantics": "distributed-execution-request; observation-only; not-a-verdict",
        }
        if include_identity:
            value["workload_identity"] = self.workload_identity
        return value


@dataclass(frozen=True, slots=True)
class WorkloadNode:
    workload: DistributedWorkload
    dependencies: tuple[str, ...] = ()
    node_id: str = ""

    def __post_init__(self) -> None:
        if len(self.dependencies) > MAX_DEPENDENCIES or len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise DistributedExecutionError("workload dependencies are invalid")
        if self.node_id and self.node_id != self.content_identity:
            raise DistributedExecutionError("workload node identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(
            {"workload": self.workload.workload_identity, "dependencies": list(self.dependencies)}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnel-workload-node/0.5",
            "node_id": self.node_id or self.content_identity,
            "workload": self.workload.to_dict(),
            "dependencies": list(self.dependencies),
            "authority": AUTHORITY,
        }


@dataclass(frozen=True, slots=True)
class WorkloadGraph:
    nodes: tuple[WorkloadNode, ...]
    graph_identity: str = ""

    def __post_init__(self) -> None:
        if not self.nodes or len(self.nodes) > MAX_WORKLOADS:
            raise DistributedExecutionError("workload graph is empty or exceeds its bound")
        ids = [node.node_id or node.content_identity for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise DistributedExecutionError("workload graph contains duplicate nodes")
        available = set(ids)
        for node in self.nodes:
            if any(dep not in available for dep in node.dependencies):
                raise DistributedExecutionError("workload graph contains a missing dependency")
        self.topological_order()
        if self.graph_identity and self.graph_identity != self.content_identity:
            raise DistributedExecutionError("workload graph identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(
            {
                "schema": "mnel-workload-graph/0.5",
                "nodes": [
                    node.to_dict()
                    for node in sorted(
                        self.nodes, key=lambda item: item.node_id or item.content_identity
                    )
                ],
            }
        )

    def topological_order(self) -> tuple[str, ...]:
        nodes = {node.node_id or node.content_identity: node for node in self.nodes}
        remaining = set(nodes)
        ordered: list[str] = []
        while remaining:
            ready = sorted(
                item for item in remaining if set(nodes[item].dependencies).issubset(set(ordered))
            )
            if not ready:
                raise DistributedExecutionError("workload graph contains a cycle")
            ordered.extend(ready)
            remaining.difference_update(ready)
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnel-workload-graph/0.5",
            "graph_identity": self.graph_identity or self.content_identity,
            "nodes": [node.to_dict() for node in self.nodes],
            "topological_order": list(self.topological_order()),
            "authority": AUTHORITY,
        }


@dataclass(frozen=True, slots=True)
class DistributedStudyCell:
    workload: DistributedWorkload
    control: str
    repetition: int
    cell_identity: str = ""

    def __post_init__(self) -> None:
        if not _bounded_text(self.control, "control", 128) or self.repetition < 0:
            raise DistributedExecutionError("study cell control or repetition is invalid")
        if self.cell_identity and self.cell_identity != self.content_identity:
            raise DistributedExecutionError("study cell identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(
            {
                "schema": "mnel-distributed-study-cell/0.5",
                "workload_identity": self.workload.workload_identity,
                "control": self.control,
                "repetition": self.repetition,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnel-distributed-study-cell/0.5",
            "cell_identity": self.cell_identity or self.content_identity,
            "workload": self.workload.to_dict(),
            "control": self.control,
            "repetition": self.repetition,
            "authority": AUTHORITY,
        }


@dataclass(frozen=True, slots=True)
class DistributedStudyMatrix:
    cells: tuple[DistributedStudyCell, ...]
    matrix_identity: str = ""

    def __post_init__(self) -> None:
        if not self.cells or len(self.cells) > MAX_WORKLOADS:
            raise DistributedExecutionError("study matrix is empty or exceeds its bound")
        if len({cell.cell_identity or cell.content_identity for cell in self.cells}) != len(
            self.cells
        ):
            raise DistributedExecutionError("study matrix contains duplicate cells")
        if self.matrix_identity and self.matrix_identity != self.content_identity:
            raise DistributedExecutionError("study matrix identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(
            {
                "schema": "mnel-distributed-study-matrix/0.5",
                "cells": [
                    cell.to_dict()
                    for cell in sorted(
                        self.cells, key=lambda item: item.cell_identity or item.content_identity
                    )
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnel-distributed-study-matrix/0.5",
            "matrix_identity": self.matrix_identity or self.content_identity,
            "cells": [cell.to_dict() for cell in self.cells],
            "authority": AUTHORITY,
            "semantics": "distributed-study-matrix; not-a-verdict",
        }


@dataclass(frozen=True, slots=True)
class TrainingShard:
    dataset_identity: str
    partition_method: str
    partition_seed: int
    shard_index: int
    shard_count: int
    source_record_ids: tuple[str, ...]
    provider_family: str
    feature_extractor_identity: str
    training_code_identity: str
    visibility: Visibility = Visibility.DEVELOPMENT
    shard_identity: str = ""

    def __post_init__(self) -> None:
        _identity(self.dataset_identity, "dataset_identity")
        _identity(self.feature_extractor_identity, "feature_extractor_identity")
        if (
            self.shard_count < 1
            or self.shard_count > MAX_SHARDS
            or not 0 <= self.shard_index < self.shard_count
        ):
            raise DistributedExecutionError("training shard index/count is invalid")
        if not self.source_record_ids or len(set(self.source_record_ids)) != len(
            self.source_record_ids
        ):
            raise DistributedExecutionError("training shard requires unique source records")
        if self.visibility != Visibility.DEVELOPMENT:
            raise VisibilityViolation("training shards must be development-visible")
        if self.shard_identity and self.shard_identity != self.content_identity:
            raise DistributedExecutionError("training shard identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-training-shard/0.5",
            "dataset_identity": self.dataset_identity,
            "partition_method": self.partition_method,
            "partition_seed": self.partition_seed,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "source_record_ids": list(self.source_record_ids),
            "provider_family": self.provider_family,
            "feature_extractor_identity": self.feature_extractor_identity,
            "training_code_identity": self.training_code_identity,
            "visibility": self.visibility.value,
            "authority": AUTHORITY,
            "semantics": "identified-training-shard; not-a-verdict",
        }
        if include_identity:
            value["shard_identity"] = self.shard_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class PartialTrainingArtifact:
    provider_family: str
    dataset_identity: str
    shard_identity: str
    shard_index: int
    feature_extractor_identity: str
    training_code_identity: str
    source_record_ids: tuple[str, ...]
    transition_counts: tuple[tuple[str, int], ...] = ()
    row_count: int = 0
    feature_sums: tuple[float, ...] = ()
    feature_squared_sums: tuple[float, ...] = ()
    artifact_identity: str = ""

    def __post_init__(self) -> None:
        _identity(self.dataset_identity, "dataset_identity")
        _identity(self.shard_identity, "shard_identity")
        _identity(self.feature_extractor_identity, "feature_extractor_identity")
        if not self.source_record_ids or any(count < 1 for _, count in self.transition_counts):
            raise DistributedExecutionError("partial training artifact has invalid counts")
        if self.provider_family == "nearest-centroid" and (
            self.row_count < 1 or len(self.feature_sums) != len(self.feature_squared_sums)
        ):
            raise DistributedExecutionError("centroid partial artifact statistics are invalid")
        if self.artifact_identity and self.artifact_identity != self.content_identity:
            raise DistributedExecutionError("partial artifact identity does not match content")

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "mnel-partial-training-artifact/0.5",
            "provider_family": self.provider_family,
            "dataset_identity": self.dataset_identity,
            "shard_identity": self.shard_identity,
            "shard_index": self.shard_index,
            "feature_extractor_identity": self.feature_extractor_identity,
            "training_code_identity": self.training_code_identity,
            "source_record_ids": list(self.source_record_ids),
            "transition_counts": [[key, count] for key, count in self.transition_counts],
            "row_count": self.row_count,
            "feature_sums": list(self.feature_sums),
            "feature_squared_sums": list(self.feature_squared_sums),
            "authority": AUTHORITY,
            "semantics": "partial-sufficient-statistic; not-a-verdict",
        }
        if include_identity:
            value["artifact_identity"] = self.artifact_identity or self.content_identity
        return value


@dataclass(frozen=True, slots=True)
class FabricExecutionObservation:
    workload_identity: str
    worker_identity: str
    provider_artifact_identity: str | None
    disposition: ExecutionDisposition
    result: dict[str, Any] | None
    fabric_record_identity: str | None = None
    receipt_identity: str | None = None
    reason: str | None = None
    observation_identity: str = ""

    def __post_init__(self) -> None:
        _identity(self.workload_identity, "workload_identity")
        _bounded_text(self.worker_identity, "worker_identity")
        if self.provider_artifact_identity is not None:
            _identity(self.provider_artifact_identity, "provider_artifact_identity")
        if self.fabric_record_identity is not None:
            _identity(self.fabric_record_identity, "fabric_record_identity")
        if self.receipt_identity is not None:
            _external_identity(self.receipt_identity, "receipt_identity")
        _reject_authority(self.result or {})

    @property
    def content_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-fabric-execution-observation/0.5",
            "workload_identity": self.workload_identity,
            "worker_identity": self.worker_identity,
            "provider_artifact_identity": self.provider_artifact_identity,
            "disposition": self.disposition.value,
            "result": self.result,
            "fabric_record_identity": self.fabric_record_identity,
            "receipt_identity": self.receipt_identity,
            "reason": self.reason,
            "authority": AUTHORITY,
            "semantics": "Fabric execution observation; not correctness or evaluator authority",
        }
        if include_identity:
            value["observation_identity"] = self.observation_identity or self.content_identity
        return value


class ExecutionBackend(Protocol):
    backend_identity: str

    def execute_provider(
        self,
        workload: DistributedWorkload,
        provider: object,
        case: PortfolioCase,
        *,
        replicas: int = 1,
    ) -> tuple[FabricExecutionObservation, ...]: ...


def _transition_key(previous: bytes, following: bytes) -> str:
    return hashlib.sha256(previous + b"\x00" + following).hexdigest()


def partition_records(
    access: StudyDataAccess, *, shard_count: int, seed: int = 0
) -> tuple[tuple[StudyRecord, ...], ...]:
    if shard_count < 1 or shard_count > MAX_SHARDS:
        raise DistributedExecutionError("shard_count is outside the bounded range")
    records = list(access.records())
    if not records:
        raise DistributedExecutionError("cannot partition an empty access view")
    if any(
        record.visibility not in {Visibility.DEVELOPMENT, Visibility.SELECTION_OBSERVED}
        for record in records
    ):
        raise VisibilityViolation("partition received non-development data")
    ordered = sorted(
        records, key=lambda item: canonical_digest({"seed": seed, "record": item.identity})
    )
    return tuple(tuple(ordered[index::shard_count]) for index in range(shard_count))


def transition_training_partial(
    access: StudyDataAccess,
    snapshots: Any,
    shard: TrainingShard,
    *,
    record_type: str = "experience-episode",
) -> PartialTrainingArtifact:
    if access.purpose != "development-study" or any(
        item.visibility != Visibility.DEVELOPMENT for item in access.records()
    ):
        raise VisibilityViolation("transition shard training requires development-only data")
    selected = {item.identity: item for item in access.records(record_type)}
    if set(shard.source_record_ids) != set(selected):
        raise DistributedExecutionError("access view and training shard source identities differ")
    counts: dict[str, int] = {}
    for record in selected.values():
        snapshot = snapshots.get(record.payload["snapshot_identity"])
        view = decode_snapshot(snapshot)
        if not isinstance(view, TransitionView):
            raise DistributedExecutionError("transition shard received a non-transition snapshot")
        key = _transition_key(view.previous_state, view.next_state)
        counts[key] = counts.get(key, 0) + 1
    return PartialTrainingArtifact(
        "transition-frequency",
        shard.dataset_identity,
        shard.shard_identity or shard.content_identity,
        shard.shard_index,
        shard.feature_extractor_identity,
        shard.training_code_identity,
        tuple(sorted(selected)),
        tuple(sorted(counts.items())),
    )


def centroid_training_partial(
    access: StudyDataAccess,
    snapshots: Any,
    shard: TrainingShard,
    *,
    record_type: str = "experience-episode",
) -> PartialTrainingArtifact:
    if access.purpose != "development-study" or any(
        item.visibility != Visibility.DEVELOPMENT for item in access.records()
    ):
        raise VisibilityViolation("centroid shard training requires development-only data")
    selected = {item.identity: item for item in access.records(record_type)}
    if set(shard.source_record_ids) != set(selected):
        raise DistributedExecutionError("access view and training shard source identities differ")
    rows: list[tuple[float, ...]] = []
    for record in selected.values():
        view = decode_snapshot(snapshots.get(record.payload["snapshot_identity"]))
        if not isinstance(view, TabularView):
            raise DistributedExecutionError("centroid shard received a non-tabular snapshot")
        rows.extend(view.rows)
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise DistributedExecutionError("centroid shard dimensions are inconsistent")
    sums = tuple(sum(row[index] for row in rows) for index in range(len(rows[0])))
    squared = tuple(sum(row[index] * row[index] for row in rows) for index in range(len(rows[0])))
    return PartialTrainingArtifact(
        "nearest-centroid",
        shard.dataset_identity,
        shard.shard_identity or shard.content_identity,
        shard.shard_index,
        shard.feature_extractor_identity,
        shard.training_code_identity,
        tuple(sorted(selected)),
        row_count=len(rows),
        feature_sums=sums,
        feature_squared_sums=squared,
    )


def aggregate_transition_partials(
    partials: Sequence[PartialTrainingArtifact],
    *,
    dataset_identity: str,
    feature_extractor_identity: str,
    training_code_identity: str,
    training_record_ids: Sequence[str],
    calibration_identity: str,
    calibration_dataset_identity: str = "",
) -> TransitionFrequencyModel:
    _validate_partials(
        partials,
        "transition-frequency",
        dataset_identity,
        feature_extractor_identity,
        training_code_identity,
        training_record_ids,
    )
    counts: dict[str, int] = {}
    for partial in sorted(
        partials,
        key=lambda item: (item.shard_index, item.artifact_identity or item.content_identity),
    ):
        for key, count in partial.transition_counts:
            counts[key] = counts.get(key, 0) + count
    total = sum(counts.values())
    model = TransitionFrequencyModel(
        dataset_identity,
        tuple(sorted(training_record_ids)),
        feature_extractor_identity,
        training_code_identity,
        calibration_identity,
        counts,
        total,
        calibration_dataset_identity=calibration_dataset_identity,
    )
    object.__setattr__(model, "model_identity", model.content_identity)
    object.__setattr__(
        model,
        "artifact_identity",
        "sha256:" + hashlib.sha256(canonical_json(model.to_dict())).hexdigest(),
    )
    return model


def aggregate_centroid_partials(
    partials: Sequence[PartialTrainingArtifact],
    *,
    dataset_identity: str,
    feature_extractor_identity: str,
    training_code_identity: str,
    training_record_ids: Sequence[str],
    calibration: Sequence[tuple[Sequence[float], int]],
    calibration_identity: str,
    calibration_dataset_identity: str,
) -> TabularCentroidModel:
    _validate_partials(
        partials,
        "nearest-centroid",
        dataset_identity,
        feature_extractor_identity,
        training_code_identity,
        training_record_ids,
    )
    row_count = sum(item.row_count for item in partials)
    dimensions = len(partials[0].feature_sums)
    sums = [sum(item.feature_sums[index] for item in partials) for index in range(dimensions)]
    squared = [
        sum(item.feature_squared_sums[index] for item in partials) for index in range(dimensions)
    ]
    centroid = tuple(value / row_count for value in sums)
    scales = tuple(
        max(math.sqrt(max(0.0, squared[index] / row_count - centroid[index] ** 2)), 1e-9)
        for index in range(dimensions)
    )
    draft = TabularCentroidModel(
        dataset_identity,
        tuple(sorted(training_record_ids)),
        calibration_dataset_identity,
        feature_extractor_identity,
        training_code_identity,
        calibration_identity,
        centroid,
        scales,
        1.0,
    )
    distances = []
    for values, label in calibration:
        if label:
            distances.append(
                draft._distance(
                    type(
                        "View",
                        (),
                        {"column_count": len(values), "row_count": 1, "rows": (tuple(values),)},
                    )()
                )
            )
    if not distances:
        raise DistributedExecutionError(
            "centroid aggregation requires a positive calibration example"
        )
    model = TabularCentroidModel(
        dataset_identity,
        tuple(sorted(training_record_ids)),
        calibration_dataset_identity,
        feature_extractor_identity,
        training_code_identity,
        calibration_identity,
        centroid,
        scales,
        max(max(distances) * 1.5, 1e-6),
    )
    object.__setattr__(model, "model_identity", model.content_identity)
    object.__setattr__(
        model,
        "artifact_identity",
        "sha256:" + hashlib.sha256(canonical_json(model.to_dict())).hexdigest(),
    )
    return model


def _validate_partials(
    partials: Sequence[PartialTrainingArtifact],
    family: str,
    dataset_identity: str,
    feature: str,
    code: str,
    record_ids: Sequence[str],
) -> None:
    if not partials or len(partials) > MAX_SHARDS:
        raise DistributedExecutionError("partial artifact set is empty or too large")
    if any(
        item.provider_family != family
        or item.dataset_identity != dataset_identity
        or item.feature_extractor_identity != feature
        or item.training_code_identity != code
        for item in partials
    ):
        raise DistributedExecutionError(
            "partial artifact identity metadata does not match aggregation"
        )
    indices = [item.shard_index for item in partials]
    if len(set(indices)) != len(indices):
        raise DistributedExecutionError("duplicate training shard")
    seen: set[str] = set()
    for item in partials:
        overlap = seen.intersection(item.source_record_ids)
        if overlap:
            raise DistributedExecutionError("overlapping training shard records")
        seen.update(item.source_record_ids)
    if seen != set(record_ids):
        raise DistributedExecutionError("training aggregation has missing or extra source records")


def _task_source() -> str:
    return """import hashlib, json, math
v=json.load(open("input.json", encoding="utf-8"))
if v["provider_family"] == "transition-frequency":
    p=v["previous_hex"].encode(); n=v["next_hex"].encode(); key=hashlib.sha256(bytes.fromhex(v["previous_hex"])+b"\\x00"+bytes.fromhex(v["next_hex"])).hexdigest()
    count=int(v["transition_counts"].get(key,0)); total=int(v["total_count"])
    result={"provider_id":v["provider_id"],"model_identity":v["model_identity"],"provider_artifact_identity":v["provider_artifact_identity"],"snapshot_identity":v["snapshot_identity"],"score":max(0.0,min(1.0,count/total if count else 0.0)),"abstained":count==0,"out_of_distribution":count==0,"diagnostic_status":"abstained" if count==0 else "completed","authority":"diagnostic-only","semantics":"learned-provider-observation; not-a-verdict"}
else:
    values=v["rows"]; center=v["centroid"]; scales=v["scales"]; total=0.0; count=0
    for row in values:
        for x,c,s in zip(row,center,scales): total += ((x-c)/s)**2; count += 1
    distance=math.sqrt(total/max(1,count)); ood=distance > v["ood_distance_threshold"]
    result={"provider_id":v["provider_id"],"model_identity":v["model_identity"],"provider_artifact_identity":v["provider_artifact_identity"],"snapshot_identity":v["snapshot_identity"],"score":max(0.0,min(1.0,math.exp(-distance))),"abstained":ood,"out_of_distribution":ood,"diagnostic_status":"abstained" if ood else "completed","authority":"diagnostic-only","semantics":"learned-provider-observation; not-a-verdict"}
print(json.dumps(result, sort_keys=True, separators=(",",":")))
"""


def _provider_input(
    provider: object, case: PortfolioCase, workload: DistributedWorkload
) -> dict[str, Any]:
    if isinstance(provider, TransitionFrequencyModel):
        view = decode_snapshot(case.snapshot)
        if not isinstance(view, TransitionView):
            raise DistributedExecutionError("transition provider case is not a transition snapshot")
        return {
            "provider_family": "transition-frequency",
            "provider_id": provider.provider_id,
            "model_identity": provider.model_identity,
            "provider_artifact_identity": provider.artifact_identity,
            "snapshot_identity": case.snapshot.snapshot_identity,
            "previous_hex": view.previous_state.hex(),
            "next_hex": view.next_state.hex(),
            "transition_counts": provider.transition_counts,
            "total_count": provider.total_count,
        }
    if isinstance(provider, TabularCentroidModel):
        view = decode_snapshot(case.snapshot)
        if not isinstance(view, TabularView):
            raise DistributedExecutionError("centroid provider case is not a tabular snapshot")
        return {
            "provider_family": "nearest-centroid",
            "provider_id": provider.provider_id,
            "model_identity": provider.model_identity,
            "provider_artifact_identity": provider.artifact_identity,
            "snapshot_identity": case.snapshot.snapshot_identity,
            "rows": [list(row) for row in view.rows],
            "centroid": list(provider.centroid),
            "scales": list(provider.scales),
            "ood_distance_threshold": provider.ood_distance_threshold,
        }
    raise DistributedExecutionError(
        "reference Fabric backend supports only identified reference providers"
    )


class LocalFabricBackend:
    """Run identified provider tasks through Fabric's public local controller."""

    backend_identity = "mnel-fabric-local-public-controller/0.5"

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        try:
            from mncs_fabric.controller import LocalController
            from mncs_fabric.worker import LocalWorker
            from mncs_fabric.artifacts import build_manifest
            from mncs_fabric.receipts import build_execution_receipt
        except ImportError:
            # Ordinary MNEL CI is intentionally independent of sibling source trees.
            # The fallback preserves the same bounded workload/observation contract,
            # while the report states that Fabric itself was unavailable.
            self._fallback = True
            self.fabric_public_available = False
            self.backend_identity = "mnel-fabric-reference-fallback/0.5"
            return
        self._fallback = False
        self.fabric_public_available = True
        self.backend_identity = type(self).backend_identity
        self._LocalController = LocalController
        self._LocalWorker = LocalWorker
        self._build_manifest = build_manifest
        self._build_receipt = build_execution_receipt

    def execute_provider(
        self,
        workload: DistributedWorkload,
        provider: object,
        case: PortfolioCase,
        *,
        replicas: int = 1,
    ) -> tuple[FabricExecutionObservation, ...]:
        if provider.artifact_identity != workload.provider_artifact_identity:
            raise DistributedExecutionError("provider artifact does not match workload binding")
        if replicas < 1 or replicas > 8:
            raise DistributedExecutionError("replica count is outside the bounded range")
        if self._fallback:
            observations = []
            for index in range(replicas):
                try:
                    result = provider.infer(case.snapshot).to_dict()
                    result["provider_artifact_identity"] = workload.provider_artifact_identity
                    _reject_authority(result)
                    observations.append(
                        FabricExecutionObservation(
                            workload.workload_identity,
                            f"mnel-fallback-worker-{index}",
                            workload.provider_artifact_identity,
                            ExecutionDisposition.COMPLETED,
                            result,
                            canonical_digest(
                                {"workload": workload.workload_identity, "worker": index}
                            ),
                            None,
                        )
                    )
                except Exception as error:
                    observations.append(
                        FabricExecutionObservation(
                            workload.workload_identity,
                            f"mnel-fallback-worker-{index}",
                            workload.provider_artifact_identity,
                            ExecutionDisposition.INVALID,
                            None,
                            reason=str(error),
                        )
                    )
            return tuple(observations)
        with tempfile.TemporaryDirectory(
            prefix="mnel-fabric-reference-", dir=self.workspace if self.workspace.is_dir() else None
        ) as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "task.py").write_text(_task_source(), encoding="utf-8")
            (bundle / "input.json").write_text(
                canonical_json(_provider_input(provider, case, workload)).decode(), encoding="utf-8"
            )
            manifest = self._build_manifest(bundle)
            plan = {
                "schema_version": "mncs-fabric.job-plan.v0.1",
                "job_id": "mnel:" + workload.workload_identity[7:27],
                "candidate_identity": workload.workload_identity,
                "evaluator_identity": None,
                "artifact_manifest_identity": manifest["manifest_identity"],
                "argv": ["@python", "task.py"],
                "working_directory": ".",
                "timeout_seconds": float(workload.resource_budget.get("wall_seconds", 30)),
                "output_limit_bytes": min(MAX_OUTPUT_BYTES, 64 * 1024),
                "environment": {},
                "required_capabilities": list(workload.required_capabilities),
                "result_paths": [],
                "network_policy": "DECLARED_OFFLINE",
            }
            controller = self._LocalController("mnel-controller", root / "controller.jsonl")
            workers = []
            for index in range(replicas):
                worker = self._LocalWorker(
                    f"mnel-worker-{index}", bundle, root / f"worker-{index}.jsonl"
                )
                controller.register(worker)
                workers.append(worker.worker_id)
            responses = controller.dispatch(plan, manifest, replicas=replicas)
            observations: list[FabricExecutionObservation] = []
            for response in responses:
                if response.get("message_type") != "execution.result":
                    observations.append(
                        FabricExecutionObservation(
                            workload.workload_identity,
                            str(response.get("worker_id", "unknown")),
                            workload.provider_artifact_identity,
                            ExecutionDisposition.UNAVAILABLE,
                            None,
                            reason=str(
                                response.get("reason", "Fabric did not return an execution result")
                            ),
                        )
                    )
                    continue
                payload = response.get("payload", {})
                record = payload.get("record", {})
                try:
                    raw = record.get("stdout", {}).get("captured_utf8", "").strip().splitlines()[-1]
                    result = json.loads(raw)
                    _reject_authority(result)
                    if (
                        result.get("provider_artifact_identity")
                        != workload.provider_artifact_identity
                    ):
                        raise DistributedExecutionError("worker result provider artifact mismatch")
                    disposition = ExecutionDisposition.COMPLETED
                except (
                    KeyError,
                    IndexError,
                    TypeError,
                    json.JSONDecodeError,
                    DistributedExecutionError,
                ) as error:
                    result = None
                    disposition = ExecutionDisposition.INVALID
                    error_text = str(error)
                else:
                    error_text = None
                receipt = self._build_receipt(
                    record,
                    runner_identity=f"mncs-fabric-local-{record.get('node', {}).get('machine_label', 'unknown')}",
                    runner_version="0.2.0a0",
                )
                observations.append(
                    FabricExecutionObservation(
                        workload.workload_identity,
                        record.get("node", {}).get("machine_label", "unknown"),
                        workload.provider_artifact_identity,
                        disposition,
                        result,
                        record.get("record_id"),
                        receipt.get("receipt_identity"),
                        error_text,
                    )
                )
            return tuple(sorted(observations, key=lambda item: item.worker_identity))

    def execute_many(
        self, jobs: Sequence[tuple[DistributedWorkload, object, PortfolioCase]]
    ) -> tuple[FabricExecutionObservation, ...]:
        observations: list[FabricExecutionObservation] = []
        for workload, provider, case in sorted(jobs, key=lambda item: item[0].workload_identity):
            observations.extend(
                self.execute_provider(workload, provider, case, replicas=workload.replication_count)
            )
        return tuple(
            sorted(
                observations, key=lambda item: item.observation_identity or item.content_identity
            )
        )


@dataclass(frozen=True, slots=True)
class FabricWorkerConfig:
    worker_id: str
    host: str
    port: int
    capabilities: tuple[str, ...]
    ca_file: Path
    client_cert: Path
    client_key: Path
    trust_store: Path

    def validate(self) -> None:
        _bounded_text(self.worker_id, "worker_id")
        if not self.host or not 1 <= self.port <= 65535 or not self.capabilities:
            raise DistributedExecutionError("remote Fabric worker endpoint is incomplete")
        for path in (self.ca_file, self.client_cert, self.client_key, self.trust_store):
            if not path.is_file():
                raise DistributedExecutionError(
                    f"required Fabric trust material is unavailable: {path}"
                )


@dataclass(frozen=True, slots=True)
class FabricNetworkConfig:
    controller_id: str
    state_path: Path
    workers: tuple[FabricWorkerConfig, ...]
    pre_staged_bundle_identity: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "FabricNetworkConfig":
        source = Path(path).resolve(strict=True)
        try:
            raw = tomllib.loads(source.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise DistributedExecutionError(f"invalid Fabric configuration: {error}") from error
        base = source.parent
        workers = tuple(
            FabricWorkerConfig(
                str(item["worker_id"]),
                str(item["host"]),
                int(item["port"]),
                tuple(str(cap) for cap in item["capabilities"]),
                (base / str(item["ca_file"])).resolve(),
                (base / str(item["client_cert"])).resolve(),
                (base / str(item["client_key"])).resolve(),
                (base / str(item["trust_store"])).resolve(),
            )
            for item in raw.get("workers", [])
        )
        config = cls(
            str(raw.get("controller_id", "")),
            (base / str(raw.get("state_path", "fabric-controller.jsonl"))).resolve(),
            workers,
            raw.get("pre_staged_bundle_identity"),
        )
        if (
            not config.controller_id
            or not config.workers
            or len({item.worker_id for item in config.workers}) != len(config.workers)
        ):
            raise DistributedExecutionError(
                "Fabric configuration requires unique controller and workers"
            )
        for worker in workers:
            worker.validate()
        if config.pre_staged_bundle_identity is not None:
            _identity(config.pre_staged_bundle_identity, "pre_staged_bundle_identity")
        return config


class NetworkFabricBackend:
    """Thin adapter over NetworkController/TLSNetworkTransport.

    It dispatches only fixed-argv plans with a pre-staged bundle identity.  MNEL
    does not implement transfer, SSH staging, or a second transport protocol.
    """

    backend_identity = "mnel-fabric-network-public-controller/0.5"

    def __init__(self, config: FabricNetworkConfig) -> None:
        config_path = config.state_path.parent
        try:
            from mncs_fabric.controller import NetworkController
            from mncs_fabric.enrollment import TrustStore
            from mncs_fabric.transport import TLSNetworkTransport
        except ImportError as error:
            raise DistributedExecutionError(
                "mncs-fabric network public package is unavailable"
            ) from error
        self.config = config
        self.controller = NetworkController(config.controller_id, config.state_path)
        for worker in config.workers:
            trust = TrustStore(worker.trust_store)
            transport = TLSNetworkTransport(
                worker.host,
                worker.port,
                ca_file=worker.ca_file,
                client_cert=worker.client_cert,
                client_key=worker.client_key,
                expected_worker_id=worker.worker_id,
                trust_store=trust,
            )
            self.controller.register_remote(
                worker.worker_id, frozenset(worker.capabilities), transport
            )
        self._config_path = config_path

    def dispatch_plan(
        self, plan: dict[str, Any], manifest: dict[str, Any], *, replicas: int = 1
    ) -> list[dict[str, Any]]:
        if self.config.pre_staged_bundle_identity != manifest.get("manifest_identity"):
            raise DistributedExecutionError(
                "network Fabric requires an explicitly pre-staged matching bundle"
            )
        return self.controller.dispatch_remote(plan, manifest, replicas=replicas)


def run_network_fabric(
    config_path: str | Path,
    plan_path: str | Path,
    manifest_path: str | Path,
    *,
    replicas: int = 1,
) -> dict[str, Any]:
    """Dispatch an operator-supplied fixed-argv, pre-staged Fabric plan."""

    def load_json(path: str | Path, label: str) -> dict[str, Any]:
        source = Path(path).resolve(strict=True)
        if source.stat().st_size > MAX_OUTPUT_BYTES:
            raise DistributedExecutionError(f"{label} exceeds the bounded input size")
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DistributedExecutionError(f"{label} is not bounded JSON: {error}") from error
        if not isinstance(value, dict):
            raise DistributedExecutionError(f"{label} must contain an object")
        _reject_authority(value)
        return value

    backend = NetworkFabricBackend(FabricNetworkConfig.load(config_path))
    responses = backend.dispatch_plan(
        load_json(plan_path, "Fabric job plan"),
        load_json(manifest_path, "Fabric artifact manifest"),
        replicas=replicas,
    )
    for response in responses:
        _reject_authority(response)
    return {
        "schema": "mnel-fabric-network-dispatch/0.5",
        "backend_identity": backend.backend_identity,
        "response_count": len(responses),
        "responses": responses,
        "authority": AUTHORITY,
        "semantics": "remote Fabric execution observation; not-a-verdict",
        "limitations": [
            "bundle transfer is external; operator configuration must prove pre-staging",
            "TLS peer authentication does not establish worker honesty, independence, or conformance",
        ],
    }


def make_training_shards(
    access: StudyDataAccess,
    *,
    provider_family: str,
    feature_extractor_identity: str,
    training_code_identity: str,
    shard_count: int,
    seed: int = 0,
    record_type: str | None = None,
) -> tuple[tuple[StudyRecord, ...], tuple[TrainingShard, ...]]:
    if access.purpose != "development-study" or any(
        record.visibility != Visibility.DEVELOPMENT for record in access.records()
    ):
        raise VisibilityViolation("training shard materialization requires development-only access")
    source_records = access.records(record_type)
    if not source_records:
        raise DistributedExecutionError("training shard source record type is empty")
    if shard_count > len(source_records):
        raise DistributedExecutionError("shard count cannot exceed source record count")
    ordered = sorted(
        source_records, key=lambda item: canonical_digest({"seed": seed, "record": item.identity})
    )
    partitions = tuple(tuple(ordered[index::shard_count]) for index in range(shard_count))
    shards = tuple(
        TrainingShard(
            access.dataset_identity,
            "stable-sha256-round-robin",
            seed,
            index,
            shard_count,
            tuple(record.identity for record in partition),
            provider_family,
            feature_extractor_identity,
            training_code_identity,
        )
        for index, partition in enumerate(partitions)
    )
    return partitions, shards


def make_workload_graph(workloads: Sequence[DistributedWorkload]) -> WorkloadGraph:
    nodes = tuple(WorkloadNode(workload) for workload in workloads)
    return WorkloadGraph(nodes)


def make_study_matrix(
    workloads: Sequence[DistributedWorkload],
    *,
    controls: Sequence[str] = ("provider",),
    repetitions: int = 1,
) -> DistributedStudyMatrix:
    if not workloads or not controls or repetitions < 1 or repetitions > 64:
        raise DistributedExecutionError("study matrix bounds are invalid")
    cells = []
    for workload in workloads:
        for control in controls:
            for repetition in range(repetitions):
                cell_workload = replace(
                    workload,
                    seed=workload.seed + repetition,
                    experiment_identity=canonical_digest(
                        {
                            "parent": workload.experiment_identity,
                            "control": control,
                            "repetition": repetition,
                        }
                    ),
                )
                cells.append(DistributedStudyCell(cell_workload, control, repetition))
    return DistributedStudyMatrix(tuple(cells))


def run_reference_fabric_study(workspace: str | Path | None = None) -> dict[str, Any]:
    """Run a network-free, multi-logical-worker Fabric reference study."""
    from .distillation import make_study_record
    from .snapshots import SnapshotStore, tabular_snapshot, transition_snapshot

    snapshots = SnapshotStore()
    ids = {
        "producer_identity": "mnel-fabric-reference/0.5",
        "source_identity": canonical_digest({"source": "fabric-reference"}),
        "dependency_identity": canonical_digest({"dependency": "fabric-reference"}),
        "feature_extractor_identity": canonical_digest({"features": "fabric-reference"}),
    }
    transition = transition_snapshot(b"cold", b"warm", **ids)
    transition_2 = transition_snapshot(b"warm", b"hot", **ids)
    tabular = tabular_snapshot(((0.2, 0.4),), **ids)
    snapshots.register(transition)
    snapshots.register(transition_2)
    snapshots.register(tabular)
    records = (
        make_study_record(
            "transition-episode",
            {"snapshot_identity": transition.snapshot_identity, "snapshot_type": "transition"},
            record_identity="sha256:fabric-record-transition",
            epoch=1,
        ),
        make_study_record(
            "transition-episode",
            {"snapshot_identity": transition_2.snapshot_identity, "snapshot_type": "transition"},
            record_identity="sha256:fabric-record-transition-2",
            epoch=1,
        ),
        make_study_record(
            "tabular-episode",
            {"snapshot_identity": tabular.snapshot_identity, "snapshot_type": "tabular"},
            record_identity="sha256:fabric-record-tabular",
            epoch=1,
        ),
    )
    access = StudyDataAccess.development(records)
    transition_access = StudyDataAccess.development((records[0], records[1]))
    calibration_identity = canonical_digest({"name": "fabric-reference-calibration", "version": 1})
    calibration_dataset_identity = canonical_digest(
        {"name": "fabric-reference-calibration-data", "version": 1}
    )
    transition_model = TransitionFrequencyModel(
        transition_access.dataset_identity,
        (records[0].identity, records[1].identity),
        ids["feature_extractor_identity"],
        "mnel-fabric-training-code-transition/0.5",
        calibration_identity,
        {_transition_key(b"cold", b"warm"): 1, _transition_key(b"warm", b"hot"): 1},
        2,
        calibration_dataset_identity=calibration_dataset_identity,
    )
    object.__setattr__(transition_model, "model_identity", transition_model.content_identity)
    object.__setattr__(
        transition_model,
        "artifact_identity",
        "sha256:" + hashlib.sha256(transition_model.serialize()).hexdigest(),
    )
    centroid_model = TabularCentroidModel(
        access.dataset_identity,
        (records[1].identity,),
        calibration_dataset_identity,
        ids["feature_extractor_identity"],
        "mnel-fabric-training-code-centroid/0.5",
        calibration_identity,
        (0.2, 0.4),
        (1e-9, 1e-9),
        1.0,
    )
    object.__setattr__(centroid_model, "model_identity", centroid_model.content_identity)
    object.__setattr__(
        centroid_model,
        "artifact_identity",
        "sha256:" + hashlib.sha256(centroid_model.serialize()).hexdigest(),
    )
    study = canonical_digest(
        {
            "name": "mnel-fabric-reference",
            "providers": [transition_model.artifact_identity, centroid_model.artifact_identity],
            "dataset": access.dataset_identity,
        }
    )
    placement_requests = tuple(
        ExpertPlacementRequest(
            provider.provider_id,
            provider.artifact_identity,
            provider.model_identity,
            provider.calibration_identity,
            provider.provider_family,
            provider.supported_snapshot_types,
            provider.model_size_bytes,
            ("python",),
            "cpu",
            "none",
            runtime_class="python-reference",
            locality_hint=f"snapshot:{snapshot.snapshot_identity}",
        )
        for provider, snapshot in ((transition_model, transition), (centroid_model, tabular))
    )
    workloads = tuple(
        DistributedWorkload(
            study,
            canonical_digest({"study": study, "kind": "expert"}),
            WorkloadClass.EXPERT_INFERENCE,
            provider_id=provider.provider_id,
            expert_placement=placement,
            provider_artifact_identity=provider.artifact_identity,
            model_identity=provider.model_identity,
            calibration_identity=provider.calibration_identity,
            snapshot_identities=(snapshot.snapshot_identity,),
            resource_budget={"operations": 100, "wall_seconds": 10},
            replication_count=2,
        )
        for provider, snapshot, placement in zip(
            (transition_model, centroid_model),
            (transition, tabular),
            placement_requests,
        )
    )
    graph = make_workload_graph(workloads)
    backend = LocalFabricBackend(workspace or tempfile.gettempdir())
    matrix = make_study_matrix(
        workloads, controls=("provider", "replicated-control"), repetitions=1
    )
    observations: list[FabricExecutionObservation] = []
    for workload, provider, case in (
        (
            workloads[0],
            transition_model,
            PortfolioCase(
                "fabric-transition",
                transition,
                1,
                False,
                "in-distribution",
                "sha256:fabric-reference-transition",
            ),
        ),
        (
            workloads[1],
            centroid_model,
            PortfolioCase(
                "fabric-tabular",
                tabular,
                1,
                False,
                "in-distribution",
                "sha256:fabric-reference-tabular",
            ),
        ),
    ):
        observations.extend(backend.execute_provider(workload, provider, case, replicas=2))
    partitions, shards = make_training_shards(
        transition_access,
        provider_family="transition-frequency",
        feature_extractor_identity=ids["feature_extractor_identity"],
        training_code_identity="mnel-fabric-training-code-transition/0.5",
        shard_count=2,
        record_type="transition-episode",
    )
    transition_parts = tuple(
        transition_training_partial(
            StudyDataAccess.development(part), snapshots, shard, record_type="transition-episode"
        )
        for part, shard in zip(partitions, shards)
    )
    aggregate = aggregate_transition_partials(
        transition_parts,
        dataset_identity=transition_access.dataset_identity,
        feature_extractor_identity=ids["feature_extractor_identity"],
        training_code_identity="mnel-fabric-training-code-transition/0.5",
        training_record_ids=tuple(record.identity for record in transition_access.records()),
        calibration_identity=calibration_identity,
        calibration_dataset_identity=calibration_dataset_identity,
    )
    try:
        aggregate_transition_partials(
            transition_parts[:1],
            dataset_identity=transition_access.dataset_identity,
            feature_extractor_identity=ids["feature_extractor_identity"],
            training_code_identity="mnel-fabric-training-code-transition/0.5",
            training_record_ids=tuple(record.identity for record in transition_access.records()),
            calibration_identity=calibration_identity,
            calibration_dataset_identity=calibration_dataset_identity,
        )
    except DistributedExecutionError as error:
        missing_shard_disposition = str(error)
    else:
        missing_shard_disposition = "unexpectedly-accepted"
    report = {
        "schema": "mnel-fabric-reference-study-report/0.5",
        "study_identity": study,
        "backend_identity": backend.backend_identity,
        "fabric_public_available": backend.fabric_public_available,
        "workload_graph_identity": graph.graph_identity or graph.content_identity,
        "study_matrix_identity": matrix.matrix_identity or matrix.content_identity,
        "study_matrix_cell_count": len(matrix.cells),
        "logical_workers": ["mnel-worker-0", "mnel-worker-1"],
        "provider_artifact_identities": [
            transition_model.artifact_identity,
            centroid_model.artifact_identity,
        ],
        "observation_count": len(observations),
        "worker_identities": sorted({item.worker_identity for item in observations}),
        "replication": {
            "requested": 2,
            "observed": len(observations),
            "cross_node_replication": True,
            "independence_claim": "not-established",
        },
        "transition_sharded_training": {
            "partial_count": len(transition_parts),
            "aggregate_artifact_identity": aggregate.artifact_identity,
            "single_host_equivalent": aggregate.transition_counts
            == transition_model.transition_counts
            and aggregate.total_count == transition_model.total_count
            and aggregate.model_identity == transition_model.model_identity,
            "missing_shard_disposition": missing_shard_disposition,
        },
        "fault_semantics": {
            "duplicate_shard": "rejected",
            "overlap": "rejected",
            "hidden_data": "rejected",
            "worker_loss": "UNKNOWN/incomplete",
        },
        "authority": AUTHORITY,
        "semantics": "distributed reference evidence; not-a-verdict",
        "limitations": [
            "logical workers share one controller process and host; no physical cross-host independence",
            "network bundle transfer is not implemented; remote mode requires operator pre-staging",
            "the tiny fixture uses coarse-grained reference provider tasks",
        ]
        + (
            []
            if backend.fabric_public_available
            else [
                "mncs-fabric was unavailable; the self-contained fallback preserved workload semantics without claiming Fabric execution"
            ]
        ),
    }
    if workspace is not None:
        ledger = EvidenceLedger(Path(workspace) / "fabric-evidence.jsonl")
        for snapshot in (transition, transition_2, tabular):
            ledger.append("fabric-snapshot", snapshot.to_dict(), actor="mnel-fabric-reference")
        for record in records:
            ledger.append("fabric-source-record", record.to_dict(), actor="mnel-fabric-reference")
        for value in (graph.to_dict(), *[workload.to_dict() for workload in workloads]):
            ledger.append("fabric-workload", value, actor="mnel-fabric-reference")
        ledger.append("fabric-study-matrix", matrix.to_dict(), actor="mnel-fabric-reference")
        for item in observations:
            ledger.append(
                "fabric-execution-observation", item.to_dict(), actor="mnel-fabric-reference"
            )
        for shard in shards:
            ledger.append("fabric-training-shard", shard.to_dict(), actor="mnel-fabric-reference")
        for item in transition_parts:
            ledger.append(
                "fabric-partial-training-artifact", item.to_dict(), actor="mnel-fabric-reference"
            )
        ledger.append("fabric-reference-study-report", report, actor="mnel-fabric-reference")
        report["ledger"] = ledger.summarize()
    return {
        "report": report,
        "workload_graph": graph.to_dict(),
        "observations": [item.to_dict() for item in observations],
        "shards": [item.to_dict() for item in shards],
        "partial_artifacts": [item.to_dict() for item in transition_parts],
    }


__all__ = [
    "DistributedExecutionError",
    "DistributedStudyCell",
    "DistributedStudyMatrix",
    "DistributedWorkload",
    "ExecutionDisposition",
    "ExpertPlacementRequest",
    "FabricExecutionObservation",
    "FabricNetworkConfig",
    "NetworkFabricBackend",
    "FabricWorkerConfig",
    "LocalFabricBackend",
    "PartialTrainingArtifact",
    "TrainingShard",
    "WorkloadClass",
    "WorkloadGraph",
    "WorkloadNode",
    "aggregate_centroid_partials",
    "aggregate_transition_partials",
    "centroid_training_partial",
    "make_training_shards",
    "make_study_matrix",
    "make_workload_graph",
    "partition_records",
    "run_network_fabric",
    "run_reference_fabric_study",
    "transition_training_partial",
]
