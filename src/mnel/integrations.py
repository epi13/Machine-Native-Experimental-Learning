"""Provider-neutral adapters for the MNCS project family."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .core import canonical_digest


@dataclass(frozen=True)
class AdapterResult:
    returncode: int
    stdout: str
    stderr: str
    parsed: dict[str, Any] | None


class JSONCommandAdapter:
    """Invoke an identified command with JSON I/O and no shell interpolation."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("command must contain non-empty strings")
        self.command = tuple(command)
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.timeout_seconds = timeout_seconds

    def run(self, request: dict[str, Any]) -> AdapterResult:
        completed = subprocess.run(
            self.command,
            input=json.dumps(request, sort_keys=True),
            text=True,
            capture_output=True,
            cwd=self.cwd,
            timeout=self.timeout_seconds,
            check=False,
            shell=False,
        )
        parsed = None
        if completed.stdout.strip():
            try:
                value = json.loads(completed.stdout)
                if isinstance(value, dict):
                    parsed = value
            except json.JSONDecodeError:
                pass
        return AdapterResult(completed.returncode, completed.stdout, completed.stderr, parsed)


@dataclass(frozen=True)
class LocalHarnessRequest:
    task_id: str
    role: str
    prompt: str
    eligible_record_ids: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    workspace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mnel-local-investigator-request/0.1",
            **asdict(self),
            "authority": "proposal-only",
        }


@dataclass(frozen=True)
class ForgeProbeRequest:
    probe_id: str
    question: str
    subject_identities: dict[str, str]
    expected_witness_type: str
    resource_budget: dict[str, int]
    mutation_forbidden: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "mnel-forge-probe-request/0.1", **asdict(self)}


class ForgeProbeProvider(Protocol):
    def run_probe(self, request: ForgeProbeRequest) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FabricExperimentRequest:
    experiment_id: str
    artifact_manifest_id: str
    job_plan_id: str
    required_capabilities: tuple[str, ...]
    replication_count: int
    visibility: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "mnel-fabric-experiment-request/0.1", **asdict(self)}


@dataclass(frozen=True)
class RavelKnowledgeProposal:
    proposal_id: str
    proposal_type: str
    parent_candidate_id: str
    supporting_record_ids: tuple[str, ...]
    declared_scope: dict[str, Any]
    predicted_effects: tuple[dict[str, Any], ...]
    rollback_target: str
    promotion_authorized: bool = False

    def __post_init__(self) -> None:
        if self.promotion_authorized:
            raise ValueError("MNEL cannot authorize RAVEL promotion")
        if not self.supporting_record_ids:
            raise ValueError("RAVEL proposals must preserve source lineage")

    def to_dict(self) -> dict[str, Any]:
        value = {"schema": "mnel-ravel-knowledge-proposal/0.1", **asdict(self)}
        value["content_identity"] = canonical_digest(value)
        return value
