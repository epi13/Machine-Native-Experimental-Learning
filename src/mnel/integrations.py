"""Provider-neutral adapters for the MNCS project family."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .core import canonical_digest
from .investigator_harness import RuntimeIdentityEnvelope, WorkspaceAccess


@dataclass(frozen=True)
class AdapterResult:
    returncode: int | None
    stdout: str
    stderr: str
    parsed: dict[str, Any] | None
    timed_out: bool = False
    error: str | None = None
    duration_ns: int | None = None


class JSONCommandAdapter:
    """Invoke an identified command with JSON I/O and no shell interpolation."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout_seconds: int = 300,
        max_output_bytes: int = 256 * 1024,
    ) -> None:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("command must contain non-empty strings")
        if timeout_seconds < 1 or timeout_seconds > 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600")
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        self.command = tuple(command)
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def run(
        self,
        request: dict[str, Any],
        *,
        cwd: str | Path | None = None,
        timeout_seconds: int | None = None,
    ) -> AdapterResult:
        selected_cwd = Path(cwd).resolve() if cwd is not None else self.cwd
        if selected_cwd is not None and not selected_cwd.is_dir():
            raise ValueError(f"adapter cwd is not a directory: {selected_cwd}")
        selected_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        if selected_timeout < 1 or selected_timeout > 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600")
        started = time.monotonic_ns()
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request, sort_keys=True) + "\n",
                text=True,
                capture_output=True,
                cwd=selected_cwd,
                timeout=selected_timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _decode_process_output(exc.stdout)
            stderr = _decode_process_output(exc.stderr)
            return AdapterResult(
                None,
                stdout[: self.max_output_bytes],
                stderr[: self.max_output_bytes],
                None,
                timed_out=True,
                error="timeout",
                duration_ns=time.monotonic_ns() - started,
            )
        duration_ns = time.monotonic_ns() - started
        if len(completed.stdout.encode("utf-8")) > self.max_output_bytes or len(
            completed.stderr.encode("utf-8")
        ) > self.max_output_bytes:
            return AdapterResult(
                completed.returncode,
                completed.stdout[: self.max_output_bytes],
                completed.stderr[: self.max_output_bytes],
                None,
                error="output-size-limit",
                duration_ns=duration_ns,
            )
        parsed = None
        error = None
        if completed.stdout.strip():
            try:
                value = json.loads(completed.stdout)
                if isinstance(value, dict):
                    parsed = value
                else:
                    error = "response-not-object"
            except json.JSONDecodeError:
                error = "malformed-json"
        else:
            error = "empty-response"
        return AdapterResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
            parsed,
            error=error,
            duration_ns=duration_ns,
        )


def _decode_process_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@dataclass(frozen=True)
class LocalHarnessRequest:
    task_id: str
    role: str
    prompt: str
    eligible_record_ids: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    workspace: str
    eligible_context_identity: str = ""
    runtime_identity: RuntimeIdentityEnvelope | None = None
    workspace_access: WorkspaceAccess = WorkspaceAccess.PROPOSAL
    proposal_only: bool = True
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.role.strip() or not self.prompt.strip():
            raise ValueError("local harness requests require task, role, and prompt")
        if not self.eligible_record_ids:
            raise ValueError("local harness requests require eligible context records")
        if any(not item.strip() for item in self.eligible_record_ids):
            raise ValueError("eligible record identities must be non-empty")
        if any(not item.strip() for item in self.allowed_tools):
            raise ValueError("allowed tool identities must be non-empty")
        if not Path(self.workspace).is_absolute():
            raise ValueError("local harness workspace must be absolute")
        if not self.eligible_context_identity.strip():
            raise ValueError("eligible context identity is required")
        if self.workspace_access not in {WorkspaceAccess.READ_ONLY, WorkspaceAccess.PROPOSAL}:
            raise ValueError("workspace access must be read-only or proposal")
        if not self.proposal_only:
            raise ValueError("MNEL local harness requests are always proposal-only")
        if self.timeout_seconds < 1 or self.timeout_seconds > 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600")
        forbidden = {"hidden-transfer", "future-final", "promotion", "evaluator"}
        if forbidden.intersection(self.allowed_tools):
            raise ValueError("local harness tools may not expand authority")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": "mnel-local-investigator-request/0.2",
            "task_id": self.task_id,
            "role": self.role,
            "prompt": self.prompt,
            "eligible_context_identity": self.eligible_context_identity,
            "eligible_record_ids": list(self.eligible_record_ids),
            "allowed_tools": list(self.allowed_tools),
            "workspace": self.workspace,
            "workspace_access": self.workspace_access.value,
            "proposal_only": self.proposal_only,
            "timeout_seconds": self.timeout_seconds,
            "authority": "proposal-only",
        }
        if self.runtime_identity is not None:
            value["runtime_identity"] = self.runtime_identity.to_dict()
        value["request_identity"] = canonical_digest(value)
        return value


@dataclass(frozen=True)
class LocalHarnessObservation:
    """A bounded harness observation; it intentionally has no verdict field."""

    request_identity: str
    status: str
    returncode: int | None
    timed_out: bool
    duration_ns: int | None
    stdout_bytes: int
    stderr_bytes: int
    route: dict[str, Any] | None
    model_output: str
    attempts: tuple[dict[str, Any], ...]
    harness_successful: bool | None
    response_identity: str | None
    error: str | None = None
    authority: str = "proposal-only"
    semantics: str = "diagnostic-only; not-a-verdict"

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": "mnel-local-investigator-observation/0.2",
            "request_identity": self.request_identity,
            "status": self.status,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_ns": self.duration_ns,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "route": self.route,
            "model_output": self.model_output,
            "attempts": list(self.attempts),
            "harness_successful": self.harness_successful,
            "response_identity": self.response_identity,
            "error": self.error,
            "authority": self.authority,
            "semantics": self.semantics,
        }
        value["observation_identity"] = canonical_digest(value)
        return value


class LocalHarnessAdapter:
    """Execute a configured local harness bridge without granting it MNEL authority."""

    FORBIDDEN_KEYS = frozenset(
        {
            "verdict",
            "conformance",
            "promotion_authorized",
            "promotion",
            "evaluator_verdict",
            "evaluator_eligible",
            "hidden_transfer",
            "future_final",
            "ravel_promotion",
        }
    )

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int = 300,
        max_output_bytes: int = 256 * 1024,
    ) -> None:
        self.command_adapter = JSONCommandAdapter(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def execute(self, request: LocalHarnessRequest) -> LocalHarnessObservation:
        envelope = {
            "protocolVersion": 1,
            "requestId": request.task_id,
            "method": "chat/start",
            "params": {
                "messages": [{"role": "user", "content": request.prompt}],
                "lane": request.role,
                "mnel_request": request.to_dict(),
            },
        }
        result = self.command_adapter.run(
            envelope,
            cwd=request.workspace,
            timeout_seconds=request.timeout_seconds,
        )
        request_identity = request.to_dict()["request_identity"]
        base = {
            "request_identity": request_identity,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration_ns": result.duration_ns,
            "stdout_bytes": len(result.stdout.encode("utf-8")),
            "stderr_bytes": len(result.stderr.encode("utf-8")),
        }
        if result.timed_out:
            return LocalHarnessObservation(
                **base,
                status="unknown",
                route=None,
                model_output="",
                attempts=(),
                harness_successful=None,
                response_identity=None,
                error="timeout",
            )
        if result.returncode != 0 or result.parsed is None:
            return LocalHarnessObservation(
                **base,
                status="quarantined",
                route=None,
                model_output="",
                attempts=(),
                harness_successful=None,
                response_identity=None,
                error=result.error or "command-failed",
            )
        try:
            response = self._validate_response(result.parsed, request)
        except ValueError as exc:
            return LocalHarnessObservation(
                **base,
                status="quarantined",
                route=None,
                model_output="",
                attempts=(),
                harness_successful=None,
                response_identity=None,
                error=str(exc),
            )
        return LocalHarnessObservation(
            **base,
            status="completed",
            route=response["route"],
            model_output=response["final_content"],
            attempts=tuple(response["attempts"]),
            harness_successful=response["successful"],
            response_identity=canonical_digest(response),
        )

    @classmethod
    def _validate_response(
        cls, response: dict[str, Any], request: LocalHarnessRequest
    ) -> dict[str, Any]:
        cls._reject_authority(response)
        if response.get("protocolVersion") != 1 or response.get("method") != "chat/start":
            raise ValueError("unexpected local harness protocol or method")
        if response.get("requestId") != request.task_id:
            raise ValueError("local harness response request identity mismatch")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError("local harness result must be an object")
        cls._reject_authority(result)
        if result.get("ok") is False:
            raise ValueError("local harness rejected the request")
        route = result.get("route")
        final_content = result.get("final_content")
        successful = result.get("successful")
        attempts = result.get("attempts", [])
        if not isinstance(route, dict) or not isinstance(final_content, str):
            raise ValueError("local harness result is missing bounded diagnostic content")
        if not isinstance(successful, bool) or not isinstance(attempts, list):
            raise ValueError("local harness result has invalid execution fields")
        normalized_attempts: list[dict[str, Any]] = []
        for attempt in attempts[:32]:
            if not isinstance(attempt, dict):
                raise ValueError("local harness attempt must be an object")
            cls._reject_authority(attempt)
            normalized_attempts.append(
                {
                    key: attempt[key]
                    for key in ("role", "model", "content", "error", "verification")
                    if key in attempt
                }
            )
        if len(final_content.encode("utf-8")) > 128 * 1024:
            raise ValueError("local harness model output exceeds the bounded limit")
        return {
            "route": route,
            "final_content": final_content,
            "successful": successful,
            "attempts": normalized_attempts,
        }

    @classmethod
    def _reject_authority(cls, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in cls.FORBIDDEN_KEYS:
                    raise ValueError(f"local harness response contains forbidden authority field: {key}")
                if key == "authority" and child != "proposal-only":
                    raise ValueError("local harness response attempted to change authority")
                if key == "visibility" and child in {"transfer-hidden", "future-final"}:
                    raise ValueError("local harness response attempted hidden visibility")
                cls._reject_authority(child)
        elif isinstance(value, list):
            for child in value:
                cls._reject_authority(child)


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
