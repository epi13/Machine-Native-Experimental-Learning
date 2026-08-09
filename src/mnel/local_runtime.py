"""Executable bounded local-investigator path composed from existing MNEL contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .core import EvidenceLedger, canonical_digest
from .integrations import LocalHarnessAdapter, LocalHarnessObservation, LocalHarnessRequest
from .investigator_harness import (
    CandidateTransaction,
    PackedContext,
    RuntimeIdentityEnvelope,
    pack_eligible_context,
)
from .worktrees import GitWorktreeMaterializer, MaterializedWorktree, SourceRevision


@dataclass(frozen=True, slots=True)
class LocalInvestigatorRun:
    source: SourceRevision
    context: PackedContext
    worktree: MaterializedWorktree
    request: LocalHarnessRequest
    observation: LocalHarnessObservation

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": "mnel-local-investigator-run/0.2",
            "source": self.source.to_dict(),
            "context": self.context.to_dict(),
            "worktree": self.worktree.to_dict(),
            "request": self.request.to_dict(),
            "observation": self.observation.to_dict(),
            "authority": "proposal-only",
            "semantics": "diagnostic-only; not-a-verdict",
        }
        value["run_identity"] = canonical_digest(value)
        return value

    def append_to_ledger(self, ledger: EvidenceLedger) -> tuple[dict[str, Any], ...]:
        """Record reproducible run components without creating evaluator state."""

        return (
            ledger.append("investigator-materialization", self.worktree.to_dict(), actor="investigator"),
            ledger.append("investigator-request", self.request.to_dict(), actor="investigator"),
            ledger.append("investigator-observation", self.observation.to_dict(), actor="local-harness"),
        )


def run_local_investigator(
    *,
    repository: str | Path,
    base_ref: str,
    workspace_root: str | Path,
    task_id: str,
    parent_candidate_id: str,
    prompt: str,
    role: str,
    allowed_tools: tuple[str, ...],
    runtime_identity: RuntimeIdentityEnvelope,
    records: Iterable[dict[str, Any]],
    adapter: LocalHarnessAdapter,
    max_records: int = 32,
    max_context_bytes: int = 256 * 1024,
) -> LocalInvestigatorRun:
    """Materialize, invoke, and normalize one explicit investigator experiment.

    Failed or malformed runs intentionally leave their worktree available for
    diagnosis. Call ``GitWorktreeMaterializer.cleanup(..., preserve=False)`` only
    after an operator has decided that the material is no longer needed.
    """

    context = pack_eligible_context(
        records,
        max_records=max_records,
        max_bytes=max_context_bytes,
    )
    materializer = GitWorktreeMaterializer(workspace_root)
    source = materializer.identify(repository, base_ref)
    candidate_name = "mnel-" + canonical_digest(
        {"task_id": task_id, "parent_candidate_id": parent_candidate_id, "context": context.snapshot_identity}
    ).removeprefix("sha256:")[:24]
    workspace = Path(workspace_root).resolve() / candidate_name
    transaction = CandidateTransaction.create(
        parent_candidate_id=parent_candidate_id,
        context_snapshot_identity=context.snapshot_identity,
        workspace=workspace,
    )
    worktree = materializer.materialize(source, transaction, name=candidate_name)
    materializer.write_metadata(
        worktree,
        {
            "context_snapshot_identity": context.snapshot_identity,
            "runtime_identity": runtime_identity.to_dict(),
            "task_id": task_id,
        },
    )
    request = LocalHarnessRequest(
        task_id=task_id,
        role=role,
        prompt=prompt,
        eligible_context_identity=context.snapshot_identity,
        eligible_record_ids=context.record_ids,
        allowed_tools=allowed_tools,
        workspace=worktree.path,
        runtime_identity=runtime_identity,
    )
    observation = adapter.execute(request)
    return LocalInvestigatorRun(source, context, worktree, request, observation)
