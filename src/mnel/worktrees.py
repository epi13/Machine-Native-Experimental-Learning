"""Bounded Git worktree materialization for proposal-only investigator runs."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .core import canonical_digest
from .investigator_harness import CandidateTransaction, InvestigatorWorkspace


class WorktreeError(ValueError):
    """A source identity, path, or Git operation was not safe to use."""


@dataclass(frozen=True, slots=True)
class SourceRevision:
    repository: str
    requested_ref: str
    commit: str
    source_identity: str

    def to_dict(self) -> dict[str, str]:
        return {
            "repository": self.repository,
            "requested_ref": self.requested_ref,
            "commit": self.commit,
            "source_identity": self.source_identity,
        }


@dataclass(frozen=True, slots=True)
class MaterializedWorktree:
    source: SourceRevision
    path: str
    transaction: CandidateTransaction

    def workspace(self) -> InvestigatorWorkspace:
        return InvestigatorWorkspace.proposal(self.path)

    def to_dict(self) -> dict[str, object]:
        value = {
            "schema": "mnel-materialized-worktree/0.2",
            "source": self.source.to_dict(),
            "path": self.path,
            "transaction": self.transaction.to_dict(),
            "access": "proposal",
            "proposal_only": True,
        }
        value["materialization_identity"] = canonical_digest(value)
        return value


class GitWorktreeMaterializer:
    """Materialize detached proposal worktrees under one configured root.

    The implementation never interpolates a shell command and never removes a
    worktree implicitly.  Git's worktree metadata is the only intentional
    mutation outside the configured materialization root.
    """

    _SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        git_executable: Sequence[str] = ("git",),
        timeout_seconds: int = 30,
    ) -> None:
        if not git_executable or any(not part for part in git_executable):
            raise ValueError("git_executable must contain non-empty command parts")
        if timeout_seconds < 1 or timeout_seconds > 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        self.workspace_root = Path(workspace_root).resolve()
        self.git_executable = tuple(git_executable)
        self.timeout_seconds = timeout_seconds

    def identify(self, repository: str | Path, ref: str) -> SourceRevision:
        repository_root = self._repository_root(repository)
        self._validate_ref(ref)
        commit = self._git(
            repository_root,
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"{ref}^{{commit}}",
        ).strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
            raise WorktreeError("Git did not return a concrete commit identity")
        source_identity = canonical_digest(
            {"repository": str(repository_root), "commit": commit.lower()}
        )
        return SourceRevision(str(repository_root), ref, commit.lower(), source_identity)

    def materialize(
        self,
        source: SourceRevision,
        transaction: CandidateTransaction,
        *,
        name: str | None = None,
    ) -> MaterializedWorktree:
        source_root = Path(source.repository).resolve()
        if not source_root.is_dir() or not self._is_git_root(source_root):
            raise WorktreeError("source repository is no longer a valid Git checkout")
        self._validate_root(source_root)
        if transaction.access.value != "proposal" or not transaction.proposal_only:
            raise WorktreeError("materialized investigator worktrees must be proposal-only")
        selected_name = name or f"mnel-{transaction.transaction_identity.removeprefix('sha256:')[:24]}"
        if not self._SAFE_NAME.fullmatch(selected_name):
            raise WorktreeError("unsafe worktree name")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        path = (self.workspace_root / selected_name).resolve()
        if not self._within(path, self.workspace_root) or path == self.workspace_root:
            raise WorktreeError("worktree path escapes the configured root")
        if path.exists():
            raise WorktreeError("refusing to overwrite an existing experiment worktree")
        self._git(
            source_root,
            "worktree",
            "add",
            "--detach",
            str(path),
            source.commit,
        )
        return MaterializedWorktree(source, str(path), transaction)

    def write_metadata(self, worktree: MaterializedWorktree, extra: dict[str, object] | None = None) -> Path:
        path = Path(worktree.path).resolve()
        self._assert_materialized_path(path)
        value: dict[str, object] = {"worktree": worktree.to_dict()}
        if extra:
            value["context"] = dict(extra)
        metadata = path / ".mnel-worktree.json"
        metadata.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return metadata

    def cleanup(self, worktree: MaterializedWorktree, *, preserve: bool = True) -> bool:
        """Remove only when explicitly requested with ``preserve=False``."""

        path = Path(worktree.path).resolve()
        self._assert_materialized_path(path)
        if preserve:
            return False
        if not path.exists():
            return False
        self._git(Path(worktree.source.repository), "worktree", "remove", "--force", str(path))
        return True

    def _repository_root(self, repository: str | Path) -> Path:
        candidate = Path(repository).expanduser().resolve()
        if not candidate.is_dir():
            raise WorktreeError("repository is not a directory")
        output = self._git(candidate, "rev-parse", "--show-toplevel").strip()
        root = Path(output).resolve()
        if not root.is_dir():
            raise WorktreeError("Git returned an invalid repository root")
        return root

    def _is_git_root(self, repository: Path) -> bool:
        try:
            return Path(self._git(repository, "rev-parse", "--show-toplevel").strip()).resolve() == repository
        except WorktreeError:
            return False

    def _validate_root(self, source_root: Path) -> None:
        if self.workspace_root == source_root or self._within(self.workspace_root, source_root):
            raise WorktreeError("materialization root must not be inside the authoritative checkout")

    def _assert_materialized_path(self, path: Path) -> None:
        if not self._within(path, self.workspace_root) or path == self.workspace_root:
            raise WorktreeError("worktree path escapes the configured root")

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_ref(ref: str) -> None:
        if not ref or not ref.strip() or ref.startswith("-") or "\x00" in ref:
            raise WorktreeError("invalid Git ref")

    def _git(self, repository: Path, *arguments: str) -> str:
        try:
            completed = subprocess.run(
                (*self.git_executable, "-C", str(repository), *arguments),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WorktreeError(f"Git command failed: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
            raise WorktreeError(f"Git command failed: {detail}")
        return completed.stdout
