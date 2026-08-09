import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from mnel.core import EvidenceLedger
from mnel.integrations import LocalHarnessAdapter, LocalHarnessRequest
from mnel.investigator_harness import RuntimeIdentityEnvelope
from mnel.local_runtime import run_local_investigator
from mnel.worktrees import GitWorktreeMaterializer, WorktreeError


class LocalRuntimeTests(unittest.TestCase):
    def _fixture(self, directory: Path) -> Path:
        path = directory / "fake_harness.py"
        path.write_text(
            textwrap.dedent(
                """
                import json
                import sys
                import time

                request = json.loads(sys.stdin.read())
                mode = sys.argv[1]
                if mode == "timeout":
                    time.sleep(2)
                if mode == "malformed":
                    print("not-json")
                    raise SystemExit(0)
                result = {
                    "route": {"primary_role": "investigator", "escalation_roles": [], "reasons": []},
                    "final_content": "bounded proposal",
                    "successful": True,
                    "attempts": [{"role": "investigator", "model": "fixture", "content": "bounded proposal"}],
                }
                if mode == "authority":
                    result["verdict"] = "PASS"
                print(json.dumps({
                    "protocolVersion": 1,
                    "requestId": request["requestId"],
                    "method": "chat/start",
                    "result": result,
                }))
                """
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _request(workspace: Path) -> LocalHarnessRequest:
        return LocalHarnessRequest(
            task_id="task-1",
            role="investigator",
            prompt="propose a bounded diagnostic question",
            eligible_context_identity="sha256:context",
            eligible_record_ids=("record-1",),
            allowed_tools=("read",),
            workspace=str(workspace.resolve()),
            runtime_identity=RuntimeIdentityEnvelope("model", "q", "runtime", "prompt", "tools"),
            timeout_seconds=1,
        )

    def test_adapter_accepts_bounded_diagnostic_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = (sys.executable, str(self._fixture(root)), "success")
            observation = LocalHarnessAdapter(command).execute(self._request(root))
            self.assertEqual(observation.status, "completed")
            self.assertEqual(observation.model_output, "bounded proposal")
            self.assertIsNone(observation.to_dict().get("verdict"))
            self.assertEqual(observation.authority, "proposal-only")

    def test_adapter_fails_closed_for_malformed_authority_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, expected in (("malformed", "quarantined"), ("authority", "quarantined"), ("timeout", "unknown")):
                observation = LocalHarnessAdapter(
                    (sys.executable, str(self._fixture(root)), mode), timeout_seconds=1
                ).execute(self._request(root))
                self.assertEqual(observation.status, expected)
                self.assertEqual(observation.model_output, "")

    def test_request_rejects_authority_expanding_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                LocalHarnessRequest(
                    task_id="task-1",
                    role="investigator",
                    prompt="bounded",
                    eligible_context_identity="sha256:context",
                    eligible_record_ids=("record-1",),
                    allowed_tools=("promotion",),
                    workspace=str(Path(directory).resolve()),
                )

    def test_git_materialization_validates_refs_paths_and_explicit_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "source"
            workspace_root = root / "experiments"
            repository.mkdir()
            self._git(repository, "init", "-q")
            self._git(repository, "config", "user.email", "fixture@example.invalid")
            self._git(repository, "config", "user.name", "MNEL fixture")
            (repository / "source.txt").write_text("source\n", encoding="utf-8")
            self._git(repository, "add", "source.txt")
            self._git(repository, "commit", "-qm", "initial")
            materializer = GitWorktreeMaterializer(workspace_root)
            source = materializer.identify(repository, "HEAD")
            transaction_path = workspace_root / "candidate"
            from mnel.investigator_harness import CandidateTransaction

            transaction = CandidateTransaction.create(
                parent_candidate_id="candidate-1",
                context_snapshot_identity="sha256:context",
                workspace=transaction_path,
            )
            with self.assertRaises(WorktreeError):
                materializer.identify(repository, "--bad-ref")
            with self.assertRaises(WorktreeError):
                materializer.materialize(source, transaction, name="../escape")
            worktree = materializer.materialize(source, transaction, name="candidate")
            self.assertEqual((Path(worktree.path) / "source.txt").read_text(encoding="utf-8"), "source\n")
            self.assertFalse(materializer.cleanup(worktree, preserve=True))
            self.assertTrue(Path(worktree.path).exists())
            self.assertTrue(materializer.cleanup(worktree, preserve=False))
            self.assertFalse(Path(worktree.path).exists())

    def test_end_to_end_run_keeps_source_immutable_and_records_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "source"
            repository.mkdir()
            self._git(repository, "init", "-q")
            self._git(repository, "config", "user.email", "fixture@example.invalid")
            self._git(repository, "config", "user.name", "MNEL fixture")
            (repository / "source.txt").write_text("authoritative\n", encoding="utf-8")
            self._git(repository, "add", "source.txt")
            self._git(repository, "commit", "-qm", "initial")
            fixture = self._fixture(root)
            run = run_local_investigator(
                repository=repository,
                base_ref="HEAD",
                workspace_root=root / "experiments",
                task_id="task-e2e",
                parent_candidate_id="candidate-1",
                prompt="propose one bounded probe",
                role="investigator",
                allowed_tools=("read",),
                runtime_identity=RuntimeIdentityEnvelope("model", "q", "runtime", "prompt", "tools"),
                records=({"record_id": "record-1", "value": "visible"},),
                adapter=LocalHarnessAdapter((sys.executable, str(fixture), "success")),
            )
            self.assertEqual(run.observation.status, "completed")
            self.assertEqual((repository / "source.txt").read_text(encoding="utf-8"), "authoritative\n")
            ledger = EvidenceLedger(root / "evidence.jsonl")
            run.append_to_ledger(ledger)
            self.assertTrue(ledger.verify().valid)
            self.assertEqual(ledger.verify().record_count, 3)
            self.assertTrue((Path(run.worktree.path) / ".mnel-worktree.json").exists())
            GitWorktreeMaterializer(root / "experiments").cleanup(run.worktree, preserve=False)

    @staticmethod
    def _git(repository: Path, *args: str) -> None:
        completed = subprocess.run(
            ("git", "-C", str(repository), *args),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)


if __name__ == "__main__":
    unittest.main()
