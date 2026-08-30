"""MNCS-native one-step specialist source and backend-matrix tests."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "mncs" / "source" / "mnel" / "one_step.mncs"
CORPUS = REPO_ROOT / "mncs" / "source" / "mnel" / "one_step-corpus.json"
RUNNER = REPO_ROOT / "tools" / "run_mnel_native_one_step.py"


def dependency_root() -> Path:
    worktree = REPO_ROOT.parent / ".mncs-language-luna-worktree"
    return worktree if worktree.is_dir() else REPO_ROOT.parent / "mncs-language"


def mncs_bin() -> Path | None:
    configured = os.environ.get("MNCS_BIN")
    candidate = (
        Path(configured)
        if configured
        else dependency_root() / "target" / "debug" / "mncs"
    )
    return candidate if candidate.exists() else None


def library_env() -> dict[str, str]:
    library = dependency_root() / "library"
    if library.is_dir():
        return {**os.environ, "MNCS_LIBRARY_PATH": str(library)}
    return dict(os.environ)


@unittest.skipIf(mncs_bin() is None, "mncs CLI binary not available")
class MncsNativeOneStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mncs = str(mncs_bin())
        self.assertTrue(SOURCE.exists(), f"native source missing: {SOURCE}")
        self.assertTrue(CORPUS.exists(), f"native corpus missing: {CORPUS}")

    def test_native_source_studies_with_no_error_diagnostics(self) -> None:
        completed = subprocess.run(
            [self.mncs, "source-study", str(SOURCE), "--node-id", "unittest-native-one-step"],
            capture_output=True,
            text=True,
            check=True,
            env=library_env(),
        )
        payload = json.loads(completed.stdout[completed.stdout.find("{") :])
        errors = [d for d in payload.get("diagnostics", []) if d.get("severity") == "error"]
        self.assertEqual(errors, [])
        self.assertTrue(SOURCE.read_text().startswith("mncs 0.10;"))
        self.assertEqual(payload.get("compilation_status"), "completed_with_unresolved_obligations")

    def test_native_backend_matrix_agrees_over_stateful_corpus(self) -> None:
        evidence_path = REPO_ROOT / "docs" / "mncs-reconstruction" / "evidence" / "mnel-native-one-step-study.json"
        work_dir = REPO_ROOT / "target" / "mnel-native-one-step-unittest"
        completed = subprocess.run(
            [
                "python3",
                str(RUNNER),
                "--mncs-bin",
                self.mncs,
                "--work-dir",
                str(work_dir),
                "--library-path",
                str(dependency_root() / "library"),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=library_env(),
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        evidence = json.loads(evidence_path.read_text())
        self.assertEqual(
            evidence["cross_backend_agreement"]["status"],
            "AGREEMENT_OVER_DECLARED_STATEFUL_CORPUS",
        )
        self.assertEqual(len(evidence["backends"]), 5)
        self.assertTrue(all(item["execution_status"] == "PASS" for item in evidence["backends"]))
        self.assertTrue(all(item["all_expectations_met"] for item in evidence["backends"]))
        self.assertEqual(
            evidence["native_reference_comparison"]["relevant-in-domain"]["native_decision_code"],
            1,
        )
        self.assertEqual(
            evidence["native_reference_comparison"]["relevant-in-domain"]["reference_decision"],
            "relevant",
        )
        self.assertEqual(
            evidence["native_reference_comparison"]["out-of-distribution-abstains"]["reference_decision"],
            "ABSTAIN",
        )
        for backend in evidence["backends"]:
            cases = {item["case_id"]: item for item in backend["stateful_cases"]}
            self.assertEqual(cases["relevant-in-domain"]["final_decision_code"], 1)
            self.assertEqual(cases["relevant-in-domain"]["decision"]["learned_forward_passes"], 1)
            self.assertEqual(cases["out-of-distribution-abstains"]["final_decision_code"], 0)
            self.assertEqual(cases["out-of-distribution-abstains"]["decision"]["learned_forward_passes"], 0)
            self.assertTrue(cases["out-of-distribution-abstains"]["decision"]["fallback_required"])
            for case_id in ("generation-mismatch-abstains", "schema-mismatch-abstains"):
                self.assertEqual(cases[case_id]["final_decision_code"], 0)
                self.assertEqual(cases[case_id]["decision"]["learned_forward_passes"], 0)
                self.assertTrue(cases[case_id]["decision"]["fallback_required"])
            self.assertTrue(
                all(
                    item["decision"]["authority"].endswith("Authority::DIAGNOSTIC_ONLY")
                    for item in cases.values()
                )
            )
        self.assertEqual(
            evidence["python_reference"]["teacher"]["decisions"],
            evidence["python_reference"]["python_one_step_control"]["decisions"],
        )


if __name__ == "__main__":
    unittest.main()
