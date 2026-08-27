"""MNCS training slice tests: dataset + training + evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNCS = REPO_ROOT.parent / "mncs-language" / "target" / "debug" / "mncs"
DEFAULT_SOURCE = REPO_ROOT / "mncs" / "source" / "mnel" / "training.mncs"
DEFAULT_LIBRARY_ROOT = REPO_ROOT.parent / "mncs-language" / "library"

def library_env() -> dict:
    if DEFAULT_LIBRARY_ROOT.is_dir():
        return {**os.environ, "MNCS_LIBRARY_PATH": str(DEFAULT_LIBRARY_ROOT)}
    return dict(os.environ)

def mncs_bin() -> Path | None:
    configured = os.environ.get("MNCS_BIN")
    candidate = Path(configured) if configured else DEFAULT_MNCS
    return candidate if candidate.exists() else None

@unittest.skipIf(mncs_bin() is None, "mncs CLI binary not available")
class MncsTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mncs = str(mncs_bin())
        self.source = Path(os.environ.get("MNCS_TRAINING_ENTRY", str(DEFAULT_SOURCE)))
        self.assertTrue(self.source.exists(), f"entry source missing: {self.source}")

    def test_training_source_studies_cleanly(self) -> None:
        completed = subprocess.run(
            [self.mncs, "source-study", str(self.source), "--node-id", "unittest-training"],
            capture_output=True, text=True, check=True, env=library_env(),
        )
        payload = json.loads(completed.stdout[completed.stdout.find("{"):])
        errors = [d for d in payload.get("diagnostics", []) if d.get("severity") == "error"]
        self.assertEqual(errors, [])
        self.assertEqual(payload.get("compilation_status"), "completed_with_unresolved_obligations")

    def test_dataset_source_studies_cleanly(self) -> None:
        dataset = REPO_ROOT / "mncs" / "source" / "mnel" / "dataset.mncs"
        completed = subprocess.run(
            [self.mncs, "source-study", str(dataset), "--node-id", "unittest-dataset"],
            capture_output=True, text=True, check=True, env=library_env(),
        )
        payload = json.loads(completed.stdout[completed.stdout.find("{"):])
        errors = [d for d in payload.get("diagnostics", []) if d.get("severity") == "error"]
        self.assertEqual(errors, [])

    def test_training_differential_agrees(self) -> None:
        runner = REPO_ROOT / "tools" / "run_mnel_training_differential.py"
        work = REPO_ROOT / "target" / "mnel-training-differential-unittest"
        completed = subprocess.run(
            ["python3", str(runner), "--mncs-bin", self.mncs, "--backend", "mncs-research-bytecode", "--backend", "mncs-portable-wasm-mvp", "--work-dir", str(work)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        evidence = json.loads((REPO_ROOT / "docs" / "mncs-reconstruction" / "evidence" / "mnel-training-differential-study.json").read_text())
        self.assertEqual(evidence["comparison_status"], "AGREEMENT_OVER_CORPUS")

if __name__ == "__main__":
    unittest.main()
