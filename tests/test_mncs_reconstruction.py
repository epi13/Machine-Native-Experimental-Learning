"""MNCS-language reconstruction tests for the mnel.core slice.

These tests exercise the MNCS-native reconstruction of MNEL's core decision
semantics against the reference Python implementation. They require the `mncs`
CLI binary from the mncs-language repository; when it is unavailable the tests
skip with a clear message rather than failing.

Run locally with:
    python -m unittest tests.test_mncs_reconstruction -v

Environment:
    MNCS_BIN: path to the mncs CLI (default: ../../../mncs-language/target/debug/mncs)
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNCS = (
    REPO_ROOT.parent / "mncs-language" / "target" / "debug" / "mncs"
)
DEFAULT_SOURCE = REPO_ROOT / "mncs" / "source" / "mnel" / "all.mncs"


def mncs_bin() -> Path | None:
    configured = os.environ.get("MNCS_BIN")
    candidate = Path(configured) if configured else DEFAULT_MNCS
    return candidate if candidate.exists() else None


@unittest.skipIf(mncs_bin() is None, "mncs CLI binary not available")
class MncsReconstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mncs = str(mncs_bin())
        self.source = Path(os.environ.get("MNCS_ENTRY", str(DEFAULT_SOURCE)))
        self.assertTrue(self.source.exists(), f"entry source missing: {self.source}")

    def test_source_studies_cleanly_with_only_conservative_obligations(self) -> None:
        completed = subprocess.run(
            [self.mncs, "source-study", str(self.source), "--node-id", "unittest-core"],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout[completed.stdout.find("{"):])
        errors = [
            d for d in payload.get("diagnostics", [])
            if d.get("severity") == "error"
        ]
        self.assertEqual(errors, [])
        self.assertEqual(payload.get("compilation_status"),
                         "completed_with_unresolved_obligations")

    def test_differential_study_agrees_over_corpus_on_executing_backend(self) -> None:
        corpus_script = REPO_ROOT / "tools" / "generate_mncs_core_corpus.py"
        subprocess.run(
            [sys.executable, str(corpus_script)], check=True, cwd=str(REPO_ROOT),
            capture_output=True,
        )
        runner = REPO_ROOT / "tools" / "run_mncs_differential.py"
        work = REPO_ROOT / "target" / "mncs-differential-unittest"
        completed = subprocess.run(
            ["python3", str(runner), "--mncs-bin", self.mncs,
             "--backend", "mncs-research-bytecode", "--work-dir", str(work)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        evidence = json.loads(
            (REPO_ROOT / "docs" / "mncs-reconstruction" / "evidence" /
             "mnel-core-differential-study.json").read_text()
        )
        self.assertEqual(evidence["comparison_status"], "AGREEMENT_OVER_CORPUS")

    def test_negative_fixtures_are_rejected(self) -> None:
        checker = REPO_ROOT / "tools" / "check_mncs_negative_fixtures.py"
        completed = subprocess.run(
            ["python3", str(checker), self.mncs],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_evidence_record_preserves_non_claims(self) -> None:
        path = (REPO_ROOT / "docs" / "mncs-reconstruction" / "evidence" /
                "mnel-core-differential-study.json")
        if not path.exists():
            self.skipTest("differential evidence not generated yet")
        evidence = json.loads(path.read_text())
        self.assertIn("not_universal_equivalence", evidence["interpretation"])
        self.assertTrue(evidence["non_claims"])


if __name__ == "__main__":
    unittest.main()
