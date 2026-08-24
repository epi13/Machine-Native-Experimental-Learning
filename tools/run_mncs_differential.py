#!/usr/bin/env python3
"""Run the MNEL-core differential study: reference MNEL vs MNCS-language MNEL.

Pipeline:

    same frozen inputs (mncs/corpora/mnel-core-reference.json)
        |
        +--> existing MNEL implementation (oracle: expected values in corpus,
        |     produced by src/mnel classes by generate_mncs_core_corpus.py)
        |
        +--> MNCS-language implementation (mncs/source/mnel-core.mncs)
                 executed through one or more compiler backends
                 |
                 v
          bounded comparison (case-level expectation agreement)

Outputs a machine-readable evidence record under
docs/mncs-reconstruction/evidence/. The record never claims universal
equivalence: it reports observed case-level agreement, retained UNKNOWN
obligations, and per-backend support refusals as distinct outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPO_ROOT / "mncs" / "corpora" / "mnel-core-reference.json"
SOURCE_PATH = REPO_ROOT / "mncs" / "source" / "mnel-core.mncs"
EVIDENCE_DIR = REPO_ROOT / "docs" / "mncs-reconstruction" / "evidence"

RUNNER_IDENTITY = "mnel-mncs-differential-runner/0.1"
STUDY_IDENTITY = "mnel-core-concept-reconstruction"
INTERPRETATION = (
    "bounded_observational_agreement_over_declared_corpus; "
    "not_universal_equivalence_not_conformance_not_assurance"
)

BACKENDS = {
    "mncs-research-bytecode": "bytecode",
    "mncs-portable-wasm-mvp": "wasm",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_mncs(mncs_bin: list[str], backend: str, out_dir: Path) -> tuple[int, dict | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *mncs_bin,
        "experiment",
        "run",
        str(SOURCE_PATH),
        "--backend",
        backend,
        "--corpus",
        str(CORPUS_PATH),
        "--output-dir",
        str(out_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    stdout = completed.stdout
    try:
        payload = json.loads(stdout[stdout.find("{"):])
    except (ValueError, TypeError):
        payload = None
    return completed.returncode, payload


def classify_backend_result(rc: int, payload: dict | None) -> tuple[str, list[dict], dict]:
    """Classify one backend run into an outcome kind plus detail records."""
    if payload is None:
        return "runner-error", [], {"exit_code": rc}
    diagnostics = payload.get("diagnostics") or []
    refusal_codes = [
        d for d in diagnostics if str(d.get("code", "")).startswith(("CGN3", "CGR3"))
    ]
    cases = payload.get("cases") or []
    if refusal_codes and not cases:
        return "backend-refused-out-of-envelope", refusal_codes, {
            "exit_code": rc,
            "compilation_status": payload.get("status"),
        }
    met = sum(1 for c in cases if c.get("expectation_met") is True)
    unmet = [c for c in cases if c.get("expectation_met") is not True]
    return "corpus-executed", unmet, {
        "exit_code": rc,
        "cases_total": len(cases),
        "cases_met": met,
        "experiment_status": payload.get("status"),
        "unresolved_reasons": payload.get("unresolved_reasons") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mncs-bin",
        nargs="+",
        default=["cargo", "run", "--quiet", "-p", "mncs-cli", "--"],
        help="command prefix that runs the mncs CLI",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "target" / "mncs-differential",
        help="directory for per-backend outputs",
    )
    parser.add_argument("--backend", choices=sorted(BACKENDS), action="append",
                        help="restrict to one backend (repeatable)")
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        print("corpus missing; run tools/generate_mncs_core_corpus.py first", file=sys.stderr)
        return 2

    corpus = json.loads(CORPUS_PATH.read_text())
    selected_backends = args.backend or list(BACKENDS)

    backend_observations = []
    disagreements = []
    for backend in selected_backends:
        tag = BACKENDS[backend]
        rc, payload = run_mncs(args.mncs_bin, backend, args.work_dir / tag)
        outcome, details_list, summary = classify_backend_result(rc, payload)
        observation = {
            "backend": backend,
            "outcome": outcome,
            "summary": summary,
        }
        if outcome == "backend-refused-out-of-envelope":
            observation["refusal_diagnostics"] = [
                {"code": d.get("code"), "message": d.get("message")}
                for d in details_list[:8]
            ]
            observation["interpretation"] = (
                "fail-closed envelope refusal; absence of execution is not disagreement"
            )
        elif outcome == "corpus-executed":
            for case in details_list:
                disagreements.append({"backend": backend, "case_id": case.get("case_id"),
                                      "failure_reason": case.get("failure_reason")})
        backend_observations.append(observation)

    executed = [b for b in backend_observations if b["outcome"] == "corpus-executed"]
    consistent = (
        len(executed) > 0
        and all(b["summary"]["cases_met"] == b["summary"]["cases_total"] for b in executed)
        and not disagreements
    )

    if disagreements:
        comparison_status = "MISMATCH_DETECTED"
    elif executed:
        comparison_status = "AGREEMENT_OVER_CORPUS"
    else:
        comparison_status = "NO_EXECUTING_BACKEND"

    evidence = {
        "schema_version": "0.1",
        "identity_kind": "bounded-differential-study-record",
        "study": STUDY_IDENTITY,
        "runner_identity": RUNNER_IDENTITY,
        "interpretation": INTERPRETATION,
        "reference_side": {
            "implementation": "Machine-Native-Experimental-Learning src/mnel (Python control plane)",
            "corpus": str(CORPUS_PATH.relative_to(REPO_ROOT)),
            "corpus_sha256": sha256_file(CORPUS_PATH),
            "cases": len(corpus.get("cases", [])),
            "oracle_kinds": sorted({
                c.get("oracle", {}).get("kind")
                for c in corpus.get("cases", [])
                if c.get("oracle")
            }),
        },
        "mncs_side": {
            "source": str(SOURCE_PATH.relative_to(REPO_ROOT)),
            "source_sha256": sha256_file(SOURCE_PATH),
            "module": "mnel.core",
            "language_profile": "0.5",
        },
        "backends": backend_observations,
        "comparison_status": comparison_status,
        "disagreements": disagreements,
        "non_claims": [
            "observed agreement is not proof of semantic equivalence",
            "the corpus covers a bounded slice of MNEL concepts only",
            "backend envelope refusals are recorded, not resolved",
            "unresolved obligations remain UNKNOWN; nothing here certifies the "
            "MNCS implementation against the reference beyond the corpus",
        ],
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVIDENCE_DIR / "mnel-core-differential-study.json"
    out_path.write_text(json.dumps(evidence, indent=1) + "\n")

    print(f"comparison: {comparison_status}")
    for b in backend_observations:
        summary = b.get("summary", {})
        if "cases_met" in summary:
            print(f"  {b['backend']}: {summary['cases_met']}/{summary['cases_total']} "
                  f"(status={summary.get('experiment_status')})")
        else:
            print(f"  {b['backend']}: {b['outcome']}")
    print(f"evidence: {out_path.relative_to(REPO_ROOT)}")

    if disagreements:
        for d in disagreements[:20]:
            print(f"  DISAGREE {d['backend']} {d['case_id']}: {d['failure_reason']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
