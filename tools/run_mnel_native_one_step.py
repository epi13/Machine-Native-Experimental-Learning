#!/usr/bin/env python3
"""Run the MNEL native one-step specialist study across MNCS backends.

The native source is a bounded fixed-point student implementation.  This
runner records backend execution separately from compiler status, because the
source intentionally retains checked-arithmetic obligations as UNKNOWN rather
than turning those obligations into an unsupported claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "mncs" / "source" / "mnel" / "one_step.mncs"
CORPUS_PATH = REPO_ROOT / "mncs" / "source" / "mnel" / "one_step-corpus.json"
EVIDENCE_DIR = REPO_ROOT / "docs" / "mncs-reconstruction" / "evidence"

RUNNER_IDENTITY = "mnel-mncs-native-one-step-runner/0.1"
STUDY_IDENTITY = "mnel-native-fixed-point-one-step-v1"
INTERPRETATION = (
    "bounded_native_execution_and_cross_backend_observation; "
    "not_universal_equivalence_not_conformance_not_authority"
)

BACKENDS = {
    "mncs-research-bytecode": "research-bytecode",
    "mncs-portable-wasm-mvp": "portable-wasm",
    "mncs-llvm-ir": "llvm",
    "mncs-c11": "c11",
    "mncs-cranelift": "cranelift",
}

DEFAULT_LIBRARY_ROOT = REPO_ROOT.parent / "mncs-language" / "library"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_mncs(
    mncs_bin: list[str], backend: str, out_dir: Path, library_path: Path | None
) -> tuple[int, dict[str, Any] | None]:
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
    environment = {**os.environ}
    if library_path:
        environment["MNCS_LIBRARY_PATH"] = str(library_path)
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    result_path = out_dir / "result.json"
    if result_path.exists():
        try:
            return completed.returncode, json.loads(result_path.read_text())
        except json.JSONDecodeError:
            pass
    start = completed.stdout.find("{")
    if start >= 0:
        try:
            return completed.returncode, json.loads(completed.stdout[start:])
        except json.JSONDecodeError:
            pass
    return completed.returncode, None


def field_map(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("record"), dict):
        return {}
    fields = value["record"].get("fields", [])
    return {item[0]: item[1] for item in fields if isinstance(item, list) and len(item) == 2}


def integer_value(value: dict[str, Any] | None) -> int | None:
    if not isinstance(value, dict) or not isinstance(value.get("integer"), dict):
        return None
    raw = value["integer"].get("value")
    return raw if isinstance(raw, int) else None


def boolean_value(value: dict[str, Any] | None) -> bool | None:
    if not isinstance(value, dict) or not isinstance(value.get("boolean"), dict):
        return None
    raw = value["boolean"].get("value")
    return raw if isinstance(raw, bool) else None


def decision_observation(observation: dict[str, Any]) -> dict[str, Any] | None:
    returned = observation.get("returned")
    if not isinstance(returned, list) or not returned:
        return None
    fields = field_map(returned[0])
    if not fields:
        return None
    authority = fields.get("authority", {}).get("finite", {})
    return {
        "authority": authority.get("variant_identity"),
        "request_identity": integer_value(fields.get("request_identity")),
        "source_observation_identity": integer_value(fields.get("source_observation_identity")),
        "teacher_generation_identity": integer_value(fields.get("teacher_generation_identity")),
        "student_model_identity": integer_value(fields.get("student_model_identity")),
        "artifact_identity": integer_value(fields.get("artifact_identity")),
        "calibration_identity": integer_value(fields.get("calibration_identity")),
        "decision_code": integer_value(fields.get("decision_code")),
        "confidence_milli": integer_value(fields.get("confidence")),
        "abstained": boolean_value(fields.get("abstained")),
        "out_of_distribution": boolean_value(fields.get("out_of_distribution")),
        "fallback_required": boolean_value(fields.get("fallback_required")),
        "learned_forward_passes": integer_value(fields.get("learned_forward_passes")),
        "preprocessing_operations": integer_value(fields.get("preprocessing_operations")),
        "inference_operations": integer_value(fields.get("inference_operations")),
    }


def summarize_stateful_case(case: dict[str, Any]) -> dict[str, Any]:
    execution = case.get("execution") or {}
    observations = execution.get("observations") or []
    infer = next((item for item in observations if item.get("step_id") == "infer"), None)
    final_returned = execution.get("returned") or []
    final_code = integer_value(final_returned[0]) if final_returned else None
    return {
        "case_id": case.get("case_id"),
        "status": execution.get("status"),
        "final_status": case.get("final_status_met"),
        "final_expectation": case.get("final_expectation_met"),
        "step_expectations": case.get("step_expectations_met"),
        "final_decision_code": final_code,
        "transition_statuses": [
            {"step_id": item.get("step_id"), "status": item.get("status"), "steps": item.get("steps")}
            for item in observations
        ],
        "decision": decision_observation(infer) if infer else None,
        "failure": execution.get("failure"),
    }


def summarize_backend(rc: int, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"outcome": "runner-error", "exit_code": rc}
    cases = [summarize_stateful_case(case) for case in payload.get("stateful_cases", [])]
    executed = all(case["status"] == "returned" for case in cases) and bool(cases)
    expectations = all(
        case["final_status"] is True
        and case["final_expectation"] is True
        and case["step_expectations"] is True
        for case in cases
    )
    return {
        "outcome": "corpus-executed" if executed else "execution-failed",
        "exit_code": rc,
        "compiler_status": payload.get("status"),
        "execution_status": "PASS" if executed and expectations else "FAIL",
        "unresolved_reasons": payload.get("unresolved_reasons") or [],
        "stateful_case_count": len(cases),
        "stateful_cases": cases,
        "all_expectations_met": expectations,
        "artifact_identity": (payload.get("artifact") or {}).get("identity"),
        "backend_identity": (payload.get("backend") or {}).get("identity"),
    }


def python_reference() -> dict[str, Any]:
    """Run the existing Python teacher and one-step student as a control."""

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mnel.one_step_specialist import (
        _evaluate_reference,
        _reference_teacher,
        _study_holdout_rows,
        _study_training_rows,
        build_distillation_dataset,
        calibrate_one_step_student,
        collect_teacher_observation,
        train_one_step_student,
    )

    teacher = _reference_teacher()
    observations = tuple(
        collect_teacher_observation(teacher, row, independent_target=row.get("expected"))
        for row in _study_training_rows()
    )
    dataset = build_distillation_dataset(observations)
    student = train_one_step_student(dataset)
    student, calibration = calibrate_one_step_student(student, _study_holdout_rows())
    evaluation = _evaluate_reference(teacher, student, _study_holdout_rows())
    return {
        "teacher": {
            "provider_identity": teacher.provider_id,
            "model_identity": teacher.model_identity,
            "generation_identity": teacher.generation_identity,
            "architecture_identity": teacher.architecture_identity,
            "decisions": evaluation["teacher"]["decisions"],
            "abstentions": evaluation["teacher"]["abstentions"],
        },
        "python_one_step_control": {
            "model_identity": student.model_identity,
            "calibration_identity": calibration.calibration_identity,
            "decisions": evaluation["student"]["decisions"],
            "abstentions": evaluation["student"]["abstentions"],
            "learned_forward_passes": evaluation["student"]["learned_forward_passes"],
            "inference_operations": evaluation["student"]["inference_operations"],
        },
        "holdout_rows": list(evaluation["expected"]),
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
        default=REPO_ROOT / "target" / "mnel-native-one-step",
        help="directory for per-backend outputs",
    )
    parser.add_argument("--backend", choices=sorted(BACKENDS), action="append")
    parser.add_argument("--library-path", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument("--skip-python-reference", action="store_true")
    args = parser.parse_args()

    if not SOURCE_PATH.exists() or not CORPUS_PATH.exists():
        print("native one-step source or corpus is missing", file=sys.stderr)
        return 2
    corpus = json.loads(CORPUS_PATH.read_text())
    selected = args.backend or list(BACKENDS)
    backends: list[dict[str, Any]] = []
    any_failure = False
    for backend in selected:
        tag = BACKENDS[backend]
        rc, payload = run_mncs(args.mncs_bin, backend, args.work_dir / tag, args.library_path)
        summary = summarize_backend(rc, payload)
        summary["backend"] = backend
        backends.append(summary)
        any_failure |= summary.get("execution_status") != "PASS"

    semantic_codes = {
        item["backend"]: {
            case["case_id"]: case.get("final_decision_code")
            for case in item.get("stateful_cases", [])
        }
        for item in backends
        if item.get("stateful_cases")
    }
    cross_backend_agreement = bool(semantic_codes) and len({json.dumps(item, sort_keys=True) for item in semantic_codes.values()}) == 1
    reference = None if args.skip_python_reference else python_reference()
    native_cases = {
        "relevant-in-domain": {"expected_code": 1, "reference_decision": "relevant"},
        "out-of-distribution-abstains": {"expected_code": 0, "reference_decision": "ABSTAIN"},
        "generation-mismatch-abstains": {
            "expected_code": 0,
            "reference_decision": "not-invoked; lineage gate",
        },
        "schema-mismatch-abstains": {
            "expected_code": 0,
            "reference_decision": "not-invoked; schema gate",
        },
    }
    reference_comparison_backend = next(
        (item for item in backends if item.get("stateful_cases")), None
    )
    native_reference_comparison = {
        case_id: {
            "native_decision_code": next(
                (
                    case.get("final_decision_code")
                    for case in reference_comparison_backend.get("stateful_cases", [])
                    if case.get("case_id") == case_id
                ),
                None,
            )
            if reference_comparison_backend
            else None,
            "reference_decision": details["reference_decision"],
            "comparison_scope": (
                "bounded decision-code mapping"
                if not details["reference_decision"].startswith("not-invoked")
                else "native deterministic gate precedes teacher"
            ),
        }
        for case_id, details in native_cases.items()
    }
    evidence = {
        "schema_version": "0.1",
        "identity_kind": "bounded-native-one-step-study-record",
        "study": STUDY_IDENTITY,
        "runner_identity": RUNNER_IDENTITY,
        "interpretation": INTERPRETATION,
        "source": {
            "path": str(SOURCE_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(SOURCE_PATH),
            "language_profile": "0.10",
            "module": "mnel.one_step",
            "native_semantics": {
                "numeric_representation": "signed i32 fixed-point lanes with scale 1000",
                "training": "bounded eight-epoch SGD-style fixed-point updates over six VERIFIED rows",
                "retention": "eight rows retained; REJECTED_TEACHER and UNKNOWN rows are excluded from updates",
                "inference": "deterministic lineage/schema/OOD gates followed by at most one learned forward pass",
                "identity": "bounded u64 fingerprints for this executable slice; not cryptographic identities",
            },
            "authority": "DIAGNOSTIC_ONLY; native decisions carry explicit fallback_required and never invoke or replace the teacher",
        },
        "corpus": {
            "path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_file(CORPUS_PATH),
            "schema_version": corpus.get("schema_version"),
            "stateful_case_count": len(corpus.get("stateful_cases", [])),
            "cases": native_cases,
            "bounded_state_policy": "each trace trains once, then passes the logical StudentArtifact through infer and decision_code",
        },
        "mncs_library": {
            "bindings": ["mncs.core.numeric.v1", "mncs.core.random.v1", "mncs.core.vector.v1"],
            "library_path": str(args.library_path) if args.library_path else None,
        },
        "backends": backends,
        "native_reference_comparison": native_reference_comparison,
        "cross_backend_agreement": {
            "status": "AGREEMENT_OVER_DECLARED_STATEFUL_CORPUS" if cross_backend_agreement and not any_failure else "MISMATCH_OR_FAILURE",
            "final_decision_codes": semantic_codes,
        },
        "python_reference": reference,
        "cost_and_performance_boundary": {
            "native_student": "preprocessing_operations and inference_operations are explicit fields in StudentDecision; learned_forward_passes is 0 or 1",
            "teacher": "teacher fallback is an outer Python control-plane action and is not called by mnel.one_step.infer",
            "claim": "this study records bounded semantic and operation-count evidence, not wall-clock speed or a universal cost reduction",
        },
        "non_claims": [
            "compiler UNKNOWN status reflects retained checked-arithmetic obligations and is not silently promoted to PASS",
            "cross-backend agreement is finite evidence, not semantic equivalence, compiler correctness, conformance, or production authorization",
            "native u64 fingerprints are bounded lineage fields in this slice, not cryptographic hashes",
            "the diagnostic-only student cannot authorize, promote, or issue evaluator verdicts",
        ],
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVIDENCE_DIR / "mnel-native-one-step-study.json"
    out_path.write_text(json.dumps(evidence, indent=1) + "\n")
    print(
        "comparison: "
        + ("AGREEMENT_OVER_DECLARED_STATEFUL_CORPUS" if cross_backend_agreement and not any_failure else "MISMATCH_OR_FAILURE")
    )
    for item in backends:
        print(
            f"  {item['backend']}: {item.get('execution_status', item.get('outcome'))} "
            f"(compiler_status={item.get('compiler_status')})"
        )
    print(f"evidence: {out_path.relative_to(REPO_ROOT)}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
