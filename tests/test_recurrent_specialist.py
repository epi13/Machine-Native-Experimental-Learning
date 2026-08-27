from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mnel.core import canonical_digest
from mnel.recurrent_specialist import (
    RecurrentSpecialistModel,
    SpecialistError,
    build_reference_artifacts,
    calibrate_recurrent_specialist,
    context_update,
    empty_context,
    infer_batch,
    train_recurrent_specialist,
)

ROWS = (
    {"record_id": "train-a", "features": [900, 820, 760, 880], "label": "relevant"},
    {"record_id": "train-b", "features": [820, 760, 700, 800], "label": "relevant"},
    {"record_id": "train-c", "features": [120, 180, 160, 100], "label": "irrelevant"},
    {"record_id": "train-d", "features": [220, 120, 180, 160], "label": "irrelevant"},
)


def calibrated_model():
    model = train_recurrent_specialist(
        ROWS,
        target_role="forge.evidence-relevance",
        generation_identity=canonical_digest({"test": "generation-0"}),
    )
    return calibrate_recurrent_specialist(
        model,
        (*ROWS, {"record_id": "calibration-boundary", "features": [760, 700, 660, 720], "label": "relevant"}),
    )[0]


def test_context_and_recurrent_reasoning_are_separate_and_bounded() -> None:
    model = calibrated_model()
    context = context_update(empty_context(model), "observation-1", [500, 500, 500, 500])
    decision = model.infer([760, 700, 660, 720], context=context)
    assert decision.decision == "relevant"
    assert decision.context_state_identity == context.state_identity
    assert 1 <= decision.reasoning_iterations <= 4
    assert decision.halting_reason in {"converged-mask-deactivated", "budget-exhausted"}
    assert decision.authority == "diagnostic-only"


def test_abstention_is_explicit_for_ood_and_budget_cannot_expand() -> None:
    model = calibrated_model()
    result = model.infer([1000, -1000, 1000, -1000])
    assert result.abstained is True
    assert result.decision == "ABSTAIN"
    assert result.escalation_reason == "out-of-distribution-distance"
    with pytest.raises(SpecialistError):
        model.infer([700, 700, 700, 700], max_iterations=5)


def test_artifact_reload_and_batch_replay_preserve_decisions() -> None:
    model = calibrated_model()
    payload = model.serialize()
    reloaded = RecurrentSpecialistModel.load(payload)
    queries = [
        {"query_id": "q1", "features": [760, 700, 660, 720]},
        {"query_id": "q2", "features": [160, 220, 120, 180]},
    ]
    first = infer_batch(model, queries)
    second = infer_batch(reloaded, queries)
    assert [item.decision for item in first] == ["relevant", "irrelevant"]
    assert [item.to_dict()["decision_identity"] for item in first] == [item.to_dict()["decision_identity"] for item in second]
    broken = json.loads(payload)
    broken["class_centroids"]["relevant"][0] += 1
    with pytest.raises(SpecialistError):
        RecurrentSpecialistModel.load(broken)


def test_provider_protocol_is_executable_and_structured(tmp_path: Path) -> None:
    model = calibrated_model()
    request = {
        "protocol_version": "mnel-recurrent-specialist-provider/0.1",
        "type": "infer",
        "request_id": canonical_digest({"test": "request"}),
        "artifact": json.loads(model.serialize()),
        "queries": [{"query_id": "q1", "features": [760, 700, 660, 720], "source_record_identity": "source-1"}],
    }
    process = subprocess.run(
        [sys.executable, "-m", "mnel.recurrent_provider"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        check=True,
    )
    response = json.loads(process.stdout)
    assert response["type"] == "inference_response"
    assert response["results"][0]["decision"] == "relevant"
    assert response["results"][0]["source_observation_identities"] == ["source-1"]


def test_reference_artifacts_include_generation_calibration_and_cost(tmp_path: Path) -> None:
    result = build_reference_artifacts(tmp_path)
    evidence = json.loads(Path(result["evidence"]).read_text())
    assert set(evidence["models"]) == {"forge", "control"}
    assert all(item["reload_equivalent"] for item in evidence["models"].values())
    assert evidence["evaluations"]["forge"]["abstentions"] >= 1
    assert evidence["evaluations"]["control"]["abstentions"] >= 1
    assert evidence["cost_measurements"]["larger_model_calls_avoided"] == 2
