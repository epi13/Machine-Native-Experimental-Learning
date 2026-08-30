from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mnel.core import canonical_digest
from mnel.one_step_specialist import (
    ARTIFACT_SCHEMA,
    DECISION_SCHEMA,
    DISTILLATION_DATASET_SCHEMA,
    DISTILLATION_RECORD_SCHEMA,
    OneStepSpecialistError,
    OneStepStudentModel,
    StudentOperatingEnvelope,
    TargetStatus,
    build_distillation_dataset,
    build_one_step_reference_artifacts,
    calibrate_one_step_student,
    collect_teacher_observation,
    infer_with_teacher_fallback,
    train_one_step_student,
)
from mnel.recurrent_specialist import (
    SpecialistContextState,
    calibrate_recurrent_specialist,
    train_recurrent_specialist,
)

TRAINING_ROWS = (
    {"record_id": "train-relevant-a", "features": [900, 820, 760, 880], "expected": "relevant"},
    {"record_id": "train-relevant-b", "features": [820, 760, 700, 800], "expected": "relevant"},
    {"record_id": "train-relevant-c", "features": [760, 700, 660, 720], "expected": "relevant"},
    {"record_id": "train-relevant-d", "features": [700, 740, 720, 760], "expected": "relevant"},
    {"record_id": "train-irrelevant-a", "features": [120, 180, 160, 100], "expected": "irrelevant"},
    {"record_id": "train-irrelevant-b", "features": [220, 120, 180, 160], "expected": "irrelevant"},
    {"record_id": "train-irrelevant-c", "features": [160, 220, 120, 180], "expected": "irrelevant"},
    {"record_id": "train-irrelevant-d", "features": [260, 180, 220, 200], "expected": "irrelevant"},
)


def make_teacher():
    base_rows = tuple(
        {**row, "label": row["expected"]} for row in TRAINING_ROWS[:2] + TRAINING_ROWS[4:6]
    )
    teacher = train_recurrent_specialist(
        base_rows,
        target_role="forge.evidence-relevance",
        generation_identity=canonical_digest({"test": "teacher-generation"}),
    )
    return calibrate_recurrent_specialist(
        teacher,
        (
            *base_rows,
            {"record_id": "calibration", "features": [760, 700, 660, 720], "label": "relevant"},
        ),
    )[0]


def make_dataset():
    teacher = make_teacher()
    rows = (
        *TRAINING_ROWS,
        {"record_id": "rejected", "features": [900, 820, 760, 880], "expected": "irrelevant"},
        {"record_id": "unknown", "features": [640, 620, 600, 580]},
        {"record_id": "abstention", "features": [500, 500, 500, 500], "expected": "ABSTAIN"},
    )
    observations = tuple(collect_teacher_observation(teacher, row) for row in rows)
    return teacher, rows, observations, build_distillation_dataset(observations)


class OneStepSpecialistTests(unittest.TestCase):
    def test_teacher_observations_retain_lineage_and_do_not_promote_targets(self) -> None:
        teacher, _, observations, dataset = make_dataset()
        statuses = [item.target_status for item in observations]
        self.assertEqual(statuses[:8], [TargetStatus.VERIFIED] * 8)
        self.assertEqual(
            statuses[-3:],
            [TargetStatus.REJECTED_TEACHER, TargetStatus.UNKNOWN, TargetStatus.ABSTENTION],
        )
        self.assertEqual(len(dataset.observations), 11)
        self.assertEqual(len(dataset.training_observations), 8)
        self.assertEqual(dataset.teacher_model_identity, teacher.model_identity)
        self.assertEqual(dataset.to_dict()["schema"], "mnel-distillation-dataset/0.1")
        self.assertEqual(observations[0].to_dict()["schema"], DISTILLATION_RECORD_SCHEMA)
        self.assertNotIn("verdict", observations[0].to_dict())
        self.assertEqual(
            dataset.dataset_identity, build_distillation_dataset(observations).dataset_identity
        )
        reloaded = type(dataset).load(dataset.serialize())
        self.assertEqual(reloaded.dataset_identity, dataset.dataset_identity)
        self.assertEqual(reloaded.distillation_identity, dataset.distillation_identity)
        self.assertEqual(reloaded.training_targets, dataset.training_targets)
        broken = json.loads(dataset.serialize())
        broken["teacher_model_identity"] = canonical_digest({"wrong": "teacher"})
        with self.assertRaises(OneStepSpecialistError):
            type(dataset).load(broken)

    def test_student_is_genuinely_trained_reloadable_and_one_pass(self) -> None:
        teacher, _, _, dataset = make_dataset()
        student, calibration = calibrate_one_step_student(
            train_one_step_student(dataset, distilled=True, seed=23),
            TRAINING_ROWS,
        )
        reloaded = OneStepStudentModel.load(student.serialize())
        self.assertEqual(student.model_identity, reloaded.model_identity)
        self.assertEqual(student.artifact_identity, reloaded.artifact_identity)
        decision = student.infer([780, 740, 700, 760])
        replay = reloaded.infer([780, 740, 700, 760])
        self.assertEqual(decision.decision, "relevant")
        self.assertEqual(decision.learned_forward_passes, 1)
        self.assertGreater(decision.inference_operations, 0)
        self.assertEqual(decision.decision_identity, replay.decision_identity)
        self.assertEqual(decision.to_dict()["schema"], DECISION_SCHEMA)
        self.assertEqual(student.calibration_identity, calibration.calibration_identity)
        self.assertEqual(student.teacher_model_identity, teacher.model_identity)
        self.assertEqual(student.to_dict()["architecture"]["learned_forward_passes"], 1)
        self.assertNotIn("promotion", decision.to_dict())

    def test_low_confidence_ood_and_incompatible_context_abstain(self) -> None:
        _, _, _, dataset = make_dataset()
        student = train_one_step_student(dataset)
        zero = replace(
            student,
            input_weights=tuple((0.0,) * 4 for _ in range(4)),
            hidden_bias=(0.0,) * 4,
            output_weights=tuple((0.0,) * 4 for _ in range(2)),
            output_bias=(0.0,) * 2,
            checkpoint_identity=canonical_digest({"test": "zero-checkpoint"}),
            operating_envelope=StudentOperatingEnvelope(
                minimum_confidence=700, maximum_distance=1400
            ),
            model_identity="",
            artifact_identity="",
        )
        zero = replace(zero, model_identity=zero.content_identity)
        low = zero.infer([780, 740, 700, 760])
        self.assertTrue(low.abstained)
        self.assertEqual(low.escalation_reason, "insufficient-calibrated-confidence")
        self.assertEqual(low.learned_forward_passes, 1)
        ood = student.infer([1000, -1000, 1000, -1000])
        self.assertTrue(ood.abstained)
        self.assertTrue(ood.out_of_distribution)
        self.assertEqual(ood.learned_forward_passes, 0)
        wrong_context = SpecialistContextState(
            provider_identity=student.teacher_provider_identity,
            generation_identity=canonical_digest({"wrong": "generation"}),
            role_identity=student.target_role,
            source_observation_identities=(),
            feature_mean=(0, 0, 0, 0),
        )
        incompatible = student.infer([780, 740, 700, 760], context=wrong_context)
        self.assertTrue(incompatible.abstained)
        self.assertTrue(incompatible.escalation_reason.startswith("unsupported-envelope:"))
        self.assertEqual(incompatible.learned_forward_passes, 0)
        malformed = student.infer([780, 740, 700, 760], context={"feature_mean": [0, 0, 0, 0]})  # type: ignore[arg-type]
        self.assertTrue(malformed.abstained)
        self.assertEqual(malformed.learned_forward_passes, 0)

    def test_fallback_is_explicit_and_lineage_bound(self) -> None:
        teacher, _, _, dataset = make_dataset()
        student = calibrate_one_step_student(train_one_step_student(dataset), TRAINING_ROWS)[0]
        result = infer_with_teacher_fallback(student, teacher, [1000, -1000, 1000, -1000])
        self.assertTrue(result.student_decision.abstained)
        self.assertTrue(result.fallback_invoked)
        self.assertIsNotNone(result.teacher_decision)
        self.assertEqual(result.final_decision, "ABSTAIN")
        self.assertEqual(result.to_dict()["student_decision"]["decision"], "ABSTAIN")
        other = replace(
            teacher,
            generation_identity=canonical_digest({"test": "other-generation"}),
            model_identity="",
        )
        with self.assertRaises(OneStepSpecialistError):
            infer_with_teacher_fallback(student, other, [1000, -1000, 1000, -1000])

    def test_authority_fields_and_unknown_artifact_fields_are_rejected(self) -> None:
        _, _, _, dataset = make_dataset()
        student = train_one_step_student(dataset)
        payload = json.loads(student.serialize())
        payload["verdict"] = "PASS"
        with self.assertRaises(OneStepSpecialistError):
            OneStepStudentModel.load(payload)
        payload = json.loads(student.serialize())
        payload["architecture"]["authority"] = "verifier"
        payload["artifact_identity"] = canonical_digest(
            {key: value for key, value in payload.items() if key != "artifact_identity"}
        )
        with self.assertRaises(OneStepSpecialistError):
            OneStepStudentModel.load(payload)
        payload = json.loads(student.serialize())
        payload["unexpected"] = True
        payload["artifact_identity"] = canonical_digest(
            {key: value for key, value in payload.items() if key != "artifact_identity"}
        )
        with self.assertRaises(OneStepSpecialistError):
            OneStepStudentModel.load(payload)
        payload = json.loads(student.serialize())
        payload["architecture"]["learned_forward_passes"] = 2
        payload["artifact_identity"] = canonical_digest(
            {key: value for key, value in payload.items() if key != "artifact_identity"}
        )
        with self.assertRaises(OneStepSpecialistError):
            OneStepStudentModel.load(payload)

    def test_reference_study_has_teacher_student_baseline_control_and_measurements(self) -> None:
        first = build_one_step_reference_artifacts(tempfile.mkdtemp())
        second = build_one_step_reference_artifacts(tempfile.mkdtemp())
        report = first["report"]
        self.assertEqual(report["study_identity"], second["report"]["study_identity"])
        self.assertEqual(report["target_role"], "forge.evidence-relevance")
        self.assertEqual(
            report["distillation"]["rejected_or_unresolved_statuses"],
            ["rejected-teacher-target", "unknown", "abstention"],
        )
        self.assertEqual(report["claim_boundary"]["success_claim_supported"], False)
        self.assertEqual(report["evaluations"]["iterative_teacher"]["abstentions"], 2)
        self.assertEqual(
            report["evaluations"]["one_step_distilled_student"]["learned_forward_passes"][:4],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            report["evaluations"]["no_distillation_control"]["learned_forward_passes"][:4],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            report["evaluations"]["classical_nearest_centroid_baseline"]["abstentions"], 2
        )
        self.assertTrue(report["evaluations"]["fallback"]["fallback_invocations"] >= 1)
        self.assertTrue(report["students"]["distilled"]["artifact_bytes"] > 0)

    def test_schema_files_are_explicit_and_strict(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name, schema, title in (
            ("mnel-one-step-specialist.schema.json", ARTIFACT_SCHEMA, "artifact"),
            ("mnel-one-step-specialist-decision.schema.json", DECISION_SCHEMA, "decision"),
            ("mnel-distillation-record.schema.json", DISTILLATION_RECORD_SCHEMA, "record"),
            ("mnel-distillation-dataset.schema.json", DISTILLATION_DATASET_SCHEMA, "dataset"),
        ):
            value = json.loads((root / "schemas" / name).read_text())
            self.assertEqual(value["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertFalse(value.get("additionalProperties", True))
            self.assertIn(title, value["title"].lower())


if __name__ == "__main__":
    unittest.main()
