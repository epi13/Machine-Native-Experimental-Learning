import unittest
import tempfile
import json
from pathlib import Path

from mnel.core import TransferStatus, Visibility
from mnel.distillation import (
    AblationKind,
    AblationSpec,
    DistillationError,
    StudyArm,
    StudyArmKind,
    StudyDataAccess,
    StudySpecification,
    TransferWorkflow,
    VisibilityViolation,
    build_distilled_strategy,
    calculate_calibration,
    make_study_record,
    reference_feature_groups,
    RetrievalIndex,
    run_reference_distill_study,
    shuffle_attributions,
)
from mnel.reference_provider import train_transition_frequency
from mnel.snapshots import SnapshotStore, transition_snapshot


class DistillationTests(unittest.TestCase):
    def setUp(self) -> None:
        identities = {
            "producer_identity": "producer:v1",
            "source_identity": "source:v1",
            "dependency_identity": "dependency:v1",
            "feature_extractor_identity": "extractor:v1",
        }
        self.snapshots = SnapshotStore()
        self.first = transition_snapshot(b"cold", b"warm", **identities)
        self.second = transition_snapshot(b"hot", b"warm", **identities)
        self.snapshots.register(self.first)
        self.snapshots.register(self.second)
        self.records = (
            make_study_record(
                "experience-episode",
                {"snapshot_identity": self.first.snapshot_identity, "artifact_type": "routing", "tags": ["transition"]},
                record_identity="sha256:episode-a",
            ),
            make_study_record(
                "experience-episode",
                {"snapshot_identity": self.second.snapshot_identity, "artifact_type": "routing", "tags": ["transition"]},
                record_identity="sha256:episode-b",
            ),
            make_study_record(
                "transfer-outcome",
                {"outcome": "supported"},
                visibility=Visibility.TRANSFER_HIDDEN,
                record_identity="sha256:hidden",
            ),
            make_study_record(
                "future-final",
                {"outcome": "final"},
                visibility=Visibility.FUTURE_FINAL,
                record_identity="sha256:future",
            ),
        )

    def test_visibility_access_fails_closed_and_groups_preserve_sources(self) -> None:
        development = StudyDataAccess.development(self.records)
        self.assertEqual(len(development.records()), 2)
        with self.assertRaises(VisibilityViolation):
            development.get("sha256:hidden")
        with self.assertRaises(VisibilityViolation):
            development.get("sha256:future")
        groups = reference_feature_groups(development)
        self.assertEqual(sorted(item.source_record_ids for item in groups), [("sha256:episode-a", "sha256:episode-b")])
        self.assertNotEqual(groups[0].group_identity, "")

    def test_hidden_records_cannot_train_provider(self) -> None:
        development = StudyDataAccess.development(self.records)
        model = train_transition_frequency(development, self.snapshots)
        reloaded = type(model).load(model.serialize())
        self.assertEqual(model.model_identity, reloaded.model_identity)
        self.assertEqual(model.artifact_identity, reloaded.artifact_identity)
        self.assertEqual(model.infer(self.first).to_dict(), reloaded.infer(self.first).to_dict())
        with self.assertRaises(DistillationError):
            train_transition_frequency(StudyDataAccess.transfer_evaluator(self.records), self.snapshots)
        with self.assertRaises(VisibilityViolation):
            RetrievalIndex().add_source_records((self.records[2],))

    def test_provider_abstains_on_unseen_transition_and_authority_fields_are_rejected(self) -> None:
        model = train_transition_frequency(StudyDataAccess.development(self.records), self.snapshots)
        unseen = transition_snapshot(
            b"unknown",
            b"state",
            producer_identity="producer:v1",
            source_identity="source:unseen",
            dependency_identity="dependency:v1",
            feature_extractor_identity="extractor:v1",
        )
        observation = model.infer(unseen)
        self.assertTrue(observation.abstained)
        self.assertTrue(observation.out_of_distribution)
        self.assertNotIn("verdict", observation.to_dict())
        with self.assertRaises(DistillationError):
            make_study_record("episode", {"verdict": "PASS"})

    def test_transfer_prediction_is_frozen_and_same_candidate_repair_is_rejected(self) -> None:
        strategy = build_distilled_strategy(
            trigger_conditions=("transition",),
            preconditions=("routing",),
            expected_effect={"metric": "increase"},
            known_failure_modes=("unsupported",),
            negative_memory_ids=("memory:v1",),
            counterexample_record_ids=("record:counterexample",),
            causal_attribution_ids=("attribution:v1",),
            supporting_source_record_ids=("episode:v1",),
            transfer_evidence_ids=(),
            scope={"artifact_type": "routing"},
        )
        workflow = TransferWorkflow()
        prediction = workflow.freeze_prediction(
            strategy,
            transfer_environment_identity="environment:hidden",
            predicted_effect={"metric": "increase"},
        )
        evaluation = workflow.finalize(
            prediction,
            observed_outcome_identity="outcome:hidden",
            evaluator_evidence_identity="evidence:evaluator",
            status=TransferStatus.FAILED,
        )
        with self.assertRaises(VisibilityViolation):
            workflow.reject_same_candidate_repair(strategy.strategy_identity, evaluation)
        self.assertNotIn("verdict", evaluation.to_dict())

    def test_controls_are_deterministic_and_equal_budget_is_checked(self) -> None:
        shuffled_a = shuffle_attributions(("a", "b", "c"), seed=11, source_study_identity="study:v1")
        shuffled_b = shuffle_attributions(("a", "b", "c"), seed=11, source_study_identity="study:v1")
        self.assertEqual(shuffled_a.to_dict(), shuffled_b.to_dict())
        budget = {"operations": 10, "wall_seconds": 10, "candidates": 2}
        arms = (
            StudyArm("a", StudyArmKind.RANDOM, ("development",), ("hidden",), "random", (), False, False, 0, budget),
            StudyArm("b", StudyArmKind.ATTRIBUTION, ("development",), ("hidden",), "lineage", (), True, True, 0, budget),
        )
        specification = StudySpecification(
            "study:v1", "development:v1", "hidden:v1", arms,
            (AblationSpec("ablation", AblationKind.NEGATIVE_MEMORY, arms[0].arm_identity, 11, {}),),
            budget,
        )
        self.assertTrue(specification.study_identity)
        with self.assertRaises(DistillationError):
            StudySpecification(
                "study:v2", "development:v1", "hidden:v1", arms,
                (), {"operations": 9, "wall_seconds": 10, "candidates": 2},
            )

    def test_calibration_and_reference_study_are_measured(self) -> None:
        metrics = calculate_calibration(
            (0.9, None, 0.1), (True, False, False),
            dataset_identity="dataset:v1", model_identity="model:v1",
            out_of_distribution=(False, True, False),
        )
        self.assertEqual(metrics.count, 3)
        self.assertGreaterEqual(metrics.coverage, 0.0)
        result = run_reference_distill_study()
        repeat = run_reference_distill_study()
        self.assertEqual(result["study_identity"], repeat["study_identity"])
        self.assertTrue(result["report"]["negative_memory_demoted_strategy"])
        self.assertEqual(result["report"]["transfer_status"], "supported")
        self.assertTrue(result["report"]["retrieval_metrics"]["metric_identity"])

        with tempfile.TemporaryDirectory() as workspace:
            written = run_reference_distill_study(workspace)
            self.assertTrue(written["report"]["ledger"]["valid"])
            self.assertEqual(written["report"]["ledger"]["record_count"], 35)

    def test_distillation_schema_covers_durable_reference_records(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schemas" / "mnel-distillation.schema.json").read_text())
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        for definition in ("group", "negativeMemory", "strategy", "studySpec", "providerObservation", "studyReport"):
            self.assertIn(definition, schema["$defs"])


if __name__ == "__main__":
    unittest.main()
