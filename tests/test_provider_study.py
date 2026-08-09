import json
import tempfile
import unittest
from pathlib import Path

from mnel.core import Visibility
from mnel.distillation import CalibrationDataAccess, CalibrationRecord, DistillationError, StudyDataAccess, VisibilityViolation, make_study_record
from mnel.provider_study import (
    ProviderBinding,
    ProviderLifecycleState,
    ProviderLifecycleStore,
    ProviderStudyError,
    RoutingPolicy,
    RoutingPolicyKind,
    _phi,
    measure_energy,
    _ReferenceHeuristic,
    _RandomControl,
    run_reference_portfolio_study,
    select_providers,
)
from mnel.reference_provider import TabularCentroidModel, train_tabular_centroid
from mnel.snapshots import SnapshotStore, tabular_snapshot


class ProviderStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        identities = {
            "producer_identity": "provider-study-test-producer",
            "source_identity": "provider-study-test-source",
            "dependency_identity": "provider-study-test-dependency",
            "feature_extractor_identity": "provider-study-test-extractor",
        }
        self.store = SnapshotStore()
        self.normal_a = tabular_snapshot(((0.2, 0.4),), **identities)
        self.normal_b = tabular_snapshot(((0.3, 0.5),), **identities)
        self.calibration = tabular_snapshot(((0.25, 0.45),), **identities)
        self.ood = tabular_snapshot(((10.0, 10.0),), **identities)
        for snapshot in (self.normal_a, self.normal_b, self.calibration, self.ood):
            self.store.register(snapshot)
        self.training = (
            make_study_record("tabular-episode", {"snapshot_identity": self.normal_a.snapshot_identity}, record_identity="sha256:training-a"),
            make_study_record("tabular-episode", {"snapshot_identity": self.normal_b.snapshot_identity}, record_identity="sha256:training-b"),
        )
        self.access = StudyDataAccess.development(self.training)
        self.calibration_access = CalibrationDataAccess.development(
            (
                CalibrationRecord(self.calibration.snapshot_identity, "sha256:cal-source", 1),
                CalibrationRecord(self.ood.snapshot_identity, "sha256:cal-source", 0),
            )
        )

    def test_tabular_provider_is_structurally_distinct_and_reloadable(self) -> None:
        model = train_tabular_centroid(self.access, self.store, self.calibration_access, record_type="tabular-episode")
        self.assertEqual(model.provider_family, "nearest-centroid")
        self.assertEqual(model.supported_snapshot_types, ("tabular",))
        reloaded = TabularCentroidModel.load(model.serialize())
        self.assertEqual(model.model_identity, reloaded.model_identity)
        self.assertEqual(model.artifact_identity, reloaded.artifact_identity)
        self.assertEqual(model.infer(self.calibration).to_dict(), reloaded.infer(self.calibration).to_dict())
        self.assertTrue(model.infer(self.ood).out_of_distribution)
        with self.assertRaises(DistillationError):
            model.infer(self.normal_a.__class__.build(
                snapshot_type="transition",
                schema_version=1,
                producer_identity="p",
                source_identity="s",
                dependency_identity="d",
                feature_extractor_identity="f",
                payload=b"not-a-transition",
            ))

    def test_hidden_calibration_is_rejected_before_training(self) -> None:
        hidden = CalibrationRecord(
            self.calibration.snapshot_identity,
            "sha256:hidden-calibration",
            1,
            visibility=Visibility.TRANSFER_HIDDEN,
        )
        with self.assertRaises(VisibilityViolation):
            CalibrationDataAccess.development((hidden,))

    def test_development_view_excludes_hidden_records(self) -> None:
        hidden = make_study_record(
            "transfer-outcome",
            {"provider_id": "provider:a"},
            visibility=Visibility.TRANSFER_HIDDEN,
            record_identity="sha256:hidden-study-record",
        )
        view = StudyDataAccess.development(self.training + (hidden,))
        self.assertEqual(view.records(), self.training)
        with self.assertRaises(VisibilityViolation):
            view.get(hidden.identity)

    def test_routing_is_reproducible_and_explicitly_incompatible(self) -> None:
        heuristic = ProviderBinding(_ReferenceHeuristic(), _ReferenceHeuristic.load, "explicit", "control", "low", "baseline")
        random_control = ProviderBinding(_RandomControl(), _RandomControl.load, "random", "control", "low", "baseline")
        bindings = {heuristic.provider_id: heuristic, random_control.provider_id: random_control}
        from mnel.provider_study import PortfolioCase

        portfolio_case = PortfolioCase(
            "routing-case",
            self.normal_a,
            1,
            False,
            "in-distribution",
            "sha256:reference",
        )
        policy = RoutingPolicy("random", RoutingPolicyKind.RANDOM, tuple(bindings), 1, 17)
        first = select_providers(policy, portfolio_case, bindings)
        second = select_providers(policy, portfolio_case, bindings)
        self.assertEqual(first.to_dict(), second.to_dict())
        incompatible = RoutingPolicy("single", RoutingPolicyKind.SINGLE_PROVIDER, ("missing",), 1, 17)
        self.assertEqual(select_providers(incompatible, portfolio_case, bindings).status, "incompatible")

    def test_lifecycle_requires_evidence_and_preserves_quarantine_retirement(self) -> None:
        store = ProviderLifecycleStore()
        candidate = store.register_candidate("provider:a", "sha256:artifact-a", "sha256:evidence-a", "sha256:policy")
        self.assertEqual(candidate.to_state, ProviderLifecycleState.CANDIDATE)
        store.transition("provider:a", ProviderLifecycleState.DEVELOPMENT_ADMITTED, ("sha256:admission",), "valid", "sha256:policy")
        store.transition("provider:a", ProviderLifecycleState.QUARANTINED, ("sha256:failure",), "runtime failure", "sha256:policy")
        store.transition("provider:a", ProviderLifecycleState.RETIRED, ("sha256:retirement",), "historical retirement", "sha256:policy")
        self.assertEqual(store.state("provider:a"), ProviderLifecycleState.RETIRED)
        with self.assertRaises(ProviderStudyError):
            store.transition("provider:a", ProviderLifecycleState.DEVELOPMENT_ADMITTED, ("sha256:bad",), "invalid", "sha256:policy")

    def test_correlation_and_energy_measurements_are_guarded(self) -> None:
        self.assertAlmostEqual(_phi(((0, 0), (1, 1), (0, 0), (1, 1))), 1.0)
        self.assertIsNone(_phi(((0, 0), (0, 0))))
        self.assertEqual(measure_energy(None).status, "unavailable")
        self.assertEqual(measure_energy(lambda: 2.5).joules, 2.5)
        self.assertEqual(measure_energy(lambda: float("nan")).status, "unavailable")

    def test_routing_authority_cannot_be_injected(self) -> None:
        with self.assertRaises(DistillationError):
            RoutingPolicy("bad", RoutingPolicyKind.RANDOM, ("provider",), 1, 1, {"verdict": "FAIL"})

    def test_reference_portfolio_study_has_controls_metrics_and_valid_ledger(self) -> None:
        first = run_reference_portfolio_study()
        second = run_reference_portfolio_study()
        self.assertEqual(first["report"]["study_identity"], second["report"]["study_identity"])
        self.assertGreaterEqual(len(first["report"]["architecture_families"]), 4)
        self.assertTrue(first["report"]["pairwise_comparisons"])
        self.assertEqual(first["report"]["energy_measurement"]["status"], "unavailable")
        self.assertTrue(first["report"]["calibration_metrics"])
        self.assertEqual(first["report"]["hidden_visibility_guard"], "VisibilityViolation")
        self.assertTrue(first["report"]["admission_criteria"])
        self.assertEqual(first["report"]["lifecycle_states"]["mnel-reference-tabular-centroid/0.4"], "transfer-admitted")
        with tempfile.TemporaryDirectory() as directory:
            result = run_reference_portfolio_study(Path(directory))
            self.assertTrue(result["report"]["ledger"]["valid"])
            self.assertGreater(result["report"]["ledger"]["record_count"], 30)

    def test_provider_study_schema_is_present_and_authority_is_fixed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schemas" / "mnel-provider-study.schema.json").read_text())
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIn("report", schema["$defs"])
        self.assertEqual(schema["$defs"]["report"]["properties"]["authority"]["$ref"], "#/$defs/diagnostic")
        calibration_schema = json.loads((root / "schemas" / "mnel-calibration.schema.json").read_text())
        self.assertEqual(calibration_schema["properties"]["authority"]["const"], "diagnostic-only")


if __name__ == "__main__":
    unittest.main()
