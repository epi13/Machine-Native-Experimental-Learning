import tempfile
import unittest
from dataclasses import replace
import json
from pathlib import Path

from mnel.forge_lifecycle import (
    ForgeLifecycleError,
    LearnedDiagnosticEvent,
    MutationPolicy,
    MutationRegistry,
    Precondition,
    ProbeExecutionStatus,
    ProbeRequest,
    ReferenceForgeRuntime,
    VerifierDeclaration,
    VerifierHealthStore,
    VerifierRegistry,
    VerifierState,
    build_coverage,
    compare_witnesses,
    discover_question_candidates,
    reference_verifier_registry,
    run_reference_forge_study,
)
from mnel.snapshots import SnapshotStore, transition_snapshot


class _BrokenVerifier:
    def run(self, view, parameters, operation_limit):
        raise ForgeLifecycleError("fixture verifier failure")


class _MalformedVerifier:
    def run(self, view, parameters, operation_limit):
        return ["not", "an", "object"]


class ForgeLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identities = {
            "producer_identity": "producer:v1",
            "source_identity": "source:v1",
            "dependency_identity": "dependency:v1",
            "feature_extractor_identity": "extractor:v1",
        }
        self.snapshot = transition_snapshot(b"cold", b"warm", **self.identities)
        self.store = SnapshotStore()
        self.store.register(self.snapshot)
        self.registry = reference_verifier_registry()
        self.runtime = ReferenceForgeRuntime(self.store, self.registry)

    def _request(self, verifier_id="transition-change", **overrides):
        values = {
            "probe_id": "probe-1",
            "question": "did this transition change state?",
            "subject_identities": {"subject": "source:v1"},
            "verifier_id": verifier_id,
            "snapshot_identities": (self.snapshot.snapshot_identity,),
            "expected_witness_type": "transition-witness",
            "preconditions": (),
            "resource_budget": {"operation_limit": 100, "wall_time_ms": 1000, "output_bytes": 4096},
            "mutation_policy": MutationPolicy.REGISTERED_ONLY,
            "runtime_identity": {"runtime": "fixture"},
            "lineage": {"request": "request:v1"},
            "parameters": {},
        }
        values.update(overrides)
        return ProbeRequest(**values)

    def _contains_forbidden_key(self, value) -> bool:
        if isinstance(value, dict):
            return any(
                key in {"verdict", "conformance", "promotion_authorized"}
                or self._contains_forbidden_key(child)
                for key, child in value.items()
            )
        if isinstance(value, list):
            return any(self._contains_forbidden_key(child) for child in value)
        return False

    def test_registry_rejects_collisions_and_matches_identity_bound_snapshot(self) -> None:
        declaration = VerifierDeclaration(
            "fixture",
            "1",
            "fixture-implementation",
            ("transition",),
            (1,),
            (Precondition("required_snapshot_type", "transition"),),
            "fixture-witness",
            {"operation_limit": 10, "wall_time_ms": 100, "output_bytes": 1000},
            True,
            False,
        )
        registry = VerifierRegistry()
        registry.register(declaration, _BrokenVerifier())
        with self.assertRaises(ForgeLifecycleError):
            registry.register(declaration, _BrokenVerifier())
        self.assertEqual(registry.match(self.snapshot)[0].verifier_id, "fixture")

        changed = transition_snapshot(b"cold", b"warm", **{**self.identities, "dependency_identity": "dependency:v2"})
        self.assertNotEqual(changed.snapshot_identity, self.snapshot.snapshot_identity)

        loaded = VerifierRegistry()
        loaded.load((declaration.to_dict(),))
        self.assertEqual(loaded.state("fixture"), VerifierState.DISABLED)

    def test_failed_precondition_is_ineligible_and_has_no_verdict(self) -> None:
        request = self._request(preconditions=(Precondition("required_snapshot_type", "tabular"),))
        witness = self.runtime.execute(request)
        self.assertEqual(witness.execution_status, ProbeExecutionStatus.INELIGIBLE)
        self.assertNotIn("verdict", witness.to_dict())
        self.assertEqual(witness.precondition_report.status, "failed")

    def test_probe_parameters_cannot_expand_nested_authority(self) -> None:
        with self.assertRaises(ForgeLifecycleError):
            self._request(parameters={"nested": {"authority": "evaluator"}})

    def test_reference_witness_is_bounded_and_identity_is_stable(self) -> None:
        first = self.runtime.execute(self._request())
        second = self.runtime.execute(self._request())
        self.assertEqual(first.execution_status, ProbeExecutionStatus.COMPLETED)
        self.assertEqual(first.witness_identity, second.witness_identity)
        self.assertLessEqual(first.resource_usage["output_bytes"], 4096)
        self.assertEqual(first.authority, "diagnostic-only")
        self.assertEqual(first.semantics, "not-a-verdict")

    def test_mutation_preserves_original_and_changes_identity(self) -> None:
        mutations = MutationRegistry()
        record = mutations.apply("transition.swap", self.snapshot, {}, self.store)
        self.assertNotEqual(record.original_snapshot_identity, record.resulting_snapshot_identity)
        self.assertEqual(self.store.get(record.original_snapshot_identity).payload, self.snapshot.payload)
        self.assertEqual(record.authority, "diagnostic-only")
        with self.assertRaises(ForgeLifecycleError):
            mutations.apply("arbitrary.python", self.snapshot, {}, self.store)

    def test_comparison_preserves_agreement_and_disagreement(self) -> None:
        first = self.runtime.execute(self._request("transition-change"))
        second = self.runtime.execute(
            self._request("transition-change-independent", probe_id="probe-2")
        )
        comparison = compare_witnesses((first, second), {"subject": "source:v1"})
        self.assertEqual(comparison.comparison_status, "agreement")
        contrary = replace(second, diagnostic_output={"condition_observed": False})
        disagreement = compare_witnesses((first, contrary), {"subject": "source:v1"})
        self.assertEqual(disagreement.comparison_status, "disagreement")
        self.assertNotIn("verdict", disagreement.to_dict())

    def test_repeated_errors_quarantine_verifier(self) -> None:
        declaration = VerifierDeclaration(
            "broken",
            "1",
            "broken-implementation",
            ("transition",),
            (1,),
            (),
            "broken-witness",
            {"operation_limit": 10, "wall_time_ms": 100, "output_bytes": 1000},
            True,
            False,
        )
        registry = VerifierRegistry()
        registry.register(declaration, _BrokenVerifier())
        runtime = ReferenceForgeRuntime(self.store, registry, VerifierHealthStore(2))
        request = self._request("broken", expected_witness_type="broken-witness")
        self.assertEqual(runtime.execute(request).execution_status, ProbeExecutionStatus.ERROR)
        self.assertEqual(runtime.execute(request).execution_status, ProbeExecutionStatus.ERROR)
        self.assertEqual(registry.state("broken"), VerifierState.QUARANTINED)
        self.assertEqual(runtime.execute(request).execution_status, ProbeExecutionStatus.QUARANTINED)

    def test_malformed_verifier_output_fails_closed_and_is_counted(self) -> None:
        declaration = VerifierDeclaration(
            "malformed",
            "1",
            "malformed-implementation",
            ("transition",),
            (1,),
            (),
            "malformed-witness",
            {"operation_limit": 10, "wall_time_ms": 100, "output_bytes": 1000},
            True,
            False,
        )
        registry = VerifierRegistry()
        registry.register(declaration, _MalformedVerifier())
        health = VerifierHealthStore()
        witness = ReferenceForgeRuntime(self.store, registry, health).execute(
            self._request("malformed", expected_witness_type="malformed-witness")
        )
        self.assertEqual(witness.execution_status, ProbeExecutionStatus.ERROR)
        self.assertEqual(health.to_dict("malformed")["malformed_outputs"], 1)

    def test_learned_observation_is_not_a_verifier_witness(self) -> None:
        event = LearnedDiagnosticEvent.from_observation(
            {
                "provider_id": "provider:v1",
                "observation_identity": "observation:v1",
                "snapshot_ids": [self.snapshot.snapshot_identity],
                "declaration_identity": "declaration:v1",
                "score": 0.75,
            }
        )
        value = event.to_dict()
        self.assertEqual(value["record_type"], "learned-provider-observation")
        self.assertNotIn("verifier_id", value)
        with self.assertRaises(ForgeLifecycleError):
            LearnedDiagnosticEvent.from_observation(
                {
                    "provider_id": "provider:v1",
                    "observation_identity": "observation:v1",
                    "snapshot_ids": [self.snapshot.snapshot_identity],
                    "declaration_identity": "declaration:v1",
                    "verdict": "PASS",
                }
            )

    def test_reference_study_writes_a_valid_append_only_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_reference_forge_study(Path(directory))
            self.assertEqual(result["comparison"]["comparison_status"], "agreement")
            self.assertTrue(result["question_candidates"])
            self.assertTrue(Path(directory, "forge-evidence.jsonl").is_file())
            self.assertTrue(all(not self._contains_forbidden_key(record) for record in result["records"]))

    def test_lifecycle_schema_is_machine_readable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schemas" / "mnel-forge-lifecycle.schema.json").read_text())
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIn("witness", schema["$defs"])
        self.assertIn("comparison", schema["$defs"])

    def test_coverage_surfaces_single_source_questions(self) -> None:
        witness = self.runtime.execute(self._request())
        coverage = build_coverage(self.registry, self.store, (witness,))
        self.assertIn("transition", coverage.exercised_snapshot_types)
        self.assertIn(witness.question_identity, coverage.single_source_question_identities)
        candidates = discover_question_candidates("source:v1", self.store, coverage)
        self.assertTrue(candidates)

    def test_skeptic_discovery_is_bounded_deduplicated_and_proposal_only(self) -> None:
        failed_request = self._request(
            probe_id="failed-probe-1",
            preconditions=(Precondition("required_snapshot_type", "tabular"),),
        )
        failed = self.runtime.execute(failed_request)
        failed_again = self.runtime.execute(
            self._request(
                probe_id="failed-probe-2",
                preconditions=(Precondition("required_snapshot_type", "tabular"),),
            )
        )
        mutation = MutationRegistry().apply("transition.swap", self.snapshot, {}, self.store)
        learned = LearnedDiagnosticEvent.from_observation(
            {
                "provider_id": "provider:v1",
                "observation_identity": "observation:disagree",
                "snapshot_ids": [self.snapshot.snapshot_identity],
                "declaration_identity": "declaration:v1",
                "condition_observed": False,
            }
        )
        coverage = build_coverage(self.registry, self.store, (failed, failed_again))
        candidates = discover_question_candidates(
            "source:v1",
            self.store,
            coverage,
            witnesses=(failed, failed_again),
            mutations=(mutation,),
            learned_observations=(learned,),
            registry=self.registry,
            max_candidates=12,
        )
        self.assertLessEqual(len(candidates), 12)
        self.assertEqual(len({item.candidate_identity for item in candidates}), len(candidates))
        self.assertIn("repeated-unknown", {item.candidate_kind for item in candidates})
        self.assertTrue(all(item.authority == "proposal-only" for item in candidates))
        self.assertTrue(all("verdict" not in item.to_dict() for item in candidates))
        with self.assertRaises(ForgeLifecycleError):
            discover_question_candidates(
                "source:v1",
                self.store,
                coverage,
                witnesses=(failed,),
                visible_lineage=frozenset({"development-only-id"}),
            )


if __name__ == "__main__":
    unittest.main()
