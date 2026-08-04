import json
import sys
import tempfile
import unittest
from pathlib import Path

from mnel.core import (
    Attribution,
    AuthorityBoundary,
    DistillationError,
    EvidenceLedger,
    HardGateEvaluator,
    Intervention,
    LedgerIntegrityError,
    Maturity,
    Observation,
    OutcomeClass,
    RecursionGovernor,
    TransferStatus,
    VerifiedExperienceDistiller,
    Verdict,
    canonical_digest,
    canonical_json,
    run_reference_study,
)
from mnel.integrations import JSONCommandAdapter, RavelKnowledgeProposal


class CanonicalTests(unittest.TestCase):
    def test_object_order_does_not_change_identity(self) -> None:
        self.assertEqual(canonical_json({"b": 2, "a": 1}), canonical_json({"a": 1, "b": 2}))
        self.assertEqual(canonical_digest({"b": 2, "a": 1}), canonical_digest({"a": 1, "b": 2}))

    def test_nan_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json({"value": float("nan")})


class LedgerTests(unittest.TestCase):
    def test_append_verify_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            ledger = EvidenceLedger(path)
            first = ledger.append("alpha", {"value": 1}, actor="tester")
            second = ledger.append("beta", {"value": 2}, actor="tester")
            self.assertTrue(ledger.verify().valid)
            self.assertEqual(second["previous_digest"], first["record_digest"])
            records = path.read_text(encoding="utf-8").splitlines()
            record = json.loads(records[0])
            record["payload"]["value"] = 99
            records[0] = json.dumps(record)
            path.write_text("\n".join(records) + "\n", encoding="utf-8")
            self.assertFalse(ledger.verify().valid)
            with self.assertRaises(LedgerIntegrityError):
                ledger.append("gamma", {}, actor="tester")


class AuthorityTests(unittest.TestCase):
    def test_authority_mutation_and_in_place_edit_are_rejected(self) -> None:
        decision = RecursionGovernor().validate_proposed_mutation(
            {"evaluator_identity": "investigator-controlled", "routing": "new"}
        )
        self.assertEqual(decision.verdict, Verdict.FAIL)
        with self.assertRaises(ValueError):
            Intervention("i", "same", "same", "edit", ("routing",), "same")
        with self.assertRaises(ValueError):
            AuthorityBoundary("e", "g", "p", "r", promotion_authorized=True)
        with self.assertRaises(ValueError):
            RavelKnowledgeProposal("p", "principle", "parent", ("r",), {}, (), "parent", True)


class EvaluationTests(unittest.TestCase):
    def test_failed_gate_dominates_and_missing_metric_is_unknown(self) -> None:
        observation = Observation(
            "o1",
            "e1",
            OutcomeClass.SUCCESS,
            {"accuracy": 0.9, "retention": 0.7},
            1,
            0.1,
            "provider",
        )
        evaluator = HardGateEvaluator()
        failed = evaluator.evaluate(
            experiment_id="e1",
            observation=observation,
            gates=(
                {"name": "accuracy", "metric": "accuracy", "operator": "ge", "threshold": 0.8},
                {"name": "retention", "metric": "retention", "operator": "ge", "threshold": 0.8},
            ),
        )
        self.assertEqual(failed.verdict, Verdict.FAIL)
        unknown = evaluator.evaluate(
            experiment_id="e1",
            observation=observation,
            gates=({"name": "planning", "metric": "planning", "operator": "ge", "threshold": 1},),
        )
        self.assertEqual(unknown.verdict, Verdict.UNKNOWN)


class DistillationTests(unittest.TestCase):
    def test_transfer_gate_and_failure_modes(self) -> None:
        attribution = Attribution(
            "a1",
            "e1",
            "i1",
            "v1",
            "supported",
            ("immediate",),
            ("o1",),
            (),
            ("record-o1", "record-v1"),
        )
        distiller = VerifiedExperienceDistiller()
        principle = distiller.propose_principle(
            principle_id="p1",
            statement="bounded claim",
            scope={"fixture": "one"},
            attributions=(attribution,),
            counterexample_episode_ids=(),
            falsifier="fails elsewhere",
            transfer_status=TransferStatus.UNTESTED,
            requested_maturity=Maturity.SUPPORTED,
        )
        self.assertEqual(principle.maturity, Maturity.PROVISIONAL)
        self.assertEqual(principle.source_record_ids, ("record-o1", "record-v1"))
        with self.assertRaises(DistillationError):
            distiller.propose_strategy(
                strategy_id="s1",
                trigger_conditions=("overlap",),
                intervention_class="routing",
                preconditions=("support exists",),
                known_failure_modes=(),
                applicability_scope={"fixture": "one"},
                principles=(principle,),
                transfer_status=TransferStatus.UNTESTED,
            )


class IntegrationTests(unittest.TestCase):
    def test_json_command_adapter(self) -> None:
        code = "import json,sys; data=json.load(sys.stdin); print(json.dumps({'seen':data['value']}))"
        result = JSONCommandAdapter((sys.executable, "-c", code)).run({"value": 7})
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.parsed, {"seen": 7})


class RepositoryTests(unittest.TestCase):
    def test_reference_study_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_reference_study(directory)
            self.assertEqual(result["evaluation_verdict"], "PASS")
            self.assertEqual(result["principle_maturity"], "provisional")
            self.assertEqual(result["transfer_status"], "untested")
            self.assertTrue(result["ledger"]["valid"])
            self.assertEqual(result["ledger"]["record_count"], 12)
        schema = Path(__file__).resolve().parents[1] / "schemas" / "mnel-records.schema.json"
        value = json.loads(schema.read_text(encoding="utf-8"))
        self.assertIn("$defs", value)
        self.assertGreaterEqual(len(value["$defs"]), 6)


if __name__ == "__main__":
    unittest.main()
