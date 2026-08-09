import json
import tempfile
import unittest
from pathlib import Path

from mnel.family_integration import (
    FamilyIdentityBinding,
    FamilyIntegrationError,
    FamilyEvidenceAdapter,
    ReplayGuard,
    RavelProposalAdapter,
    CommonsInterchangeAdapter,
    run_reference_family_integration,
    validate_compatibility_fixture,
)
from mnel.forge_provider import handle_request, process_line
from mnel.forge_provider import ForgeProviderError
from mnel.reference_provider import TransitionFrequencyModel


class ForgeProviderProtocolTests(unittest.TestCase):
    def test_capabilities_and_analysis_are_protocol_compatible(self):
        capabilities = handle_request({"protocol_version": "0.1", "type": "capabilities", "request_id": "r"})
        self.assertEqual(capabilities["type"], "capabilities")
        self.assertEqual(capabilities["provider"]["identity"], "mnel-family-provider-protocol-v1")
        self.assertIn("distributed_workload_inspection", capabilities["analyses"])
        response = handle_request(
            {
                "protocol_version": "0.1",
                "type": "analysis_request",
                "request_id": "r",
                "analysis": "provider_study_summary",
                "component": {"candidate_identity": "forge-tree-sha256-v1:opaque", "identities": {"study": "sha256:" + "1" * 64}},
                "limits": {"output_bytes": 4096},
            }
        )
        self.assertEqual(response["status"], "UNKNOWN")
        self.assertEqual(response["extensions"]["mnel"]["authority"], "diagnostic-only")
        distributed = handle_request({"protocol_version": "0.1", "type": "analysis_request", "request_id": "distributed", "analysis": "reconciliation_summary", "component": {"identities": {"study": "sha256:" + "2" * 64}}, "limits": {"output_bytes": 4096}})
        self.assertEqual(distributed["status"], "UNKNOWN")

    def test_protocol_rejects_framing_authority_and_bad_identity(self):
        with self.assertRaises(ForgeProviderError):
            handle_request({"protocol_version": "0.1", "type": "analysis_request", "request_id": "x", "analysis": "provider_study_summary", "promotion": True})
        with self.assertRaises(ForgeProviderError):
            handle_request({"protocol_version": "0.1", "type": "analysis_request", "request_id": "x", "analysis": "provider_study_summary", "component": {"identities": {"snapshot": "bad"}}})
        response = json.loads(process_line(b"{}\n{}\n"))
        self.assertEqual(response["type"], "error")
        self.assertIn("extensions", response)

    def test_unsupported_analysis_is_unknown(self):
        response = handle_request({"protocol_version": "0.1", "type": "analysis_request", "request_id": "x", "analysis": "not-supported"})
        self.assertEqual(response["status"], "UNKNOWN")


class FamilyCompatibilityTests(unittest.TestCase):
    def test_pinned_fixture_and_drift_fail_closed(self):
        path = Path(__file__).parents[1] / "compat" / "mncs-family-compatibility-0.1.json"
        result = validate_compatibility_fixture(path)
        self.assertEqual(result["status"], "PASS")
        value = json.loads(path.read_text(encoding="utf-8"))
        value["projects"][0]["public_contract"]["version"] = "Provider Protocol 0.2"
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "compat.json"
            mutated.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(FamilyIntegrationError):
                validate_compatibility_fixture(mutated)

    def test_replay_guard_rejects_conflict(self):
        guard = ReplayGuard()
        record = {"record_id": "sha256:" + "a" * 64, "value": 1}
        self.assertEqual(guard.accept(record), record["record_id"])
        self.assertEqual(guard.accept(dict(record)), record["record_id"])
        with self.assertRaises(FamilyIntegrationError):
            guard.accept({"record_id": record["record_id"], "value": 2})


class FamilyReferenceStudyTests(unittest.TestCase):
    def test_checked_in_native_artifact_fixture_reloads_in_python(self):
        path = Path(__file__).parents[1] / "crates" / "mnel-provider-classical" / "tests" / "fixtures" / "transition-frequency-artifact.json"
        model = TransitionFrequencyModel.load(path.read_bytes())
        self.assertEqual(model.artifact_identity, "sha256:3ddf97d1780aabf06675fd32e9121d065125d91b6481daff51ec7f97264b9a28")

    def test_reference_study_runs_and_preserves_receipt_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_reference_family_integration(directory)
            self.assertEqual(result["fabric"]["availability"], "available")
            self.assertEqual(result["fabric"]["execution_record"]["outcome"], "PASS")
            self.assertTrue(result["fabric"]["normalized"]["normalized_identity"].startswith("sha256:"))
            self.assertEqual(result["fabric"]["replication"]["scope"], "local-in-process-replication")
            self.assertEqual(result["fabric"]["negative_cases"]["capability_mismatch"]["reason"], "CAPABILITY_UNAVAILABLE")
            self.assertEqual(result["fabric"]["negative_cases"]["wrong_manifest"]["reason"], "INTEGRITY_FAILURE")
            self.assertEqual(result["fabric"]["negative_cases"]["unsupported_plan_version"]["reason"], "PLAN_INVALID")
            self.assertEqual(result["fabric"]["negative_cases"]["corrupt_record_identity"]["outcome"], "FAIL")
            self.assertEqual(result["commons"]["kind"], "Observation")
            self.assertIn(result["commons_bundle"]["status"], {"available", "unavailable", "UNKNOWN"})
            if result["commons_bundle"]["status"] == "available":
                self.assertTrue(result["commons_bundle"]["verify"]["valid"])
            self.assertEqual(result["ravel"]["authority"], "proposal-only")
            self.assertTrue(result["ledger"]["valid"])

    def test_external_identity_mismatch_is_rejected(self):
        binding = FamilyIdentityBinding("sha256:" + "1" * 64, "sha256:" + "2" * 64, "sha256:" + "3" * 64)
        with self.assertRaises(FamilyIntegrationError):
            FamilyEvidenceAdapter.normalize_fabric_execution(
                {"schema_version": "mncs-fabric.execution-record.v0.1", "record_id": "sha256:" + "4" * 64, "candidate_identity": "sha256:" + "5" * 64}, binding
            )

    def test_provider_artifact_mismatch_is_rejected(self):
        binding = FamilyIdentityBinding("sha256:" + "1" * 64, "sha256:" + "2" * 64, "sha256:" + "3" * 64)
        record = {"schema_version": "mncs-fabric.execution-record.v0.1", "record_id": "sha256:" + "4" * 64, "candidate_identity": binding.mnel_study_identity}
        with self.assertRaises(FamilyIntegrationError):
            FamilyEvidenceAdapter.normalize_fabric_execution(record, binding, observed_provider_artifact_identity="sha256:" + "5" * 64)

    def test_ravel_promotion_injection_is_rejected(self):
        binding = FamilyIdentityBinding("sha256:" + "1" * 64, "sha256:" + "2" * 64, "sha256:" + "3" * 64)
        with self.assertRaises(FamilyIntegrationError):
            RavelProposalAdapter.proposal(binding, {"schema": "ravel-development-record/0.6-preregistration", "candidate": {"candidate_id": "c"}, "authority": {"promotion_authorized": True}})

    def test_commons_adapter_is_inert_observation(self):
        value = CommonsInterchangeAdapter.observation({"external": {"record_identity": "sha256:" + "a" * 64}, "normalized_identity": "sha256:" + "b" * 64})
        self.assertEqual(value["kind"], "Observation")
        self.assertEqual(value["details"]["outcome"], "UNKNOWN")
        self.assertNotIn("promotion", value)
