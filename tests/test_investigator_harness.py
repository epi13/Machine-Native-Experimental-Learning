import unittest

from mnel.investigator_harness import (
    CandidateTransaction,
    ContextPackingError,
    MorningReport,
    QuarantineEntry,
    QuarantineQueue,
    RuntimeIdentityEnvelope,
    InvestigatorWorkspace,
    WorkspaceAccess,
    pack_eligible_context,
)


class InvestigatorHarnessTests(unittest.TestCase):
    def test_context_packing_is_stable_and_bounded(self) -> None:
        context = pack_eligible_context(
            [
                {"record_id": "b", "visibility": "development-visible", "value": 2},
                {"record_id": "a", "visibility": "selection-observed-not-repairable", "value": 1},
            ],
            max_records=2,
            max_bytes=1024,
        )
        self.assertEqual(context.record_ids, ("a", "b"))
        self.assertEqual(context.to_dict()["authority"], "proposal-only")
        self.assertEqual(
            context.snapshot_identity,
            pack_eligible_context(
                [{"record_id": "b", "visibility": "development-visible", "value": 2},
                 {"record_id": "a", "visibility": "selection-observed-not-repairable", "value": 1}],
                max_records=2,
                max_bytes=1024,
            ).snapshot_identity,
        )

    def test_hidden_context_and_oversized_context_are_rejected(self) -> None:
        with self.assertRaises(ContextPackingError):
            pack_eligible_context([{"record_id": "hidden", "visibility": "transfer-hidden"}])
        with self.assertRaises(ContextPackingError):
            pack_eligible_context([{"record_id": "large", "text": "x" * 100}], max_bytes=8)

    def test_transaction_identity_and_runtime_lineage_are_explicit(self) -> None:
        envelope = RuntimeIdentityEnvelope("model", "quant", "runtime", "prompt", "tools")
        transaction = CandidateTransaction.create(
            parent_candidate_id="candidate-parent",
            context_snapshot_identity="sha256:context",
            workspace="build/proposal",
        )
        self.assertEqual(transaction.access, WorkspaceAccess.PROPOSAL)
        self.assertTrue(transaction.proposal_only)
        self.assertEqual(envelope.to_dict()["runtime_identity"], "runtime")
        self.assertEqual(transaction.transaction_identity, CandidateTransaction.create(
            parent_candidate_id="candidate-parent",
            context_snapshot_identity="sha256:context",
            workspace="build/proposal",
        ).transaction_identity)
        with self.assertRaises(PermissionError):
            InvestigatorWorkspace.read_only("build/read-only").assert_write_allowed()
        InvestigatorWorkspace.proposal("build/proposal").assert_write_allowed()

    def test_quarantine_and_report_remain_observable(self) -> None:
        queue = QuarantineQueue()
        queue.enqueue(QuarantineEntry("provider-a", "runtime failure", "sha256:evidence"))
        self.assertEqual(queue.list()[0].provider_id, "provider-a")
        report = MorningReport("sha256:context", 2, 3, 1, 1)
        self.assertEqual(report.to_dict()["authority"], "proposal-only")
        self.assertNotIn("verdict", report.to_dict())


if __name__ == "__main__":
    unittest.main()
