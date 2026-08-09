import tempfile
import unittest
from pathlib import Path

from mnel.core import Visibility, canonical_digest
from mnel.distillation import StudyDataAccess, VisibilityViolation, make_study_record
from mnel.fabric_execution import (
    DistributedExecutionError,
    DistributedWorkload,
    ExecutionDisposition,
    ExpertPlacementRequest,
    FabricNetworkConfig,
    FabricExecutionObservation,
    WorkloadClass,
    WorkloadGraph,
    WorkloadNode,
    aggregate_transition_partials,
    aggregate_centroid_partials,
    centroid_training_partial,
    make_training_shards,
    make_study_matrix,
    run_reference_fabric_study,
    transition_training_partial,
)
from mnel.snapshots import SnapshotStore, tabular_snapshot, transition_snapshot


def identity(label: str) -> str:
    return canonical_digest({"test": label})


class FabricExecutionTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = SnapshotStore()
        common = {
            "producer_identity": "fabric-test-producer",
            "source_identity": identity("source"),
            "dependency_identity": identity("dependency"),
            "feature_extractor_identity": identity("features"),
        }
        self.first = transition_snapshot(b"a", b"b", **common)
        self.second = transition_snapshot(b"b", b"c", **common)
        self.snapshots.register(self.first)
        self.snapshots.register(self.second)
        self.records = (
            make_study_record(
                "transition-episode",
                {"snapshot_identity": self.first.snapshot_identity},
                record_identity=identity("record-a"),
            ),
            make_study_record(
                "transition-episode",
                {"snapshot_identity": self.second.snapshot_identity},
                record_identity=identity("record-b"),
            ),
        )
        self.access = StudyDataAccess.development(self.records)

    def test_workload_graph_is_deterministic_and_rejects_cycle(self):
        study = identity("study")
        first = DistributedWorkload(
            study,
            identity("experiment-a"),
            WorkloadClass.STUDY_CASE,
            snapshot_identities=(self.first.snapshot_identity,),
        )
        second = DistributedWorkload(
            study,
            identity("experiment-b"),
            WorkloadClass.STUDY_CASE,
            snapshot_identities=(self.second.snapshot_identity,),
        )
        graph = WorkloadGraph((WorkloadNode(first), WorkloadNode(second)))
        self.assertEqual(
            set(graph.topological_order()),
            {graph.nodes[0].content_identity, graph.nodes[1].content_identity},
        )
        with self.assertRaises(DistributedExecutionError):
            WorkloadGraph(
                (
                    WorkloadNode(first, (WorkloadNode(second).content_identity,)),
                    WorkloadNode(second, (WorkloadNode(first).content_identity,)),
                )
            )

    def test_hidden_training_is_rejected(self):
        hidden = make_study_record(
            "transition-episode",
            {"snapshot_identity": self.first.snapshot_identity},
            visibility=Visibility.TRANSFER_HIDDEN,
        )
        with self.assertRaises(VisibilityViolation):
            make_training_shards(
                StudyDataAccess.transfer_evaluator((hidden,)),
                provider_family="transition-frequency",
                feature_extractor_identity=identity("features"),
                training_code_identity="training",
                shard_count=1,
                record_type="transition-episode",
            )

    def test_sharded_transition_training_matches_full_counts_and_rejects_duplicate(self):
        partitions, shards = make_training_shards(
            self.access,
            provider_family="transition-frequency",
            feature_extractor_identity=identity("features"),
            training_code_identity="training",
            shard_count=2,
            record_type="transition-episode",
        )
        partials = tuple(
            transition_training_partial(
                StudyDataAccess.development(part),
                self.snapshots,
                shard,
                record_type="transition-episode",
            )
            for part, shard in zip(partitions, shards)
        )
        model = aggregate_transition_partials(
            partials,
            dataset_identity=self.access.dataset_identity,
            feature_extractor_identity=identity("features"),
            training_code_identity="training",
            training_record_ids=tuple(record.identity for record in self.records),
            calibration_identity=identity("calibration"),
        )
        self.assertEqual(model.total_count, 2)
        with self.assertRaises(DistributedExecutionError):
            aggregate_transition_partials(
                (partials[0], partials[0]),
                dataset_identity=self.access.dataset_identity,
                feature_extractor_identity=identity("features"),
                training_code_identity="training",
                training_record_ids=tuple(record.identity for record in self.records),
                calibration_identity=identity("calibration"),
            )

    def test_centroid_sufficient_statistics_match_single_host(self):
        common = {
            "producer_identity": "fabric-test-producer",
            "source_identity": identity("tab-source"),
            "dependency_identity": identity("tab-dependency"),
            "feature_extractor_identity": identity("features"),
        }
        first = tabular_snapshot(((1.0, 2.0),), **common)
        second = tabular_snapshot(((3.0, 4.0),), **common)
        store = SnapshotStore()
        store.register(first)
        store.register(second)
        records = (
            make_study_record(
                "tabular-episode",
                {"snapshot_identity": first.snapshot_identity},
                record_identity=identity("tab-a"),
            ),
            make_study_record(
                "tabular-episode",
                {"snapshot_identity": second.snapshot_identity},
                record_identity=identity("tab-b"),
            ),
        )
        access = StudyDataAccess.development(records)
        partitions, shards = make_training_shards(
            access,
            provider_family="nearest-centroid",
            feature_extractor_identity=identity("features"),
            training_code_identity="centroid-training",
            shard_count=2,
            record_type="tabular-episode",
        )
        partials = tuple(
            centroid_training_partial(
                StudyDataAccess.development(part), store, shard, record_type="tabular-episode"
            )
            for part, shard in zip(partitions, shards)
        )
        model = aggregate_centroid_partials(
            partials,
            dataset_identity=access.dataset_identity,
            feature_extractor_identity=identity("features"),
            training_code_identity="centroid-training",
            training_record_ids=tuple(record.identity for record in records),
            calibration=(((2.0, 3.0), 1),),
            calibration_identity=identity("calibration"),
            calibration_dataset_identity=identity("calibration-data"),
        )
        self.assertEqual(model.centroid, (2.0, 3.0))
        self.assertEqual(model.scales, (1.0, 1.0))

    def test_reference_study_uses_public_local_fabric_and_replicates(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_reference_fabric_study(directory)
            report = result["report"]
            self.assertTrue(report["backend_identity"].startswith("mnel-fabric-"))
            self.assertEqual(report["observation_count"], 4)
            self.assertEqual(report["replication"]["observed"], 4)
            self.assertTrue(report["transition_sharded_training"]["single_host_equivalent"])
            self.assertEqual(report["authority"], "diagnostic-only")
            self.assertTrue(Path(directory, "fabric-evidence.jsonl").is_file())

    def test_external_observation_rejects_authority_injection(self):
        with self.assertRaises(DistributedExecutionError):
            FabricExecutionObservation(
                identity("workload"),
                "worker-a",
                identity("artifact"),
                ExecutionDisposition.COMPLETED,
                {"verdict": "PASS"},
            )

    def test_study_matrix_cells_are_identity_bound(self):
        workload = DistributedWorkload(
            identity("study"),
            identity("experiment"),
            WorkloadClass.STUDY_CASE,
            snapshot_identities=(self.first.snapshot_identity,),
        )
        matrix = make_study_matrix((workload,), controls=("provider", "random"), repetitions=2)
        self.assertEqual(len(matrix.cells), 4)
        self.assertEqual(len({cell.content_identity for cell in matrix.cells}), 4)

    def test_expert_placement_is_separate_from_worker_identity(self):
        request = ExpertPlacementRequest(
            "expert",
            identity("artifact"),
            identity("model"),
            identity("calibration"),
            "transition-frequency",
            ("transition",),
            128,
            execution_device="cpu",
            offload_policy="none",
        )
        workload = DistributedWorkload(
            identity("study"),
            identity("experiment"),
            WorkloadClass.EXPERT_INFERENCE,
            provider_id="expert",
            expert_placement=request,
            provider_artifact_identity=request.provider_artifact_identity,
            model_identity=request.model_identity,
            calibration_identity=request.calibration_identity,
            snapshot_identities=(self.first.snapshot_identity,),
        )
        self.assertEqual(workload.expert_placement.placement_identity, request.placement_identity)

    def test_network_configuration_fails_closed_without_trust_material(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fabric.toml"
            path.write_text(
                "controller_id='c'\n[[workers]]\nworker_id='w'\nhost='127.0.0.1'\nport=443\ncapabilities=['python']\nca_file='ca.pem'\nclient_cert='client.pem'\nclient_key='client.key'\ntrust_store='trust.jsonl'\n",
                encoding="utf-8",
            )
            with self.assertRaises(DistributedExecutionError):
                FabricNetworkConfig.load(path)


if __name__ == "__main__":
    unittest.main()
