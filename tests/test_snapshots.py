import unittest

from mnel.snapshots import (
    SnapshotError,
    SnapshotStore,
    composite_snapshot,
    decode_composite,
    decode_graph,
    decode_pair,
    decode_snapshot,
    decode_tabular,
    decode_trace,
    graph_snapshot,
    pair_snapshot,
    tabular_snapshot,
    trace_snapshot,
    transition_snapshot,
)


class SnapshotTests(unittest.TestCase):
    def _kwargs(self) -> dict[str, str]:
        return {
            "producer_identity": "producer:v1",
            "source_identity": "source:v1",
            "dependency_identity": "dependency:v1",
            "feature_extractor_identity": "extractor:v1",
        }

    def test_transition_and_pair_snapshots_are_immutable_and_diagnostic_only(self) -> None:
        transition = transition_snapshot(b"a", b"b", **self._kwargs())
        pair = pair_snapshot(b"left", b"right", **self._kwargs())
        self.assertEqual(transition.snapshot_type, "transition")
        self.assertNotEqual(transition.snapshot_identity, pair.snapshot_identity)
        self.assertEqual(transition.to_dict()["authority"], "diagnostic-only")
        self.assertEqual(transition.to_dict()["semantics"], "not-a-verdict")
        with self.assertRaises(AttributeError):
            transition.payload = b"changed"  # type: ignore[misc]

    def test_material_dependency_changes_invalidate_identity(self) -> None:
        first = transition_snapshot(b"a", b"b", **self._kwargs())
        changed = transition_snapshot(
            b"a", b"b", **{**self._kwargs(), "dependency_identity": "dependency:v2"}
        )
        self.assertNotEqual(first.snapshot_identity, changed.snapshot_identity)
        self.assertNotEqual(first.payload_identity, "")

    def test_tabular_payload_is_binary_bounded_and_rejects_nonfinite_values(self) -> None:
        snapshot = tabular_snapshot(((1.0, 2.0), (3.0, 4.0)), **self._kwargs())
        self.assertEqual(snapshot.payload[:4], b"MNET")
        self.assertGreater(len(snapshot.payload), 4)
        self.assertEqual(decode_tabular(snapshot.payload).rows[1], (3.0, 4.0))
        with self.assertRaises(SnapshotError):
            tabular_snapshot(((float("nan"),),), **self._kwargs())

    def test_all_views_round_trip_through_shared_store(self) -> None:
        transition = transition_snapshot(b"a", b"b", **self._kwargs())
        pair = pair_snapshot(b"left", b"right", **self._kwargs())
        trace = trace_snapshot((("start", b"1"), ("stop", b"2")), **self._kwargs())
        graph = graph_snapshot(
            ((2, "target"), (1, "source")),
            ((1, 2, "calls"),),
            **self._kwargs(),
        )
        composite = composite_snapshot((transition, graph), **self._kwargs())
        self.assertEqual(decode_snapshot(transition).next_state, b"b")
        self.assertEqual(decode_pair(pair.payload).right, b"right")
        self.assertEqual(len(decode_trace(trace.payload).events), 2)
        self.assertEqual(decode_graph(graph.payload).edges[0].edge_type, "calls")
        self.assertEqual(decode_composite(composite.payload).component_snapshot_identities, (transition.snapshot_identity, graph.snapshot_identity))
        store = SnapshotStore()
        for snapshot in (transition, pair, trace, graph, composite):
            store.register(snapshot)
        self.assertEqual(store.view(graph.snapshot_identity, accepted_types=("graph",)).nodes[0].node_id, 1)
        with self.assertRaises(SnapshotError):
            store.view(graph.snapshot_identity, accepted_types=("tabular",))

    def test_malformed_views_fail_closed_and_store_rejects_tampering(self) -> None:
        transition = transition_snapshot(b"a", b"b", **self._kwargs())
        with self.assertRaises(SnapshotError):
            decode_pair(transition.payload)
        with self.assertRaises(SnapshotError):
            decode_trace(b"MNEL-R1\x00\x01\x05")
        with self.assertRaises(SnapshotError):
            decode_graph(b"MNEL-G1\x00\x01\x00\x00\x00")
        with self.assertRaises(SnapshotError):
            decode_composite(b"MNEL-C1\x00\x01\x00\x05abc")
        tampered = transition.payload[:-1]
        object.__setattr__(transition, "payload", tampered)
        with self.assertRaises(SnapshotError):
            SnapshotStore().register(transition)

    def test_graph_trace_and_composite_limits_are_explicit(self) -> None:
        with self.assertRaises(SnapshotError):
            trace_snapshot(tuple(("x", b"") for _ in range(257)), **self._kwargs())
        with self.assertRaises(SnapshotError):
            graph_snapshot(((1, "one"),), ((1, 2, "missing"),), **self._kwargs())
        component = transition_snapshot(b"a", b"b", **self._kwargs())
        with self.assertRaises(SnapshotError):
            composite_snapshot(tuple(component for _ in range(2)), **self._kwargs())


if __name__ == "__main__":
    unittest.main()
