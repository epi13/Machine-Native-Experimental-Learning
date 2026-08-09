import unittest

from mnel.snapshots import SnapshotError, pair_snapshot, tabular_snapshot, transition_snapshot


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
        with self.assertRaises(SnapshotError):
            tabular_snapshot(((float("nan"),),), **self._kwargs())


if __name__ == "__main__":
    unittest.main()
