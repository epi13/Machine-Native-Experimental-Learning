import json
import unittest
from pathlib import Path

from mnel.learned_providers import (
    DEFAULT_LEARNED_PROVIDER_REGISTRY,
    CostClass,
    LearnedProviderDeclaration,
    LearnedProviderQuery,
    LearnedProviderRegistry,
    OutputKind,
)


class LearnedProviderRegistryTests(unittest.TestCase):
    def test_default_catalog_is_heterogeneous_and_diagnostic_only(self) -> None:
        declarations = DEFAULT_LEARNED_PROVIDER_REGISTRY.list()
        self.assertGreaterEqual(len(declarations), 12)
        self.assertEqual(len({item.provider_id for item in declarations}), len(declarations))
        self.assertGreaterEqual(len({item.architecture_family for item in declarations}), 10)
        self.assertTrue(all(item.authority == "diagnostic-only" for item in declarations))
        self.assertTrue(all(not item.evaluator_eligible for item in declarations))
        self.assertTrue(all(item.declaration_identity.startswith("sha256:") for item in declarations))

    def test_transition_query_prefers_transition_jepa(self) -> None:
        query = LearnedProviderQuery(
            uncertainty_classes=("unexpected-transition",),
            artifact_types=("candidate-transition",),
            available_snapshot_types=("transition-feature-snapshot",),
            max_cost=CostClass.LOW,
        )
        matches = DEFAULT_LEARNED_PROVIDER_REGISTRY.match(query)
        self.assertTrue(matches)
        self.assertEqual(matches[0].declaration.provider_id, "latent.transition-jepa")

    def test_diverse_selection_uses_different_families_and_objectives(self) -> None:
        query = LearnedProviderQuery(
            uncertainty_classes=(
                "temporal-anomaly",
                "unexpected-transition",
                "cross-view-inconsistency",
                "metric-inconsistency",
            ),
        )
        selected = DEFAULT_LEARNED_PROVIDER_REGISTRY.select_diverse(query, limit=4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({item.declaration.architecture_family for item in selected}), 4)
        self.assertGreaterEqual(len({item.declaration.objective_family for item in selected}), 3)

    def test_snapshot_filter_is_enforced_when_availability_is_supplied(self) -> None:
        query = LearnedProviderQuery(
            uncertainty_classes=("ownership-transition",),
            artifact_types=("cfg",),
            available_snapshot_types=("ordered-event-snapshot",),
        )
        self.assertEqual(DEFAULT_LEARNED_PROVIDER_REGISTRY.match(query), ())

    def test_unknown_provider_and_duplicate_registry_are_rejected(self) -> None:
        declaration = DEFAULT_LEARNED_PROVIDER_REGISTRY.list()[0]
        with self.assertRaises(KeyError):
            DEFAULT_LEARNED_PROVIDER_REGISTRY.describe("missing.provider")
        with self.assertRaises(ValueError):
            LearnedProviderRegistry((declaration, declaration))

    def test_learned_provider_cannot_be_evaluator_eligible(self) -> None:
        value = DEFAULT_LEARNED_PROVIDER_REGISTRY.list()[0]
        fields = value.__dict__.copy()
        fields["evaluator_eligible"] = True
        fields.pop("authority", None)
        with self.assertRaises(ValueError):
            LearnedProviderDeclaration(**fields)

    def test_output_filter(self) -> None:
        query = LearnedProviderQuery(
            uncertainty_classes=("semantic-drift",),
            required_output_kinds=(OutputKind.PAIR_SIMILARITY,),
        )
        matches = DEFAULT_LEARNED_PROVIDER_REGISTRY.match(query)
        self.assertEqual([item.declaration.provider_id for item in matches], ["pair.contrastive-siamese"])

    def test_catalog_and_schema_are_machine_readable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads((root / "examples" / "learned-providers" / "catalog.json").read_text())
        schema = json.loads((root / "schemas" / "learned-provider-registry.schema.json").read_text())
        self.assertEqual(
            [item["provider_id"] for item in catalog],
            [item.provider_id for item in DEFAULT_LEARNED_PROVIDER_REGISTRY.list()],
        )
        self.assertEqual(len(catalog), 12)
        self.assertIn("declaration", schema["$defs"])
        self.assertIn("observation", schema["$defs"])
        self.assertEqual(
            schema["$defs"]["declaration"]["properties"]["evaluator_eligible"]["const"],
            False,
        )


if __name__ == "__main__":
    unittest.main()
