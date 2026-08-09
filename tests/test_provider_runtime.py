import json
import unittest
from pathlib import Path

from mnel.learned_providers import DEFAULT_LEARNED_PROVIDER_REGISTRY
from mnel.provider_runtime import (
    ExecutionTier,
    ImplementationLanguage,
    NativeLanguageException,
    ProviderRuntimeManifest,
    load_runtime_manifest,
)
from mnel.placement import ExecutionDevice, OffloadMode, PlacementPolicy, Precision


class ProviderRuntimePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.declaration = DEFAULT_LEARNED_PROVIDER_REGISTRY.describe(
            "state.hidden-markov-model"
        )

    def manifest(self, **overrides: object) -> ProviderRuntimeManifest:
        values: dict[str, object] = {
            "provider_id": self.declaration.provider_id,
            "provider_version": self.declaration.version,
            "declaration_identity": self.declaration.declaration_identity,
            "artifact_identity": "sha256:runtime-artifact",
            "implementation_language": ImplementationLanguage.RUST,
            "execution_tier": ExecutionTier.NATIVE_TRUSTED,
        }
        values.update(overrides)
        return ProviderRuntimeManifest(**values)  # type: ignore[arg-type]

    def test_rust_is_native_default(self) -> None:
        manifest = self.manifest()
        manifest.validate_declaration(self.declaration)
        self.assertEqual(manifest.to_dict()["authority"], "diagnostic-only")
        self.assertNotIn("verdict", manifest.to_dict())

    def test_python_is_rejected_from_native_hot_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-Rust native"):
            self.manifest(implementation_language=ImplementationLanguage.PYTHON)

    def test_specialized_native_language_requires_exception_evidence(self) -> None:
        manifest = self.manifest(
            implementation_language=ImplementationLanguage.CPP,
            language_exception=NativeLanguageException(
                rationale="Specialized GPU kernel",
                benchmark_evidence_ids=("sha256:benchmark",),
                threat_review_id="sha256:threat-review",
            ),
        )
        self.assertEqual(manifest.execution_tier, ExecutionTier.NATIVE_TRUSTED)

    def test_placement_metadata_round_trips_without_changing_abi(self) -> None:
        manifest = self.manifest(
            placement_policy=PlacementPolicy(
                execution_device=ExecutionDevice.AUTO,
                offload=OffloadMode.SEQUENTIAL_CPU,
                precision=Precision.FLOAT16,
                gpu_reserve_mib=512,
                max_vram_mib=4096,
                model_storage_bytes=1024,
                workspace_bytes=2048,
            ),
        )
        raw = manifest.to_dict()
        self.assertEqual(raw["runtime"]["abi"], "mnel-provider-c-abi/1")
        restored = ProviderRuntimeManifest.from_dict(raw)
        self.assertEqual(restored.placement_policy.offload, OffloadMode.SEQUENTIAL_CPU)
        self.assertEqual(restored.placement_policy.precision, Precision.FLOAT16)

    def test_example_manifest_round_trips(self) -> None:
        path = Path("examples/learned-providers/runtime-manifest.json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["declaration_identity"] = self.declaration.declaration_identity
        manifest = ProviderRuntimeManifest.from_dict(raw)
        manifest.validate_declaration(self.declaration)

    def test_checked_in_manifest_loads(self) -> None:
        manifest = load_runtime_manifest(
            "examples/learned-providers/runtime-manifest.json"
        )
        self.assertEqual(manifest.provider_id, "state.hidden-markov-model")


if __name__ == "__main__":
    unittest.main()
