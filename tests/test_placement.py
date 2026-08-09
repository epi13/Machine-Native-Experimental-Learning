import unittest

from mnel.placement import (
    MIB,
    AcceleratorDiagnostics,
    ExecutionDevice,
    ExecutionMode,
    OffloadMode,
    PlacementCapabilities,
    PlacementError,
    PlacementPolicy,
    Precision,
    decide_placement,
    effective_gpu_budget_bytes,
    fallback_policy_after_oom,
)


def cuda(*, free_mib: int = 4096, float16: bool = True, bfloat16: bool = False) -> AcceleratorDiagnostics:
    return AcceleratorDiagnostics(
        accelerator_available=True,
        execution_probe_succeeded=True,
        free_vram_bytes=free_mib * MIB,
        accelerator_identity="fake-cuda-0",
        float16_probe_succeeded=float16,
        bfloat16_probe_succeeded=bfloat16,
    )


class PlacementPolicyTests(unittest.TestCase):
    def test_reserve_and_cap_math(self) -> None:
        self.assertEqual(
            effective_gpu_budget_bytes(2048 * MIB, reserve_mib=256, max_vram_mib=1024),
            768 * MIB,
        )
        self.assertEqual(effective_gpu_budget_bytes(None, 0, None), 0)

    def test_cpu_is_selected_without_accelerator(self) -> None:
        decision = decide_placement(
            PlacementPolicy(model_storage_bytes=64 * MIB),
            AcceleratorDiagnostics(probe_error="no CUDA runtime"),
            PlacementCapabilities(supports_sequential_cpu_offload=True),
        )
        self.assertEqual(decision.execution_mode, ExecutionMode.CPU)
        self.assertEqual(decision.execution_device, ExecutionDevice.CPU)
        self.assertEqual(decision.precision, Precision.FLOAT32)

    def test_auto_selects_full_cuda_when_model_and_workspace_fit(self) -> None:
        decision = decide_placement(
            PlacementPolicy(model_storage_bytes=256 * MIB, workspace_bytes=128 * MIB),
            cuda(free_mib=1024),
            PlacementCapabilities(supports_sequential_cpu_offload=True),
        )
        self.assertEqual(decision.execution_mode, ExecutionMode.FULL_CUDA)
        self.assertEqual(decision.precision, Precision.FLOAT16)

    def test_auto_selects_sequential_offload_when_full_model_does_not_fit(self) -> None:
        decision = decide_placement(
            PlacementPolicy(model_storage_bytes=2048 * MIB, workspace_bytes=512 * MIB),
            cuda(free_mib=1024),
            PlacementCapabilities(supports_sequential_cpu_offload=True),
        )
        self.assertEqual(decision.execution_mode, ExecutionMode.SEQUENTIAL_CPU_OFFLOAD)
        self.assertEqual(decision.offload, OffloadMode.SEQUENTIAL_CPU)

    def test_auto_falls_back_to_cpu_when_offload_is_unavailable(self) -> None:
        decision = decide_placement(
            PlacementPolicy(model_storage_bytes=2048 * MIB, workspace_bytes=512 * MIB),
            cuda(free_mib=1024),
            PlacementCapabilities(supports_sequential_cpu_offload=False),
        )
        self.assertEqual(decision.execution_mode, ExecutionMode.CPU)

    def test_explicit_cuda_does_not_silently_fallback(self) -> None:
        with self.assertRaisesRegex(PlacementError, "exceeds"):
            decide_placement(
                PlacementPolicy(
                    execution_device=ExecutionDevice.CUDA,
                    offload=OffloadMode.NONE,
                    model_storage_bytes=2048 * MIB,
                    workspace_bytes=512 * MIB,
                ),
                cuda(free_mib=1024),
                PlacementCapabilities(supports_sequential_cpu_offload=True),
            )

    def test_explicit_offload_requires_real_cuda_and_capability(self) -> None:
        with self.assertRaisesRegex(PlacementError, "unsupported"):
            decide_placement(
                PlacementPolicy(
                    execution_device=ExecutionDevice.CUDA,
                    offload=OffloadMode.SEQUENTIAL_CPU,
                ),
                cuda(),
                PlacementCapabilities(supports_sequential_cpu_offload=False),
            )
        with self.assertRaisesRegex(PlacementError, "unusable"):
            decide_placement(
                PlacementPolicy(offload=OffloadMode.SEQUENTIAL_CPU),
                AcceleratorDiagnostics(probe_error="kernel probe failed"),
                PlacementCapabilities(supports_sequential_cpu_offload=True),
            )

    def test_dtype_probe_is_required_for_explicit_precision(self) -> None:
        with self.assertRaisesRegex(PlacementError, "float16"):
            decide_placement(
                PlacementPolicy(precision=Precision.FLOAT16),
                cuda(float16=False),
                PlacementCapabilities(),
            )

    def test_oom_recovery_is_bounded_and_auto_only(self) -> None:
        policy = PlacementPolicy(model_storage_bytes=512)
        retry = fallback_policy_after_oom(
            policy, ExecutionMode.FULL_CUDA, PlacementCapabilities(supports_sequential_cpu_offload=True)
        )
        self.assertIsNotNone(retry)
        self.assertEqual(retry.offload, OffloadMode.SEQUENTIAL_CPU)
        final = fallback_policy_after_oom(
            retry,
            ExecutionMode.SEQUENTIAL_CPU_OFFLOAD,
            PlacementCapabilities(supports_sequential_cpu_offload=True),
        )
        self.assertIsNotNone(final)
        self.assertEqual(final.execution_device, ExecutionDevice.CPU)
        self.assertIsNone(
            fallback_policy_after_oom(
                PlacementPolicy(execution_device=ExecutionDevice.CUDA),
                ExecutionMode.FULL_CUDA,
                PlacementCapabilities(supports_sequential_cpu_offload=True),
            )
        )

    def test_host_memory_budget_is_enforced(self) -> None:
        with self.assertRaisesRegex(PlacementError, "host/system-memory"):
            decide_placement(
                PlacementPolicy(model_storage_bytes=1024, host_memory_budget_bytes=512),
                AcceleratorDiagnostics(),
                PlacementCapabilities(),
            )


if __name__ == "__main__":
    unittest.main()
