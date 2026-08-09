"""Backend-neutral accelerator placement policy for learned providers.

The policy deliberately does not import Torch, Accelerate, or a vendor runtime. A
backend supplies capability diagnostics and an adapter applies the returned decision.
This keeps deterministic policy tests cheap and keeps the Rust host architecture
independent from any one accelerator stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

MIB = 1024 * 1024
DEFAULT_GPU_RESERVE_MIB = 256
DEFAULT_WORKSPACE_MIB = 256


class PlacementError(RuntimeError):
    """The requested placement cannot be satisfied safely."""


class ExecutionDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class OffloadMode(StrEnum):
    AUTO = "auto"
    NONE = "none"
    SEQUENTIAL_CPU = "sequential-cpu"


class Precision(StrEnum):
    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class ExecutionMode(StrEnum):
    CPU = "cpu"
    FULL_CUDA = "full-cuda"
    SEQUENTIAL_CPU_OFFLOAD = "sequential-cpu-offload"


@dataclass(frozen=True, slots=True)
class PlacementPolicy:
    execution_device: ExecutionDevice = ExecutionDevice.AUTO
    offload: OffloadMode = OffloadMode.AUTO
    precision: Precision = Precision.AUTO
    gpu_reserve_mib: int = DEFAULT_GPU_RESERVE_MIB
    max_vram_mib: int | None = None
    model_storage_bytes: int = 0
    workspace_bytes: int = DEFAULT_WORKSPACE_MIB * MIB
    host_memory_budget_bytes: int | None = None

    def validate(self) -> None:
        if self.execution_device is ExecutionDevice.CPU and self.offload is OffloadMode.SEQUENTIAL_CPU:
            raise PlacementError("sequential CPU offload requires auto or cuda execution device")
        if self.gpu_reserve_mib < 0:
            raise PlacementError("GPU reserve cannot be negative")
        if self.max_vram_mib is not None and self.max_vram_mib < 1:
            raise PlacementError("maximum VRAM must be positive")
        if self.model_storage_bytes < 0 or self.workspace_bytes < 0:
            raise PlacementError("model and workspace estimates cannot be negative")
        if self.host_memory_budget_bytes is not None and self.host_memory_budget_bytes < 1:
            raise PlacementError("host memory budget must be positive")


@dataclass(frozen=True, slots=True)
class PlacementCapabilities:
    supports_sequential_cpu_offload: bool = False
    cpu_precisions: tuple[Precision, ...] = (Precision.FLOAT32,)
    cuda_precisions: tuple[Precision, ...] = (
        Precision.FLOAT32,
        Precision.FLOAT16,
        Precision.BFLOAT16,
    )

    def validate(self) -> None:
        if not self.cpu_precisions:
            raise PlacementError("backend must declare at least one CPU precision")
        if not self.cuda_precisions:
            raise PlacementError("backend must declare at least one CUDA precision")


@dataclass(frozen=True, slots=True)
class AcceleratorDiagnostics:
    accelerator_available: bool = False
    execution_probe_succeeded: bool = False
    free_vram_bytes: int | None = None
    accelerator_identity: str | None = None
    float16_probe_succeeded: bool | None = None
    bfloat16_probe_succeeded: bool | None = None
    probe_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    execution_mode: ExecutionMode
    execution_device: ExecutionDevice
    offload: OffloadMode
    precision: Precision
    reason: str
    configured_gpu_reserve_bytes: int
    configured_max_vram_bytes: int | None
    effective_gpu_budget_bytes: int
    estimated_model_bytes: int
    estimated_workspace_bytes: int
    full_cuda_required_bytes: int
    host_memory_required_bytes: int
    sequential_offload_supported: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("execution_mode", "execution_device", "offload", "precision"):
            value[key] = value[key].value
        return value


def effective_gpu_budget_bytes(
    free_vram_bytes: int | None,
    reserve_mib: int,
    max_vram_mib: int | None,
) -> int:
    """Return free VRAM remaining after the operator reserve and optional cap."""

    if free_vram_bytes is None:
        return 0
    capped = free_vram_bytes
    if max_vram_mib is not None:
        capped = min(capped, max_vram_mib * MIB)
    return max(0, capped - reserve_mib * MIB)


def _precision_bytes(model_storage_bytes: int, precision: Precision) -> int:
    if precision in {Precision.FLOAT16, Precision.BFLOAT16}:
        return (model_storage_bytes + 1) // 2
    return model_storage_bytes


def _choose_cuda_precision(
    policy: PlacementPolicy,
    diagnostics: AcceleratorDiagnostics,
    capabilities: PlacementCapabilities,
) -> Precision:
    if policy.precision is not Precision.AUTO:
        if policy.precision not in capabilities.cuda_precisions:
            raise PlacementError(f"CUDA backend does not declare {policy.precision.value} support")
        if policy.precision is Precision.FLOAT16 and diagnostics.float16_probe_succeeded is not True:
            raise PlacementError("float16 was requested but its execution probe failed")
        if policy.precision is Precision.BFLOAT16 and diagnostics.bfloat16_probe_succeeded is not True:
            raise PlacementError("bfloat16 was requested but its execution probe failed")
        return policy.precision
    if (
        Precision.BFLOAT16 in capabilities.cuda_precisions
        and diagnostics.bfloat16_probe_succeeded is True
    ):
        return Precision.BFLOAT16
    if Precision.FLOAT16 in capabilities.cuda_precisions and diagnostics.float16_probe_succeeded is True:
        return Precision.FLOAT16
    if Precision.FLOAT32 in capabilities.cuda_precisions:
        return Precision.FLOAT32
    raise PlacementError("CUDA backend has no usable precision")


def decide_placement(
    policy: PlacementPolicy,
    diagnostics: AcceleratorDiagnostics,
    capabilities: PlacementCapabilities,
) -> PlacementDecision:
    """Choose CPU, full CUDA, or true sequential CPU offload deterministically."""

    policy.validate()
    capabilities.validate()
    model_bytes = policy.model_storage_bytes
    host_budget = policy.host_memory_budget_bytes
    if host_budget is not None and model_bytes > host_budget:
        raise PlacementError("provider model exceeds the host/system-memory budget")

    reserve_bytes = policy.gpu_reserve_mib * MIB
    maximum_bytes = policy.max_vram_mib * MIB if policy.max_vram_mib is not None else None
    gpu_budget = effective_gpu_budget_bytes(
        diagnostics.free_vram_bytes, policy.gpu_reserve_mib, policy.max_vram_mib
    )
    cuda_usable = diagnostics.accelerator_available and diagnostics.execution_probe_succeeded

    def decision(
        mode: ExecutionMode,
        precision: Precision,
        reason: str,
    ) -> PlacementDecision:
        estimated_model = _precision_bytes(model_bytes, precision)
        return PlacementDecision(
            execution_mode=mode,
            execution_device=(
                ExecutionDevice.CUDA if mode is not ExecutionMode.CPU else ExecutionDevice.CPU
            ),
            offload=(
                OffloadMode.SEQUENTIAL_CPU
                if mode is ExecutionMode.SEQUENTIAL_CPU_OFFLOAD
                else OffloadMode.NONE
            ),
            precision=precision,
            reason=reason,
            configured_gpu_reserve_bytes=reserve_bytes,
            configured_max_vram_bytes=maximum_bytes,
            effective_gpu_budget_bytes=gpu_budget,
            estimated_model_bytes=estimated_model,
            estimated_workspace_bytes=policy.workspace_bytes,
            full_cuda_required_bytes=estimated_model + policy.workspace_bytes,
            host_memory_required_bytes=model_bytes,
            sequential_offload_supported=capabilities.supports_sequential_cpu_offload,
        )

    if policy.execution_device is ExecutionDevice.CPU:
        if policy.precision is Precision.AUTO:
            precision = Precision.FLOAT32
        else:
            precision = policy.precision
        if precision not in capabilities.cpu_precisions:
            raise PlacementError(f"CPU backend does not declare {precision.value} support")
        return decision(ExecutionMode.CPU, precision, "CPU was explicitly requested")

    if not cuda_usable:
        reason = diagnostics.probe_error or "accelerator execution probe failed"
        if (
            policy.execution_device is ExecutionDevice.CUDA
            or policy.offload is OffloadMode.SEQUENTIAL_CPU
        ):
            raise PlacementError(f"CUDA execution is unusable: {reason}")
        return decision(ExecutionMode.CPU, Precision.FLOAT32, f"AUTO selected CPU because {reason}")

    precision = _choose_cuda_precision(policy, diagnostics, capabilities)
    required = _precision_bytes(model_bytes, precision) + policy.workspace_bytes
    fits = required <= gpu_budget

    if policy.offload is OffloadMode.SEQUENTIAL_CPU:
        if not capabilities.supports_sequential_cpu_offload:
            raise PlacementError("sequential CPU offload is unsupported by this provider")
        return decision(
            ExecutionMode.SEQUENTIAL_CPU_OFFLOAD,
            precision,
            "sequential CPU offload was explicitly requested",
        )

    if policy.offload is OffloadMode.NONE:
        if not fits:
            if policy.execution_device is ExecutionDevice.AUTO:
                return decision(
                    ExecutionMode.CPU,
                    Precision.FLOAT32,
                    "AUTO selected CPU because full CUDA exceeds the effective budget and offload is disabled",
                )
            raise PlacementError("full CUDA exceeds the effective GPU budget")
        return decision(
            ExecutionMode.FULL_CUDA,
            precision,
            "full CUDA was explicitly requested" if policy.execution_device is ExecutionDevice.CUDA else "full CUDA fits the effective GPU budget",
        )

    if fits:
        return decision(ExecutionMode.FULL_CUDA, precision, "full CUDA fits the effective GPU budget")
    if capabilities.supports_sequential_cpu_offload:
        return decision(
            ExecutionMode.SEQUENTIAL_CPU_OFFLOAD,
            precision,
            "full CUDA exceeds the effective GPU budget; using CPU-backed sequential execution",
        )
    if policy.execution_device is ExecutionDevice.AUTO:
        return decision(
            ExecutionMode.CPU,
            Precision.FLOAT32,
            "AUTO selected CPU because full CUDA exceeds budget and provider offload is unsupported",
        )
    raise PlacementError(
        "full CUDA exceeds the effective GPU budget and sequential offload is unsupported"
    )


def fallback_policy_after_oom(
    policy: PlacementPolicy,
    current_mode: ExecutionMode,
    capabilities: PlacementCapabilities,
) -> PlacementPolicy | None:
    """Return one bounded AUTO retry policy; explicit operator choices never retry."""

    if policy.execution_device is not ExecutionDevice.AUTO:
        return None
    if current_mode is ExecutionMode.FULL_CUDA and capabilities.supports_sequential_cpu_offload:
        return PlacementPolicy(
            execution_device=ExecutionDevice.AUTO,
            offload=OffloadMode.SEQUENTIAL_CPU,
            precision=policy.precision,
            gpu_reserve_mib=policy.gpu_reserve_mib,
            max_vram_mib=policy.max_vram_mib,
            model_storage_bytes=policy.model_storage_bytes,
            workspace_bytes=policy.workspace_bytes,
            host_memory_budget_bytes=policy.host_memory_budget_bytes,
        )
    if current_mode in {ExecutionMode.FULL_CUDA, ExecutionMode.SEQUENTIAL_CPU_OFFLOAD}:
        return PlacementPolicy(
            execution_device=ExecutionDevice.CPU,
            offload=OffloadMode.NONE,
            precision=Precision.FLOAT32,
            model_storage_bytes=policy.model_storage_bytes,
            workspace_bytes=policy.workspace_bytes,
            host_memory_budget_bytes=policy.host_memory_budget_bytes,
        )
    return None
