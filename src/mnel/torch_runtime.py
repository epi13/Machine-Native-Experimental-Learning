"""Optional Torch adapter for the generic MNEL placement policy.

Torch is intentionally imported only through a caller-supplied module. Core MNEL
installation and tests remain dependency-free. A sequential-offload result is marked
verified only when inference completed and observed module/parameter residency supports
that claim.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

from .placement import (
    AcceleratorDiagnostics,
    ExecutionMode,
    PlacementDecision,
    PlacementError,
    Precision,
)


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:512]


def _probe_dtype(torch: Any, dtype: Any) -> tuple[bool, str | None]:
    try:
        left = torch.ones((32, 32), device="cuda", dtype=dtype)
        right = torch.ones((32, 32), device="cuda", dtype=dtype)
        output = left @ right
        torch.cuda.synchronize(0)
        success = bool(torch.isfinite(output).all().item())
        return success, None if success else "CUDA dtype probe returned non-finite values"
    except Exception as error:  # backend exceptions are part of the diagnostic surface
        return False, _error_text(error)


def collect_torch_diagnostics(torch: Any) -> AcceleratorDiagnostics:
    """Require a real CUDA kernel probe instead of trusting discovery alone."""

    available = bool(torch.cuda.is_available())
    if not available:
        return AcceleratorDiagnostics(probe_error="installed Torch build has no CUDA runtime")
    try:
        free, _total = torch.cuda.mem_get_info(0)
        major, minor = torch.cuda.get_device_capability(0)
        float32_ok, float32_error = _probe_dtype(torch, torch.float32)
        float16_ok = _probe_dtype(torch, torch.float16)[0] if float32_ok else False
        bf16_reported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        bf16_ok = _probe_dtype(torch, torch.bfloat16)[0] if float32_ok and bf16_reported else False
        return AcceleratorDiagnostics(
            accelerator_available=True,
            execution_probe_succeeded=float32_ok,
            free_vram_bytes=int(free),
            accelerator_identity=f"{torch.cuda.get_device_name(0)} sm_{major}{minor}",
            float16_probe_succeeded=float16_ok,
            bfloat16_probe_succeeded=bf16_ok,
            probe_error=float32_error,
        )
    except Exception as error:
        return AcceleratorDiagnostics(
            accelerator_available=True,
            execution_probe_succeeded=False,
            probe_error=_error_text(error),
        )


def parameter_storage_bytes(model: Any) -> int:
    return int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))


def _torch_dtype(torch: Any, precision: Precision) -> Any:
    try:
        return {
            Precision.FLOAT32: torch.float32,
            Precision.FLOAT16: torch.float16,
            Precision.BFLOAT16: torch.bfloat16,
        }[precision]
    except KeyError as error:
        raise PlacementError(f"unsupported Torch precision: {precision.value}") from error


def apply_torch_placement(
    model: Any,
    torch: Any,
    decision: PlacementDecision,
    *,
    cpu_offload_fn: Callable[..., Any] | None = None,
    preload_module_classes: tuple[str, ...] = (),
) -> Any:
    """Apply a decision; sequential mode uses Accelerate's CPU-backed hooks."""

    dtype = _torch_dtype(torch, decision.precision)
    model.to(device="cpu", dtype=dtype)
    if decision.execution_mode is ExecutionMode.CPU:
        return model
    if decision.execution_mode is ExecutionMode.FULL_CUDA:
        return model.to("cuda")
    if decision.execution_mode is not ExecutionMode.SEQUENTIAL_CPU_OFFLOAD:
        raise PlacementError(f"unsupported execution mode: {decision.execution_mode.value}")
    if cpu_offload_fn is None:
        try:
            from accelerate import cpu_offload as cpu_offload_fn
        except ImportError as error:
            raise PlacementError(
                "sequential CPU offload requires Accelerate in the optional provider environment"
            ) from error
    return cpu_offload_fn(
        model,
        execution_device=torch.device("cuda"),
        offload_buffers=True,
        preload_module_classes=list(preload_module_classes) or None,
    )


def offload_evidence(model: Any, *, inference_completed: bool) -> dict[str, Any]:
    """Return observed evidence; requested offload alone never sets verified true."""

    hooks = sum(int(hasattr(module, "_hf_hook")) for module in model.modules())
    meta_bytes = 0
    cuda_bytes = 0
    devices: dict[str, int] = {}
    for parameter in model.parameters():
        device = str(parameter.device)
        devices[device] = devices.get(device, 0) + 1
        size = int(parameter.numel() * parameter.element_size())
        if device == "meta":
            meta_bytes += size
        elif device.startswith("cuda"):
            cuda_bytes += size
    return {
        "sequential_offload_hook_count": hooks,
        "offloaded_meta_parameter_bytes": meta_bytes,
        "persistent_cuda_parameter_bytes": cuda_bytes,
        "parameter_device_counts": dict(sorted(devices.items())),
        "inference_completed": inference_completed,
        "sequential_offload_verified": bool(
            inference_completed and hooks > 0 and meta_bytes > 0 and cuda_bytes == 0
        ),
    }


def cuda_memory_snapshot(torch: Any) -> dict[str, int | None]:
    if not bool(torch.cuda.is_available()):
        return {"cuda_allocated_bytes": None, "cuda_reserved_bytes": None}
    return {
        "cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def reset_cuda_peaks(torch: Any) -> None:
    if bool(torch.cuda.is_available()):
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)


def restore_model_to_cpu(model: Any, torch: Any, *, remove_hooks_fn: Callable[[Any], Any] | None = None) -> Any:
    if any(hasattr(module, "_hf_hook") for module in model.modules()):
        if remove_hooks_fn is None:
            from accelerate.hooks import remove_hook_from_submodules as remove_hooks_fn

        remove_hooks_fn(model)
    model.to("cpu")
    gc.collect()
    if bool(torch.cuda.is_available()):
        torch.cuda.empty_cache()
    return model
