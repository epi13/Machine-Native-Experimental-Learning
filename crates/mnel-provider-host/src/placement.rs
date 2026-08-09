//! Backend-neutral accelerator placement policy.
//!
//! This module records policy and decisions only. A CUDA, vendor, or Torch adapter
//! supplies diagnostics and applies the decision behind an explicit capability boundary.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionDevice {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OffloadMode {
    Auto,
    None,
    SequentialCpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Precision {
    Auto,
    Float32,
    Float16,
    Bfloat16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    Cpu,
    FullCuda,
    SequentialCpuOffload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementPolicy {
    pub execution_device: ExecutionDevice,
    pub offload: OffloadMode,
    pub precision: Precision,
    pub gpu_reserve_bytes: u64,
    pub max_vram_bytes: Option<u64>,
    pub model_storage_bytes: u64,
    pub workspace_bytes: u64,
    pub host_memory_budget_bytes: Option<u64>,
}

impl Default for PlacementPolicy {
    fn default() -> Self {
        Self {
            execution_device: ExecutionDevice::Auto,
            offload: OffloadMode::Auto,
            precision: Precision::Auto,
            gpu_reserve_bytes: 256 * 1024 * 1024,
            max_vram_bytes: None,
            model_storage_bytes: 0,
            workspace_bytes: 256 * 1024 * 1024,
            host_memory_budget_bytes: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementCapabilities {
    pub supports_sequential_cpu_offload: bool,
    pub cpu_precisions: Vec<Precision>,
    pub cuda_precisions: Vec<Precision>,
}

impl Default for PlacementCapabilities {
    fn default() -> Self {
        Self {
            supports_sequential_cpu_offload: false,
            cpu_precisions: vec![Precision::Float32],
            cuda_precisions: vec![Precision::Float32, Precision::Float16, Precision::Bfloat16],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceleratorDiagnostics {
    pub accelerator_available: bool,
    pub execution_probe_succeeded: bool,
    pub free_vram_bytes: Option<u64>,
    pub accelerator_identity: Option<String>,
    pub float16_probe_succeeded: Option<bool>,
    pub bfloat16_probe_succeeded: Option<bool>,
    pub probe_error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementDecision {
    pub execution_mode: ExecutionMode,
    pub execution_device: ExecutionDevice,
    pub offload: OffloadMode,
    pub precision: Precision,
    pub reason: String,
    pub configured_gpu_reserve_bytes: u64,
    pub configured_max_vram_bytes: Option<u64>,
    pub effective_gpu_budget_bytes: u64,
    pub estimated_model_bytes: u64,
    pub estimated_workspace_bytes: u64,
    pub full_cuda_required_bytes: u64,
    pub host_memory_required_bytes: u64,
    pub sequential_offload_supported: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementError {
    InvalidPolicy,
    MissingCpuPrecision,
    MissingCudaPrecision,
    HostMemoryBudgetExceeded,
    CudaUnavailable,
    UnsupportedPrecision,
    PrecisionProbeFailed,
    SequentialOffloadUnsupported,
    FullCudaBudgetExceeded,
}

fn precision_bytes(model_storage_bytes: u64, precision: Precision) -> u64 {
    match precision {
        Precision::Float16 | Precision::Bfloat16 => model_storage_bytes.div_ceil(2),
        Precision::Auto | Precision::Float32 => model_storage_bytes,
    }
}

fn cuda_precision(
    policy: &PlacementPolicy,
    diagnostics: &AcceleratorDiagnostics,
    capabilities: &PlacementCapabilities,
) -> Result<Precision, PlacementError> {
    let precision = match policy.precision {
        Precision::Auto => {
            if capabilities.cuda_precisions.contains(&Precision::Bfloat16)
                && diagnostics.bfloat16_probe_succeeded == Some(true)
            {
                Precision::Bfloat16
            } else if capabilities.cuda_precisions.contains(&Precision::Float16)
                && diagnostics.float16_probe_succeeded == Some(true)
            {
                Precision::Float16
            } else {
                Precision::Float32
            }
        }
        requested => requested,
    };
    if !capabilities.cuda_precisions.contains(&precision) || precision == Precision::Auto {
        return Err(PlacementError::UnsupportedPrecision);
    }
    if (precision == Precision::Float16 && diagnostics.float16_probe_succeeded != Some(true))
        || (precision == Precision::Bfloat16 && diagnostics.bfloat16_probe_succeeded != Some(true))
    {
        return Err(PlacementError::PrecisionProbeFailed);
    }
    Ok(precision)
}

pub fn effective_gpu_budget_bytes(
    free_vram_bytes: Option<u64>,
    reserve_bytes: u64,
    max_vram_bytes: Option<u64>,
) -> u64 {
    let Some(free) = free_vram_bytes else {
        return 0;
    };
    let capped = max_vram_bytes.map_or(free, |maximum| free.min(maximum));
    capped.saturating_sub(reserve_bytes)
}

pub fn decide_placement(
    policy: &PlacementPolicy,
    diagnostics: &AcceleratorDiagnostics,
    capabilities: &PlacementCapabilities,
) -> Result<PlacementDecision, PlacementError> {
    if policy.execution_device == ExecutionDevice::Cpu
        && policy.offload == OffloadMode::SequentialCpu
    {
        return Err(PlacementError::InvalidPolicy);
    }
    if capabilities.cpu_precisions.is_empty() {
        return Err(PlacementError::MissingCpuPrecision);
    }
    if capabilities.cuda_precisions.is_empty() {
        return Err(PlacementError::MissingCudaPrecision);
    }
    if policy.max_vram_bytes == Some(0) || policy.host_memory_budget_bytes == Some(0) {
        return Err(PlacementError::InvalidPolicy);
    }
    if policy
        .host_memory_budget_bytes
        .is_some_and(|budget| policy.model_storage_bytes > budget)
    {
        return Err(PlacementError::HostMemoryBudgetExceeded);
    }

    let gpu_budget = effective_gpu_budget_bytes(
        diagnostics.free_vram_bytes,
        policy.gpu_reserve_bytes,
        policy.max_vram_bytes,
    );
    let usable = diagnostics.accelerator_available && diagnostics.execution_probe_succeeded;

    let make_decision = |mode: ExecutionMode, precision: Precision, reason: &str| {
        let estimated_model = precision_bytes(policy.model_storage_bytes, precision);
        PlacementDecision {
            execution_mode: mode,
            execution_device: if mode == ExecutionMode::Cpu {
                ExecutionDevice::Cpu
            } else {
                ExecutionDevice::Cuda
            },
            offload: if mode == ExecutionMode::SequentialCpuOffload {
                OffloadMode::SequentialCpu
            } else {
                OffloadMode::None
            },
            precision,
            reason: reason.to_owned(),
            configured_gpu_reserve_bytes: policy.gpu_reserve_bytes,
            configured_max_vram_bytes: policy.max_vram_bytes,
            effective_gpu_budget_bytes: gpu_budget,
            estimated_model_bytes: estimated_model,
            estimated_workspace_bytes: policy.workspace_bytes,
            full_cuda_required_bytes: estimated_model.saturating_add(policy.workspace_bytes),
            host_memory_required_bytes: policy.model_storage_bytes,
            sequential_offload_supported: capabilities.supports_sequential_cpu_offload,
        }
    };

    if policy.execution_device == ExecutionDevice::Cpu {
        let precision = if policy.precision == Precision::Auto {
            Precision::Float32
        } else {
            policy.precision
        };
        if !capabilities.cpu_precisions.contains(&precision) {
            return Err(PlacementError::UnsupportedPrecision);
        }
        return Ok(make_decision(
            ExecutionMode::Cpu,
            precision,
            "CPU was explicitly requested",
        ));
    }

    if !usable {
        if policy.execution_device == ExecutionDevice::Cuda
            || policy.offload == OffloadMode::SequentialCpu
        {
            return Err(PlacementError::CudaUnavailable);
        }
        return Ok(make_decision(
            ExecutionMode::Cpu,
            Precision::Float32,
            "AUTO selected CPU because accelerator execution is unavailable",
        ));
    }

    let precision = cuda_precision(policy, diagnostics, capabilities)?;
    let required = precision_bytes(policy.model_storage_bytes, precision)
        .saturating_add(policy.workspace_bytes);
    let fits = required <= gpu_budget;
    if policy.offload == OffloadMode::SequentialCpu {
        if !capabilities.supports_sequential_cpu_offload {
            return Err(PlacementError::SequentialOffloadUnsupported);
        }
        return Ok(make_decision(
            ExecutionMode::SequentialCpuOffload,
            precision,
            "sequential CPU offload was explicitly requested",
        ));
    }
    if policy.offload == OffloadMode::None {
        if !fits {
            if policy.execution_device == ExecutionDevice::Auto {
                return Ok(make_decision(
                    ExecutionMode::Cpu,
                    Precision::Float32,
                    "AUTO selected CPU because full CUDA exceeds budget and offload is disabled",
                ));
            }
            return Err(PlacementError::FullCudaBudgetExceeded);
        }
        return Ok(make_decision(
            ExecutionMode::FullCuda,
            precision,
            "full CUDA fits the effective GPU budget",
        ));
    }
    if fits {
        return Ok(make_decision(
            ExecutionMode::FullCuda,
            precision,
            "full CUDA fits the effective GPU budget",
        ));
    }
    if capabilities.supports_sequential_cpu_offload {
        return Ok(make_decision(
            ExecutionMode::SequentialCpuOffload,
            precision,
            "full CUDA exceeds budget; using CPU-backed sequential execution",
        ));
    }
    if policy.execution_device == ExecutionDevice::Auto {
        return Ok(make_decision(
            ExecutionMode::Cpu,
            Precision::Float32,
            "AUTO selected CPU because full CUDA exceeds budget and offload is unsupported",
        ));
    }
    Err(PlacementError::FullCudaBudgetExceeded)
}

pub fn fallback_after_oom(
    policy: &PlacementPolicy,
    current_mode: ExecutionMode,
    capabilities: &PlacementCapabilities,
) -> Option<PlacementPolicy> {
    if policy.execution_device != ExecutionDevice::Auto {
        return None;
    }
    if current_mode == ExecutionMode::FullCuda && capabilities.supports_sequential_cpu_offload {
        return Some(PlacementPolicy {
            execution_device: ExecutionDevice::Auto,
            offload: OffloadMode::SequentialCpu,
            ..policy.clone()
        });
    }
    if matches!(
        current_mode,
        ExecutionMode::FullCuda | ExecutionMode::SequentialCpuOffload
    ) {
        return Some(PlacementPolicy {
            execution_device: ExecutionDevice::Cpu,
            offload: OffloadMode::None,
            precision: Precision::Float32,
            ..policy.clone()
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda(free_mib: u64) -> AcceleratorDiagnostics {
        AcceleratorDiagnostics {
            accelerator_available: true,
            execution_probe_succeeded: true,
            free_vram_bytes: Some(free_mib * 1024 * 1024),
            accelerator_identity: Some("fake-cuda".to_owned()),
            float16_probe_succeeded: Some(true),
            bfloat16_probe_succeeded: Some(false),
            probe_error: None,
        }
    }

    #[test]
    fn reserve_and_cap_math_is_deterministic() {
        assert_eq!(
            effective_gpu_budget_bytes(
                Some(2048 * 1024 * 1024),
                256 * 1024 * 1024,
                Some(1024 * 1024 * 1024)
            ),
            768 * 1024 * 1024
        );
    }

    #[test]
    fn auto_selects_sequential_offload_when_full_cuda_does_not_fit() {
        let policy = PlacementPolicy {
            model_storage_bytes: 2048 * 1024 * 1024,
            workspace_bytes: 512 * 1024 * 1024,
            ..PlacementPolicy::default()
        };
        let capabilities = PlacementCapabilities {
            supports_sequential_cpu_offload: true,
            ..PlacementCapabilities::default()
        };
        let decision = match decide_placement(&policy, &cuda(1024), &capabilities) {
            Ok(decision) => decision,
            Err(_) => panic!("placement should succeed"),
        };
        assert_eq!(decision.execution_mode, ExecutionMode::SequentialCpuOffload);
    }

    #[test]
    fn explicit_cuda_budget_failure_is_not_silently_recovered() {
        let policy = PlacementPolicy {
            execution_device: ExecutionDevice::Cuda,
            offload: OffloadMode::None,
            model_storage_bytes: 2048 * 1024 * 1024,
            ..PlacementPolicy::default()
        };
        assert_eq!(
            decide_placement(&policy, &cuda(1024), &PlacementCapabilities::default()),
            Err(PlacementError::FullCudaBudgetExceeded)
        );
    }
}
