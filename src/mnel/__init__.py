"""Machine-Native Experimental Learning."""

from .core import (
    EvidenceLedger,
    HardGateEvaluator,
    RecursionGovernor,
    VerifiedExperienceDistiller,
    canonical_digest,
)
from .learned_providers import (
    DEFAULT_LEARNED_PROVIDER_REGISTRY,
    LearnedProviderDeclaration,
    LearnedProviderObservation,
    LearnedProviderQuery,
    LearnedProviderRegistry,
)
from .placement import (
    AcceleratorDiagnostics,
    ExecutionDevice,
    ExecutionMode,
    OffloadMode,
    PlacementCapabilities,
    PlacementDecision,
    PlacementPolicy,
    Precision,
    decide_placement,
)
from .provider_runtime import (
    ExecutionTier,
    ImplementationLanguage,
    NativeLanguageException,
    ProviderRuntimeManifest,
    load_runtime_manifest,
)
from .recurrent_specialist import (
    CalibrationRecord,
    OperatingEnvelope,
    RecurrentSpecialistModel,
    SpecialistContextState,
    SpecialistDecision,
    SpecialistError,
    build_reference_artifacts,
    calibrate_recurrent_specialist,
    context_update,
    infer_batch,
    train_recurrent_specialist,
)

__all__ = [
    "DEFAULT_LEARNED_PROVIDER_REGISTRY",
    "AcceleratorDiagnostics",
    "CalibrationRecord",
    "EvidenceLedger",
    "ExecutionDevice",
    "ExecutionMode",
    "ExecutionTier",
    "HardGateEvaluator",
    "ImplementationLanguage",
    "LearnedProviderDeclaration",
    "LearnedProviderObservation",
    "LearnedProviderQuery",
    "LearnedProviderRegistry",
    "NativeLanguageException",
    "OffloadMode",
    "OperatingEnvelope",
    "PlacementCapabilities",
    "PlacementDecision",
    "PlacementPolicy",
    "Precision",
    "ProviderRuntimeManifest",
    "RecurrentSpecialistModel",
    "RecursionGovernor",
    "SpecialistContextState",
    "SpecialistDecision",
    "SpecialistError",
    "VerifiedExperienceDistiller",
    "build_reference_artifacts",
    "calibrate_recurrent_specialist",
    "canonical_digest",
    "context_update",
    "decide_placement",
    "infer_batch",
    "load_runtime_manifest",
    "train_recurrent_specialist",
]

__version__ = "0.1.0a0"
