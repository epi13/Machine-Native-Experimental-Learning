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
from .provider_runtime import (
    ExecutionTier,
    ImplementationLanguage,
    NativeLanguageException,
    ProviderRuntimeManifest,
    load_runtime_manifest,
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

__all__ = [
    "DEFAULT_LEARNED_PROVIDER_REGISTRY",
    "EvidenceLedger",
    "AcceleratorDiagnostics",
    "ExecutionTier",
    "HardGateEvaluator",
    "ImplementationLanguage",
    "ExecutionDevice",
    "ExecutionMode",
    "LearnedProviderDeclaration",
    "LearnedProviderObservation",
    "LearnedProviderQuery",
    "LearnedProviderRegistry",
    "NativeLanguageException",
    "OffloadMode",
    "PlacementCapabilities",
    "PlacementDecision",
    "PlacementPolicy",
    "Precision",
    "ProviderRuntimeManifest",
    "RecursionGovernor",
    "VerifiedExperienceDistiller",
    "canonical_digest",
    "decide_placement",
    "load_runtime_manifest",
]

__version__ = "0.1.0a0"
