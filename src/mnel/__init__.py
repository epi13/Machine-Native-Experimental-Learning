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

__all__ = [
    "DEFAULT_LEARNED_PROVIDER_REGISTRY",
    "EvidenceLedger",
    "HardGateEvaluator",
    "LearnedProviderDeclaration",
    "LearnedProviderObservation",
    "LearnedProviderQuery",
    "LearnedProviderRegistry",
    "RecursionGovernor",
    "VerifiedExperienceDistiller",
    "canonical_digest",
]

__version__ = "0.1.0a0"
