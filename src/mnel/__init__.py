"""Machine-Native Experimental Learning."""

from .core import (
    EvidenceLedger,
    HardGateEvaluator,
    RecursionGovernor,
    VerifiedExperienceDistiller,
    canonical_digest,
)

__all__ = [
    "EvidenceLedger",
    "HardGateEvaluator",
    "RecursionGovernor",
    "VerifiedExperienceDistiller",
    "canonical_digest",
]

__version__ = "0.1.0a0"
