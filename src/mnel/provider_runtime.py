"""Runtime admission contract for diagnostic-only learned micro-providers.

Python remains the orchestration, training, and research surface. The admitted native
hot path defaults to the versioned Rust host and C ABI defined by this repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .learned_providers import LearnedProviderDeclaration

PROVIDER_ABI_V1 = "mnel-provider-c-abi/1"
RUNTIME_MANIFEST_SCHEMA = "mnel-learned-provider-runtime-manifest/0.1"


class ImplementationLanguage(str, Enum):
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    ZIG = "zig"
    WASM = "wasm"
    PYTHON = "python"
    OTHER = "other"


class ExecutionTier(str, Enum):
    NATIVE_TRUSTED = "native-trusted"
    WASM_QUARANTINED = "wasm-quarantined"
    EXTERNAL_EXPERIMENTAL = "external-experimental"


@dataclass(frozen=True)
class NativeLanguageException:
    rationale: str
    benchmark_evidence_ids: tuple[str, ...]
    threat_review_id: str

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ValueError("language exception rationale must not be empty")
        if not self.benchmark_evidence_ids or any(
            not item.strip() for item in self.benchmark_evidence_ids
        ):
            raise ValueError("language exception requires benchmark evidence identities")
        if not self.threat_review_id.strip():
            raise ValueError("language exception requires a threat review identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "rationale": self.rationale,
            "benchmark_evidence_ids": list(self.benchmark_evidence_ids),
            "threat_review_id": self.threat_review_id,
        }


@dataclass(frozen=True)
class ProviderRuntimeManifest:
    provider_id: str
    provider_version: str
    declaration_identity: str
    artifact_identity: str
    implementation_language: ImplementationLanguage
    execution_tier: ExecutionTier
    abi: str = PROVIDER_ABI_V1
    persistent_host: bool = True
    snapshot_transport: str = "identity-bound-compact-binary"
    weight_residency: str = "resident-on-admission"
    language_exception: NativeLanguageException | None = None
    authority: str = field(default="diagnostic-only", init=False)
    verdict_semantics: str = field(default="not-a-verdict", init=False)

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "provider_version",
            "declaration_identity",
            "artifact_identity",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.abi != PROVIDER_ABI_V1:
            raise ValueError(f"unsupported provider ABI: {self.abi}")
        if not self.persistent_host:
            raise ValueError("process-per-invocation providers are forbidden")
        if self.snapshot_transport != "identity-bound-compact-binary":
            raise ValueError("hot-path snapshots must use identity-bound compact binary transport")
        if self.weight_residency != "resident-on-admission":
            raise ValueError("admitted provider weights must remain resident")
        if self.execution_tier is ExecutionTier.NATIVE_TRUSTED:
            if (
                self.implementation_language is not ImplementationLanguage.RUST
                and self.language_exception is None
            ):
                raise ValueError("non-Rust native providers require an evidence-backed exception")
        elif self.execution_tier is ExecutionTier.WASM_QUARANTINED:
            if self.implementation_language is not ImplementationLanguage.WASM:
                raise ValueError("wasm-quarantined tier requires a WASM provider")
        elif self.language_exception is not None:
            raise ValueError("language exceptions apply only to non-Rust native providers")

    def validate_declaration(self, declaration: LearnedProviderDeclaration) -> None:
        if self.provider_id != declaration.provider_id:
            raise ValueError("runtime manifest provider_id does not match declaration")
        if self.provider_version != declaration.version:
            raise ValueError("runtime manifest provider_version does not match declaration")
        if self.declaration_identity != declaration.declaration_identity:
            raise ValueError("runtime manifest declaration identity does not match declaration")
        if declaration.evaluator_eligible or declaration.authority != "diagnostic-only":
            raise ValueError("runtime cannot admit an evaluator-eligible learned provider")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "declaration_identity": self.declaration_identity,
            "artifact_identity": self.artifact_identity,
            "runtime": {
                "implementation_language": self.implementation_language.value,
                "execution_tier": self.execution_tier.value,
                "abi": self.abi,
                "persistent_host": self.persistent_host,
                "snapshot_transport": self.snapshot_transport,
                "weight_residency": self.weight_residency,
            },
            "authority": self.authority,
            "verdict_semantics": self.verdict_semantics,
        }
        if self.language_exception is not None:
            value["language_exception"] = self.language_exception.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProviderRuntimeManifest":
        if value.get("schema") != RUNTIME_MANIFEST_SCHEMA:
            raise ValueError("unsupported runtime manifest schema")
        runtime = value.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("runtime manifest requires a runtime object")
        exception_value = value.get("language_exception")
        exception = None
        if exception_value is not None:
            if not isinstance(exception_value, dict):
                raise ValueError("language_exception must be an object")
            exception = NativeLanguageException(
                rationale=str(exception_value.get("rationale", "")),
                benchmark_evidence_ids=tuple(exception_value.get("benchmark_evidence_ids", ())),
                threat_review_id=str(exception_value.get("threat_review_id", "")),
            )
        manifest = cls(
            provider_id=str(value.get("provider_id", "")),
            provider_version=str(value.get("provider_version", "")),
            declaration_identity=str(value.get("declaration_identity", "")),
            artifact_identity=str(value.get("artifact_identity", "")),
            implementation_language=ImplementationLanguage(
                runtime.get("implementation_language")
            ),
            execution_tier=ExecutionTier(runtime.get("execution_tier")),
            abi=str(runtime.get("abi", "")),
            persistent_host=bool(runtime.get("persistent_host", False)),
            snapshot_transport=str(runtime.get("snapshot_transport", "")),
            weight_residency=str(runtime.get("weight_residency", "")),
            language_exception=exception,
        )
        if value.get("authority") != manifest.authority:
            raise ValueError("runtime manifest authority must be diagnostic-only")
        if value.get("verdict_semantics") != manifest.verdict_semantics:
            raise ValueError("runtime manifest verdict semantics must be not-a-verdict")
        return manifest


def load_runtime_manifest(path: str | Path) -> ProviderRuntimeManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime manifest must be a JSON object")
    return ProviderRuntimeManifest.from_dict(value)
