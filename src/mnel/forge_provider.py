"""MNCS Forge Provider Protocol 0.1 adapter for bounded MNEL diagnostics.

This module is intentionally a narrow protocol boundary.  It exposes summaries of
identified MNEL material; it does not expose arbitrary Python execution, hidden
partitions, evaluator results, or promotion operations.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .core import canonical_digest, canonical_json

PROTOCOL_VERSION = "0.1"
PROVIDER_ID = "mnel-family-provider"
PROVIDER_IDENTITY = "mnel-family-provider-protocol-v1"
PROVIDER_VERSION = "0.2"
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 128 * 1024
ANALYSES = (
    "evidence_derivation",
    "mncs_bundle_validation",
    "provider_study_summary",
    "distributed_workload_inspection",
    "fabric_worker_capability_summary",
    "distributed_training_provenance",
    "shard_completeness",
    "reconciliation_summary",
)
FORBIDDEN_KEYS = frozenset(
    {
        "verdict",
        "evaluator_verdict",
        "evaluator_authority",
        "conformance",
        "mncs_conformance",
        "mncds_conformance",
        "promotion",
        "promotion_authorized",
        "ravel_promotion",
        "hidden_transfer",
        "future_final",
        "future-final",
        "hidden-transfer",
    }
)


class ForgeProviderError(ValueError):
    """A malformed or authority-expanding Provider Protocol request."""


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ForgeProviderError(f"forbidden authority field: {key}")
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def _provider() -> dict[str, str]:
    return {"id": PROVIDER_ID, "identity": PROVIDER_IDENTITY, "version": PROVIDER_VERSION}


def capabilities(request_id: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "capabilities",
        "provider": _provider(),
        "analyses": list(ANALYSES),
        "statuses": ["PASS", "FAIL", "UNKNOWN"],
        "cancellation": False,
        "health_checks": True,
        "extensions": {
            "supported_constructs": [
                "identified-study-summary",
                "provider-artifact-binding",
                "execution-receipt-binding",
                "distributed-workload-identity-binding",
                "fabric-worker-capability-observation",
                "sharded-training-provenance",
                "cross-node-reconciliation-observation",
            ],
            "unsupported_constructs": [
                "hidden-transfer-content",
                "future-final-content",
                "evaluator-verdict",
                "ravel-promotion",
            ],
            "limitations": [
                "diagnostic-only summaries; no external authority is created",
                "material must be supplied as bounded identities, not hidden records",
            ],
        },
    }
    if request_id is not None:
        value["request_id"] = request_id
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ForgeProviderError(f"{label} must be a sha256 identity")
    return value


def _analysis_response(request: dict[str, Any], status: str, summary: str, *, limitations: list[str]) -> dict[str, Any]:
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "analysis_response",
        "request_id": request.get("request_id"),
        "provider": _provider(),
        "status": status,
        "summary": summary[:1024],
        "witnesses_or_counterexamples": [],
        "limitations": limitations[:16],
        "extensions": {
            "mnel": {
                "authority": "diagnostic-only",
                "semantics": "diagnostic-only; not-a-verdict",
                "response_identity": canonical_digest(
                    {"request": request.get("request_id"), "status": status, "summary": summary}
                ),
            }
        },
    }
    return response


def handle_request(request: Any) -> dict[str, Any]:
    """Validate and handle exactly one decoded Protocol 0.1 request."""

    if not isinstance(request, dict):
        raise ForgeProviderError("request must be an object")
    _reject_forbidden(request)
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ForgeProviderError("unsupported protocol version")
    request_type = request.get("type")
    if request_type == "capabilities":
        request_id = request.get("request_id")
        if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
            raise ForgeProviderError("request_id must be a non-empty string")
        return capabilities(request_id)
    if request_type != "analysis_request":
        raise ForgeProviderError("request type must be capabilities or analysis_request")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ForgeProviderError("analysis request_id is required")
    analysis = request.get("analysis")
    if analysis not in ANALYSES:
        return _analysis_response(
            request,
            "UNKNOWN",
            "requested analysis is unsupported",
            limitations=["unsupported-analysis"],
        )
    component = request.get("component", {})
    if not isinstance(component, dict):
        raise ForgeProviderError("component must be an object")
    for key in ("candidate_identity", "source_epoch"):
        if key in component and not isinstance(component[key], (str, int)):
            raise ForgeProviderError(f"component.{key} must be scalar")
    identities = component.get("identities", {})
    if identities and not isinstance(identities, dict):
        raise ForgeProviderError("component.identities must be an object")
    for key, value in identities.items():
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ForgeProviderError(f"component.identities.{key} must be a bounded identity")
        if any(token in str(key).lower() for token in ("snapshot", "artifact", "study")):
            _identity(value, f"component.identities.{key}")
    limits = request.get("limits", {})
    if not isinstance(limits, dict):
        raise ForgeProviderError("limits must be an object")
    if limits.get("output_bytes", MAX_RESPONSE_BYTES) > MAX_RESPONSE_BYTES:
        raise ForgeProviderError("requested output limit exceeds provider ceiling")
    summary = f"bounded diagnostic analysis accepted: {analysis}"
    if analysis == "distributed_workload_inspection":
        summary = "distributed workload identity and declared capability requirements were inspected; execution placement remains a Fabric observation"
    elif analysis == "fabric_worker_capability_summary":
        summary = "Fabric worker capability observations were accepted as operator-supplied execution facts; they do not establish worker honesty or independence"
    elif analysis == "distributed_training_provenance":
        summary = "distributed training provenance was inspected without opening hidden or future-final records"
    elif analysis == "shard_completeness":
        summary = "training shard completeness was inspected; omitted or overlapping shards remain incomplete or invalid"
    elif analysis == "reconciliation_summary":
        summary = "Fabric reconciliation was retained as cross-node execution evidence, not a correctness result"
    return _analysis_response(
        request,
        "UNKNOWN",
        summary,
        limitations=["external evaluator and promotion authority remain outside MNEL"],
    )


def _error_response(message: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "error",
        "provider": _provider(),
        "code": "MNEL_PROVIDER_PROTOCOL_ERROR",
        "message": message[:512],
        "extensions": {},
    }


def process_line(raw: bytes) -> bytes:
    """Process one complete input buffer, preserving Forge's one-line framing."""

    if len(raw) > MAX_REQUEST_BYTES:
        return canonical_json(_error_response("request exceeds byte ceiling")) + b"\n"
    try:
        text = raw.decode("utf-8")
        lines = text.splitlines()
        if len(lines) != 1 or not lines[0].strip() or text.endswith("\n\n"):
            raise ForgeProviderError("request must contain exactly one JSON line")
        request = json.loads(lines[0])
        response = handle_request(request)
    except (UnicodeDecodeError, json.JSONDecodeError, ForgeProviderError) as error:
        response = _error_response(str(error))
    encoded = canonical_json(response)
    if len(encoded) > MAX_RESPONSE_BYTES:
        encoded = canonical_json(_error_response("response exceeds byte ceiling"))
    return encoded + b"\n"


def main() -> int:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    sys.stdout.buffer.write(process_line(raw))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
