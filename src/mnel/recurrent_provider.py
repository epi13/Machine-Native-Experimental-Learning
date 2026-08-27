"""Bounded stdin/stdout provider boundary for the recurrent specialist."""

from __future__ import annotations

import json
import sys
from typing import Any

from .core import canonical_digest
from .recurrent_specialist import (
    AUTHORITY,
    PROTOCOL_VERSION,
    RecurrentSpecialistModel,
    SpecialistError,
    infer_batch,
)

MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 512 * 1024


def capabilities(request_id: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "capabilities",
        "provider": {
            "id": "mnel-bounded-recurrent-specialist",
            "identity": "mnel-bounded-recurrent-specialist-provider-v1",
            "version": "0.1",
        },
        "analyses": ["bounded_recurrent_inference", "structured_abstention", "context_state_update"],
        "statuses": ["PASS", "FAIL", "UNKNOWN"],
        "cancellation": False,
        "health_checks": True,
        "extensions": {
            "provider_abi": "mnel-specialist-provider-abi/0.1",
            "supported_constructs": ["identity-bound-artifact", "persistent-context-state", "masked-recurrent-update", "explicit-budget"],
            "unsupported_constructs": ["verdict", "permission", "promotion", "credentials", "general-language-generation"],
            "limitations": ["diagnostic-only structured proposals; no authority is created"],
        },
    }
    if request_id is not None:
        value["request_id"] = request_id
    return value


def handle_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise SpecialistError("request must be an object")
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise SpecialistError("unsupported specialist protocol version")
    if request.get("type") == "capabilities":
        return capabilities(request.get("request_id"))
    if request.get("type") != "infer":
        raise SpecialistError("request type must be capabilities or infer")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise SpecialistError("request_id is required")
    artifact = request.get("artifact")
    model = RecurrentSpecialistModel.load(artifact if isinstance(artifact, dict) else json.dumps(artifact).encode("utf-8"))
    queries = request.get("queries")
    if not isinstance(queries, list):
        raise SpecialistError("queries must be a bounded array")
    context_observations = request.get("context_observations", [])
    if not isinstance(context_observations, list):
        raise SpecialistError("context_observations must be an array")
    results = infer_batch(
        model,
        queries,
        context_observations=context_observations,
        max_iterations=request.get("max_iterations"),
        lineage_identity=request.get("lineage_identity"),
    )
    value = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "inference_response",
        "request_id": request_id,
        "provider": capabilities()["provider"],
        "provider_abi": model.provider_abi,
        "model_identity": model.model_identity,
        "generation_identity": model.generation_identity,
        "target_role": model.target_role,
        "results": [result.to_dict() for result in results],
        "authority": AUTHORITY,
        "semantics": "bounded-recurrent-structured-decisions; not-a-verdict",
    }
    value["response_identity"] = canonical_digest(value)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise SpecialistError("specialist response exceeds its output bound")
    return value


def main() -> int:
    for line in sys.stdin:
        if len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
            response = {"status": "UNKNOWN", "error": "request exceeds byte bound", "authority": AUTHORITY}
        else:
            try:
                response = handle_request(json.loads(line))
            except (SpecialistError, json.JSONDecodeError) as error:
                response = {"status": "UNKNOWN", "error": str(error), "authority": AUTHORITY}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
