"""Bounded MNEL adapters for the current MNCS-family public boundaries.

The adapters preserve external identities as opaque references.  They are not a
Forge, Fabric, Commons, RAVEL, or MNCS implementation and never upgrade an
execution observation into a correctness or promotion decision.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .core import EvidenceLedger, canonical_digest, canonical_json
from .forge_provider import handle_request
from .provider_study import run_reference_portfolio_study

COMPAT_SCHEMA = "mnel-family-compatibility/0.1"
INTEGRATION_SCHEMA = "mnel-family-integration-report/0.1"
NORMALIZED_SCHEMA = "mnel-family-execution-evidence/0.1"
PINNED_COMMITS = {
    "mncs-forge-mcp": "7710ea606bd592e0be95957c96132e8732fbb955",
    "mncs-fabric": "740b4b3a2590f76aa6eeb5365d3bdd8a40e39964",
    "machine-native-complexity-standard": "80f08d312dce963265c7f69ac5b4bae8245bd692",
    "MNCS-Commons": "b1eb5a1081bbb63ee3a6284e8046035bd72a47bc",
    "mncs-language": "f234cc8079faa5895a38b7abce0c96031f7d2565",
    "RAVEL": "d572d68ab9c8eaf163425748d44729aaa8028e98",
}
EXPECTED_CONTRACT_VERSIONS = {
    "mncs-forge-mcp": "Provider Protocol 0.1",
    "mncs-fabric": "mncs-fabric public controller/service 0.2.0a0",
    "machine-native-complexity-standard": "execution receipt 0.1-experimental",
    "MNCS-Commons": "commons.mncs.dev/v0alpha1",
    "mncs-language": "semantic identity boundary",
    "RAVEL": "ravel-development-record/0.6-preregistration",
}
FORBIDDEN = {"verdict", "conformance", "promotion", "promotion_authorized", "evaluator_authority"}


class FamilyIntegrationError(ValueError):
    pass


def _reject_authority(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            inert = child is False or child is None or child in ("not-asserted", "UNKNOWN", "unknown") if isinstance(child, (str, bool)) or child is None else False
            if str(key).lower() in FORBIDDEN and not inert:
                raise FamilyIntegrationError(f"authority-expanding field: {key}")
            _reject_authority(child)
    elif isinstance(value, list):
        for child in value:
            _reject_authority(child)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise FamilyIntegrationError(f"{label} must be a sha256 identity")
    return value


@dataclass(frozen=True, slots=True)
class FamilyIdentityBinding:
    mnel_study_identity: str
    mnel_experiment_identity: str
    provider_artifact_identity: str
    snapshot_identity: str | None = None
    forge_request_identity: str | None = None
    forge_result_identity: str | None = None
    fabric_manifest_identity: str | None = None
    fabric_job_identity: str | None = None
    fabric_record_identity: str | None = None
    mncs_receipt_identity: str | None = None
    commons_record_identity: str | None = None
    ravel_candidate_identity: str | None = None
    language_semantic_identity: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "mnel_study_identity",
            "mnel_experiment_identity",
            "provider_artifact_identity",
        ):
            _sha(getattr(self, name), name)
        for name in (
            "snapshot_identity",
            "forge_request_identity",
            "forge_result_identity",
            "fabric_manifest_identity",
            "fabric_job_identity",
            "fabric_record_identity",
            "mncs_receipt_identity",
            "commons_record_identity",
            "ravel_candidate_identity",
            "language_semantic_identity",
        ):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise FamilyIntegrationError(f"{name} cannot be empty")

    @property
    def binding_identity(self) -> str:
        return canonical_digest(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": "mnel-family-identity-binding/0.1",
            "mnel_study_identity": self.mnel_study_identity,
            "mnel_experiment_identity": self.mnel_experiment_identity,
            "provider_artifact_identity": self.provider_artifact_identity,
            "optional_external_identities": {
                key: getattr(self, key)
                for key in (
                    "snapshot_identity",
                    "forge_request_identity",
                    "forge_result_identity",
                    "fabric_manifest_identity",
                    "fabric_job_identity",
                    "fabric_record_identity",
                    "mncs_receipt_identity",
                    "commons_record_identity",
                    "ravel_candidate_identity",
                    "language_semantic_identity",
                )
                if getattr(self, key) is not None
            },
            "authority": "diagnostic-only",
            "semantics": "cross-family identity binding; not-a-verdict",
        }
        if include_identity:
            value["binding_identity"] = self.binding_identity
        return value


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _live_forge_probe(forge_root: Path) -> dict[str, Any]:
    """Use the sibling Forge CLI in a temporary project state, never MNEL state."""

    config_source = Path(__file__).parents[2] / "mncs-forge.toml"
    if not (forge_root / "src" / "mncs_forge").is_dir() or not config_source.is_file():
        return {"status": "unavailable", "reason": "sibling Forge source or config is unavailable"}
    try:
        with tempfile.TemporaryDirectory(prefix="mnel-forge-probe-") as directory:
            root = Path(directory)
            (root / "README.md").write_text("temporary Forge control state\n", encoding="utf-8")
            config = root / "mncs-forge.toml"
            config.write_text(config_source.read_text(encoding="utf-8"), encoding="utf-8")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(forge_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
            command = [sys.executable, "-m", "mncs_forge.cli", "--config", str(config), "providers", "probe", "mnel-family-provider"]
            completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=20, check=False, shell=False)
            if len(completed.stdout.encode()) > 128 * 1024 or len(completed.stderr.encode()) > 16 * 1024:
                return {"status": "UNKNOWN", "reason": "Forge probe output exceeded adapter ceiling"}
            try:
                parsed = json.loads(completed.stdout)
            except json.JSONDecodeError:
                parsed = {"status": "UNKNOWN", "reason": "Forge probe returned malformed JSON"}
            return {"status": "available" if completed.returncode == 0 and parsed.get("status") == "PASS" else "UNKNOWN", "returncode": completed.returncode, "result": parsed, "stderr": completed.stderr[:1024]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "reason": str(error)}


def validate_compatibility_fixture(path: str | Path, sibling_roots: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
    """Validate a pinned shape snapshot and, when supplied, its local checkout heads."""

    try:
        fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FamilyIntegrationError(f"cannot read compatibility fixture: {error}") from error
    if fixture.get("schema") != COMPAT_SCHEMA:
        raise FamilyIntegrationError("unsupported family compatibility snapshot")
    projects = fixture.get("projects")
    if not isinstance(projects, list) or {item.get("repository") for item in projects} != set(PINNED_COMMITS):
        raise FamilyIntegrationError("compatibility snapshot project set drifted")
    results: list[dict[str, Any]] = []
    for item in projects:
        repository = item.get("repository")
        if item.get("commit") != PINNED_COMMITS[repository]:
            raise FamilyIntegrationError(f"pinned commit drift for {repository}")
        if not isinstance(item.get("public_contract"), dict) or item["public_contract"].get("version") != EXPECTED_CONTRACT_VERSIONS[repository]:
            raise FamilyIntegrationError(f"public contract is incomplete for {repository}")
        live = None
        status = "fixture-only"
        if sibling_roots and repository in sibling_roots:
            root = Path(sibling_roots[repository]).resolve()
            live = _git_head(root)
            status = "live-exact" if live == item["commit"] else "live-drift" if live else "unavailable"
        results.append({"repository": repository, "pinned_commit": item["commit"], "live_commit": live, "status": status, "public_contract": item["public_contract"]})
    return {
        "schema": COMPAT_SCHEMA,
        "fixture_identity": canonical_digest(fixture),
        "status": "PASS" if all(item["status"] in {"fixture-only", "live-exact"} for item in results) else "UNKNOWN",
        "projects": results,
        "limitations": ["compatibility snapshots pin public shapes; they do not prove live implementation behavior"],
        "authority": "diagnostic-only",
    }


class FamilyEvidenceAdapter:
    """Normalize Fabric observations without changing their claim boundary."""

    @staticmethod
    def normalize_fabric_execution(
        record: Mapping[str, Any],
        binding: FamilyIdentityBinding,
        receipt: Mapping[str, Any] | None = None,
        observed_provider_artifact_identity: str | None = None,
    ) -> dict[str, Any]:
        if record.get("schema_version") != "mncs-fabric.execution-record.v0.1":
            raise FamilyIntegrationError("unsupported Fabric execution-record version")
        record_identity = _sha(record.get("record_id"), "Fabric record_id")
        if binding.fabric_record_identity and binding.fabric_record_identity != record_identity:
            raise FamilyIntegrationError("Fabric record identity does not match binding")
        if binding.fabric_manifest_identity and record.get("artifact_manifest_identity") != binding.fabric_manifest_identity:
            raise FamilyIntegrationError("Fabric manifest identity does not match binding")
        if record.get("candidate_identity") != binding.mnel_study_identity:
            raise FamilyIntegrationError("Fabric candidate identity does not match MNEL study")
        if observed_provider_artifact_identity is not None and observed_provider_artifact_identity != binding.provider_artifact_identity:
            raise FamilyIntegrationError("observed provider artifact does not match binding")
        if receipt is not None:
            if receipt.get("schema_version") != "0.1-experimental" or receipt.get("record_type") != "mncs-execution-receipt":
                raise FamilyIntegrationError("unsupported MNCS receipt version")
            boundary = receipt.get("claim_boundary")
            if not isinstance(boundary, dict) or any(
                value not in {False, "not-asserted", "UNKNOWN", "unknown", None}
                for value in boundary.values()
            ):
                raise FamilyIntegrationError("receipt claim boundary expanded authority")
        value = {
            "schema": NORMALIZED_SCHEMA,
            "normalized_identity": canonical_digest({"record": record, "receipt": receipt}),
            "external": {"project": "mncs-fabric", "protocol": record["schema_version"], "record_identity": record_identity},
            "binding": binding.to_dict(),
            "execution_observation": dict(record),
            "receipt_observation": dict(receipt) if receipt is not None else None,
            "claim_boundary": "execution-observation-only",
            "limitations": [
                "process completion and receipt status do not establish provider correctness or conformance",
                "local/in-process replication is not multi-host independence or protected custody",
            ],
            "authority": "diagnostic-only",
            "semantics": "normalized external execution evidence; not-a-verdict",
        }
        return value


class ReplayGuard:
    """Reject conflicting reuse of one external record identity."""

    def __init__(self) -> None:
        self._records: dict[str, str] = {}

    def accept(self, record: Mapping[str, Any]) -> str:
        identity = str(record.get("record_id", ""))
        if not identity:
            raise FamilyIntegrationError("external record has no identity")
        digest = canonical_digest(dict(record))
        prior = self._records.get(identity)
        if prior is not None and prior != digest:
            raise FamilyIntegrationError("conflicting replay for external record")
        self._records[identity] = digest
        return identity


class CommonsInterchangeAdapter:
    """Create an inert Commons Observation-shaped record; never publish it."""

    @staticmethod
    def observation(normalized: Mapping[str, Any]) -> dict[str, Any]:
        external = normalized.get("external", {})
        record_id = str(external.get("record_identity", canonical_digest(normalized)))
        value = {
            "apiVersion": "commons.mncs.dev/v0alpha1",
            "kind": "Observation",
            "metadata": {
                "recordId": "mnel-observation-" + record_id.removeprefix("sha256:")[:32],
                "createdAt": "1970-01-01T00:00:00Z",
                "author": {"type": "system", "id": "mnel-family-adapter"},
                "labels": ["mnel", "diagnostic-only", "fabric-execution"],
            },
            "subject": {"type": "mnel-family-execution", "identity": record_id},
            "scope": {"context": {"type": "local-execution-observation"}, "limitations": ["diagnostic-only"]},
            "statement": {"summary": "Fabric execution was observed through the public local boundary."},
            "evidence": [{"id": record_id, "status": "UNKNOWN"}],
            "dependencies": [],
            "affectedContracts": [],
            "provenance": {"producer": {"type": "adapter", "id": "mnel-family-adapter"}},
            "confidence": {"level": "unreported", "rationale": "execution observation is not a correctness claim"},
            "security": {"sensitivity": "public", "executableAttachments": False, "instructionsAreUntrusted": True},
            "lifecycle": {"initialState": "proposed", "reviewWhen": []},
            "relationships": [],
            "details": {
                "outcome": "UNKNOWN",
                "trustDomain": "mnel-local",
                "claimBoundary": "execution-observation-only",
            },
            "extensions": {"mnel": {"normalized_identity": normalized.get("normalized_identity")}},
        }
        _reject_authority(value)
        return value


def _commons_bundle_smoke(observation: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    """Exercise the installed Commons application boundary without publishing."""

    try:
        from mncs_commons.application.services import CommonsApplication
        from mncs_commons.store import CommonsStore
    except ImportError as error:
        return {"status": "unavailable", "reason": str(error)}
    try:
        source = CommonsStore(workspace / "commons-store")
        source.init()
        record = CommonsApplication(source).add(observation)
        bundle_path = workspace / "commons-bundle.zip"
        created = CommonsApplication(source).create_bundle(bundle_path, roots=[record.digest])
        verified = CommonsApplication.verify_bundle(bundle_path)
        target = CommonsStore(workspace / "commons-imported")
        target.init()
        imported = CommonsApplication.import_bundle(bundle_path, target)
        return {"status": "available", "record_identity": record.digest, "create": created, "verify": verified, "import": imported, "trust": "inert-local-interchange"}
    except (OSError, ValueError, RuntimeError) as error:
        return {"status": "UNKNOWN", "reason": str(error)[:512]}


class RavelProposalAdapter:
    @staticmethod
    def proposal(binding: FamilyIdentityBinding, fixture: Mapping[str, Any]) -> dict[str, Any]:
        if fixture.get("schema") != "ravel-development-record/0.6-preregistration":
            raise FamilyIntegrationError("unsupported RAVEL development-record fixture")
        authority = fixture.get("authority", {})
        if authority.get("promotion_authorized") is not False:
            raise FamilyIntegrationError("RAVEL fixture unexpectedly grants promotion")
        candidate = fixture.get("candidate", {})
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise FamilyIntegrationError("RAVEL candidate identity is missing")
        value = {
            "schema": "mnel-ravel-proposal-context/0.6",
            "ravel_candidate_identity": candidate_id,
            "parent_identity": candidate.get("parent_identity"),
            "source_evidence_identities": [binding.mnel_study_identity, binding.provider_artifact_identity],
            "declared_scope": "candidate-context-only",
            "predicted_effects": [],
            "rollback_target": None,
            "authority": "proposal-only",
            "semantics": "RAVEL proposal context; no freeze, selection, evaluation, or promotion",
        }
        _reject_authority(value)
        value["proposal_identity"] = canonical_digest(value)
        return value


def _run_fabric(binding: FamilyIdentityBinding, workspace: Path) -> dict[str, Any]:
    try:
        from mncs_fabric.artifacts import build_manifest
        from mncs_fabric.receipts import build_execution_assurance, build_execution_receipt
        from mncs_fabric.service import FabricService
    except ImportError as error:
        return {"availability": "unavailable", "reason": str(error)}
    bundle = workspace / "fabric-bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "result.json").write_text("{}\n", encoding="utf-8")
    (bundle / "provider-artifact.json").write_text(
        canonical_json(
            {
                "schema": "mnel-provider-artifact-reference/0.1",
                "provider_artifact_identity": binding.provider_artifact_identity,
                "study_identity": binding.mnel_study_identity,
                "authority": "diagnostic-only",
            }
        ).decode("utf-8")
        + "\n",
        encoding="utf-8",
    )
    task = (
        "import json\n"
        f"json.dump({{'study_identity': {binding.mnel_study_identity!r}, 'provider_artifact_identity': {binding.provider_artifact_identity!r}}}, open('result.json', 'w', encoding='utf-8'), sort_keys=True)\n"
    )
    (bundle / "task.py").write_text(task, encoding="utf-8")
    manifest = build_manifest(bundle)
    plan = {
        "schema_version": "mncs-fabric.job-plan.v0.1",
        "job_id": "mnel-family-study",
        "candidate_identity": binding.mnel_study_identity,
        "artifact_manifest_identity": manifest["manifest_identity"],
        "argv": ["@python", "task.py"],
        "working_directory": ".",
        "timeout_seconds": 10,
        "output_limit_bytes": 16 * 1024,
        "environment": {"PYTHONHASHSEED": "0"},
        "required_capabilities": ["python"],
        "result_paths": ["result.json"],
        "network_policy": "DECLARED_OFFLINE",
    }
    service = FabricService()
    service.validate_plan(plan)
    record = service.execute_local(plan, bundle, manifest, "mnel-local", results_dir=workspace / "fabric-results", work_root=workspace)
    service.verify_record(record)
    receipt = build_execution_receipt(record, subject_family="MNEL", subject_kind="provider-study")
    assurance = build_execution_assurance(receipt)
    observed = json.loads((workspace / "fabric-results" / "result.json").read_text(encoding="utf-8")) if record.get("outcome") == "PASS" else {}
    if observed and observed.get("provider_artifact_identity") != binding.provider_artifact_identity:
        raise FamilyIntegrationError("Fabric result provider artifact identity mismatch")
    fabric_binding = replace(
        binding,
        fabric_manifest_identity=manifest["manifest_identity"],
        fabric_job_identity=record.get("job_identity"),
        fabric_record_identity=record.get("record_id"),
        mncs_receipt_identity=receipt.get("receipt_identity"),
    )
    normalized = FamilyEvidenceAdapter.normalize_fabric_execution(record, fabric_binding, receipt)
    duplicate = service.execute_local(plan, bundle, manifest, "mnel-local", results_dir=workspace / "fabric-results-duplicate", work_root=workspace)
    replication = service.reconcile([record, duplicate], require_distinct_nodes=False)
    bad_capability = dict(plan)
    bad_capability["job_id"] = "mnel-family-study-capability-mismatch"
    bad_capability["required_capabilities"] = ["capability-that-is-not-present"]
    capability_record = service.execute_local(bad_capability, bundle, manifest, "mnel-local", work_root=workspace)
    wrong_manifest = dict(plan)
    wrong_manifest["job_id"] = "mnel-family-study-manifest-mismatch"
    wrong_manifest["artifact_manifest_identity"] = "sha256:" + "f" * 64
    manifest_record = service.execute_local(wrong_manifest, bundle, manifest, "mnel-local", work_root=workspace)
    malformed_plan = dict(plan)
    malformed_plan["job_id"] = "mnel-family-study-plan-invalid"
    malformed_plan["schema_version"] = "mncs-fabric.job-plan.v0.2"
    malformed_record = service.execute_local(malformed_plan, bundle, manifest, "mnel-local", work_root=workspace)
    corrupted = dict(record)
    corrupted["record_id"] = "sha256:" + "0" * 64
    corrupted_verification = service.verify_record(corrupted)
    return {
        "availability": "available",
        "manifest": manifest,
        "plan": plan,
        "execution_record": record,
        "receipt": receipt,
        "assurance": assurance,
        "normalized": normalized,
        "duplicate_record": duplicate,
        "replication": {**replication, "scope": "local-in-process-replication", "limitations": ["same-node repetition is not independent multi-host evidence"]},
        "negative_cases": {
            "capability_mismatch": {"outcome": capability_record.get("outcome"), "reason": capability_record.get("termination_reason")},
            "wrong_manifest": {"outcome": manifest_record.get("outcome"), "reason": manifest_record.get("termination_reason")},
            "unsupported_plan_version": {"outcome": malformed_record.get("outcome"), "reason": malformed_record.get("termination_reason")},
            "corrupt_record_identity": corrupted_verification,
        },
        "provider_artifact_identity": binding.provider_artifact_identity,
    }


def run_reference_family_integration(workspace: str | Path | None = None) -> dict[str, Any]:
    """Run the dependency-aware local family integration reference study."""

    root = Path(workspace).resolve() if workspace is not None else Path(tempfile.mkdtemp(prefix="mnel-family-"))
    root.mkdir(parents=True, exist_ok=True)
    portfolio = run_reference_portfolio_study()
    report = portfolio["report"]
    provider_identity = report["provider_artifact_identities"][0]
    binding = FamilyIdentityBinding(report["study_identity"], canonical_digest({"study": report["study_identity"]}), provider_identity)
    fixture_path = Path(__file__).parents[2] / "compat" / "mncs-family-compatibility-0.1.json"
    sibling_roots = {}
    sibling_parent = Path(__file__).resolve().parents[2].parent
    for name in PINNED_COMMITS:
        candidate = sibling_parent / name
        if not candidate.is_dir():
            candidate = root.parent / name
        if candidate.is_dir():
            sibling_roots[name] = candidate
    compatibility = validate_compatibility_fixture(fixture_path, sibling_roots)
    forge_live = _live_forge_probe(sibling_roots["mncs-forge-mcp"]) if "mncs-forge-mcp" in sibling_roots else {"status": "unavailable", "reason": "sibling Forge checkout not discovered"}
    forge_request = {
        "protocol_version": "0.1",
        "type": "analysis_request",
        "request_id": "sha256:" + "1" * 64,
        "analysis": "provider_study_summary",
        "component": {"candidate_identity": report["study_identity"], "identities": {"provider_artifact": provider_identity}},
        "limits": {"timeout_seconds": 5, "output_bytes": 16 * 1024},
    }
    forge_response = handle_request(forge_request)
    fabric = _run_fabric(binding, root)
    commons = None
    ravel = None
    try:
        commons = CommonsInterchangeAdapter.observation(fabric["normalized"]) if fabric.get("normalized") else {"status": "unavailable", "reason": "Fabric evidence unavailable"}
    except KeyError:
        commons = {"status": "unavailable", "reason": "Fabric evidence unavailable"}
    commons_bundle = _commons_bundle_smoke(commons, root) if commons.get("kind") == "Observation" else {"status": "unavailable", "reason": "Commons observation was not produced"}
    ravel_path = Path(__file__).parents[2] / "compat" / "ravel-development-record-0.6.json"
    try:
        ravel = RavelProposalAdapter.proposal(binding, json.loads(ravel_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, FamilyIntegrationError) as error:
        ravel = {"status": "fixture-unavailable", "reason": str(error)}
    language = {"status": "fixture-only", "identity": "opaque-semantic-identity-fixture"}
    result = {
        "schema": INTEGRATION_SCHEMA,
        "study_identity": report["study_identity"],
        "identity_binding": binding.to_dict(),
        "compatibility": compatibility,
        "forge": {"protocol": "0.1", "provider_identity": "mnel-family-provider-protocol-v1", "response": forge_response, "live_probe": forge_live, "status": "live-forge-probe" if forge_live["status"] == "available" else "local-provider-boundary"},
        "fabric": fabric,
        "commons": commons,
        "commons_bundle": commons_bundle,
        "ravel": ravel,
        "language": language,
        "availability": {"forge_live": forge_live["status"] == "available", "fabric_live": fabric.get("availability") == "available", "commons_fixture": commons is not None, "ravel_fixture": isinstance(ravel, dict), "language_fixture": True},
        "limitations": [
            "Forge is exercised through the protocol adapter here; the sibling executable is optional and not a runtime dependency",
            "Fabric evidence is local-process execution only; no remote worker independence or protected custody is claimed",
            "Commons output is inert and not published or accepted into a trust domain",
            "RAVEL output is proposal context only",
        ],
        "authority": "diagnostic-only",
        "semantics": "family integration evidence; not-a-verdict",
    }
    _reject_authority(result)
    result["report_identity"] = canonical_digest(result)
    if workspace is not None:
        ledger = EvidenceLedger(root / "family-integration-evidence.jsonl")
        ledger.append("family-compatibility-assessment", compatibility, actor="mnel-family-adapter")
        ledger.append("forge-provider-response", forge_response, actor="mnel-family-adapter")
        if fabric.get("execution_record"):
            ledger.append("fabric-execution-record", fabric["execution_record"], actor="mnel-family-adapter")
            ledger.append("mncs-execution-receipt", fabric["receipt"], actor="mnel-family-adapter")
            ledger.append("mnel-normalized-execution-evidence", fabric["normalized"], actor="mnel-family-adapter")
        ledger.append("commons-inert-observation", commons if isinstance(commons, dict) else {"status": "unavailable"}, actor="mnel-family-adapter")
        ledger.append("commons-bundle-smoke", commons_bundle, actor="mnel-family-adapter")
        ledger.append("ravel-proposal-context", ravel if isinstance(ravel, dict) else {"status": "unavailable"}, actor="mnel-family-adapter")
        ledger.append("family-integration-report", result, actor="mnel-family-adapter")
        result["ledger"] = ledger.summarize()
    return result
