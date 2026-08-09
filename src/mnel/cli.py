"""MNEL command-line interface."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

from . import __version__
from .core import EvidenceLedger, run_reference_study
from .distillation import run_reference_distill_study
from .forge_lifecycle import run_reference_forge_study
from .provider_study import run_reference_portfolio_study
from .family_integration import run_reference_family_integration
from .fabric_execution import run_network_fabric, run_reference_fabric_study
from .investigators import DEFAULT_ROLE_CONTRACTS
from .learned_providers import (
    DEFAULT_LEARNED_PROVIDER_REGISTRY,
    CostClass,
    LearnedProviderQuery,
    OutputKind,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mnel", description="Machine-Native Experimental Learning")
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--workspace", default=".")
    investigator = commands.add_parser("investigator")
    investigator.add_subparsers(dest="subcommand", required=True).add_parser("list")

    learned_provider = commands.add_parser(
        "learned-provider",
        description="Inspect diagnostic-only learned micro-provider declarations",
    )
    learned_commands = learned_provider.add_subparsers(dest="subcommand", required=True)
    learned_commands.add_parser("list")
    learned_describe = learned_commands.add_parser("describe")
    learned_describe.add_argument("provider_id")
    learned_match = learned_commands.add_parser("match")
    learned_match.add_argument("--uncertainty", action="append", required=True)
    learned_match.add_argument("--artifact", action="append", default=[])
    learned_match.add_argument("--snapshot", action="append", default=[])
    learned_match.add_argument(
        "--output-kind", action="append", default=[], choices=[item.value for item in OutputKind]
    )
    learned_match.add_argument("--prefer-family", action="append", default=[])
    learned_match.add_argument("--exclude", action="append", default=[])
    learned_match.add_argument(
        "--max-cost", choices=[item.value for item in CostClass], default=CostClass.HIGH.value
    )
    learned_match.add_argument("--limit", type=int, default=10)
    learned_match.add_argument("--diverse", action="store_true")

    ledger = commands.add_parser("ledger")
    ledger_commands = ledger.add_subparsers(dest="subcommand", required=True)
    for name in ("verify", "summarize"):
        command = ledger_commands.add_parser(name)
        command.add_argument("path")
    demo = commands.add_parser("demo")
    demo.add_argument("--workspace", default="build/demo")
    forge_reference = commands.add_parser(
        "forge-reference", description="Run the deterministic MNEL Forge diagnostic lifecycle"
    )
    forge_reference.add_argument("--workspace", default=None)
    distill_reference = commands.add_parser(
        "distill-reference", description="Run the deterministic MNEL distillation study"
    )
    distill_reference.add_argument("--workspace", default=None)
    provider_study_reference = commands.add_parser(
        "provider-study-reference", description="Run the deterministic heterogeneous provider study"
    )
    provider_study_reference.add_argument("--workspace", default=None)
    family_integration_reference = commands.add_parser(
        "family-integration-reference", description="Run the bounded MNCS-family integration study"
    )
    family_integration_reference.add_argument("--workspace", default=None)
    fabric_reference = commands.add_parser(
        "fabric-reference",
        description="Run the deterministic distributed MNEL/Fabric reference study",
    )
    fabric_reference.add_argument("--workspace", default=None)
    fabric_run = commands.add_parser(
        "fabric-run",
        description="Dispatch an operator-supplied fixed-argv plan through remote Fabric",
    )
    fabric_run.add_argument("--config", required=True)
    fabric_run.add_argument("--plan", required=True)
    fabric_run.add_argument("--manifest", required=True)
    fabric_run.add_argument("--replicas", type=int, default=1)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "doctor":
        result = {
            "mnel_version": __version__,
            "python": platform.python_version(),
            "python_supported": sys.version_info >= (3, 11),
            "workspace": str(Path(args.workspace).resolve()),
            "commands": {"elh": shutil.which("elh"), "mncs-fabric": shutil.which("mncs-fabric")},
            "forge": "MNCS Provider Protocol 0.1 adapter; external Forge remains optional",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["python_supported"] else 1
    if args.command == "investigator":
        print(json.dumps([item.to_dict() for item in DEFAULT_ROLE_CONTRACTS], indent=2))
        return 0
    if args.command == "learned-provider":
        if args.subcommand == "list":
            payload = [item.to_dict() for item in DEFAULT_LEARNED_PROVIDER_REGISTRY.list()]
        elif args.subcommand == "describe":
            try:
                payload = DEFAULT_LEARNED_PROVIDER_REGISTRY.describe(args.provider_id).to_dict()
            except KeyError as error:
                print(json.dumps({"error": str(error)}, indent=2), file=sys.stderr)
                return 2
        else:
            if args.limit < 1:
                print(json.dumps({"error": "--limit must be positive"}, indent=2), file=sys.stderr)
                return 2
            query = LearnedProviderQuery(
                uncertainty_classes=tuple(args.uncertainty),
                artifact_types=tuple(args.artifact),
                available_snapshot_types=tuple(args.snapshot),
                required_output_kinds=tuple(OutputKind(value) for value in args.output_kind),
                preferred_architecture_families=tuple(args.prefer_family),
                excluded_provider_ids=tuple(args.exclude),
                max_cost=CostClass(args.max_cost),
            )
            matches = (
                DEFAULT_LEARNED_PROVIDER_REGISTRY.select_diverse(query, args.limit)
                if args.diverse
                else DEFAULT_LEARNED_PROVIDER_REGISTRY.match(query)[: args.limit]
            )
            payload = [item.to_dict() for item in matches]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "ledger":
        ledger = EvidenceLedger(args.path)
        result = ledger.verify()
        payload = (
            ledger.summarize()
            if args.subcommand == "summarize"
            else {
                "valid": result.valid,
                "record_count": result.record_count,
                "head_digest": result.head_digest,
                "errors": list(result.errors),
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if result.valid else 1
    if args.command == "forge-reference":
        print(json.dumps(run_reference_forge_study(args.workspace), indent=2, sort_keys=True))
        return 0
    if args.command == "distill-reference":
        print(json.dumps(run_reference_distill_study(args.workspace), indent=2, sort_keys=True))
        return 0
    if args.command == "provider-study-reference":
        print(json.dumps(run_reference_portfolio_study(args.workspace), indent=2, sort_keys=True))
        return 0
    if args.command == "family-integration-reference":
        print(
            json.dumps(run_reference_family_integration(args.workspace), indent=2, sort_keys=True)
        )
        return 0
    if args.command == "fabric-reference":
        print(json.dumps(run_reference_fabric_study(args.workspace), indent=2, sort_keys=True))
        return 0
    if args.command == "fabric-run":
        try:
            result = run_network_fabric(
                args.config, args.plan, args.manifest, replicas=args.replicas
            )
        except Exception as error:
            print(
                json.dumps({"status": "UNKNOWN", "error": str(error)}, indent=2, sort_keys=True),
                file=sys.stderr,
            )
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(json.dumps(run_reference_study(args.workspace), indent=2, sort_keys=True))
    return 0
