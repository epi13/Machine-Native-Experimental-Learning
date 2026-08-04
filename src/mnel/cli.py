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
from .investigators import DEFAULT_ROLE_CONTRACTS


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mnel", description="Machine-Native Experimental Learning")
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--workspace", default=".")
    investigator = commands.add_parser("investigator")
    investigator.add_subparsers(dest="subcommand", required=True).add_parser("list")
    ledger = commands.add_parser("ledger")
    ledger_commands = ledger.add_subparsers(dest="subcommand", required=True)
    for name in ("verify", "summarize"):
        command = ledger_commands.add_parser(name)
        command.add_argument("path")
    demo = commands.add_parser("demo")
    demo.add_argument("--workspace", default="build/demo")
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
            "forge": "provider adapter; no local executable assumed",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["python_supported"] else 1
    if args.command == "investigator":
        print(json.dumps([item.to_dict() for item in DEFAULT_ROLE_CONTRACTS], indent=2))
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
    print(json.dumps(run_reference_study(args.workspace), indent=2, sort_keys=True))
    return 0
