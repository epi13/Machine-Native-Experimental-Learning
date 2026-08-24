#!/usr/bin/env python3
"""Assert that the MNCS compiler rejects MNEL negative fixtures.

Each negative fixture encodes an MNEL authority/evidence boundary that must
fail closed. The expected outcome is a compiler error diagnostic, never a
running program.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_DIR = REPO_ROOT / "mncs" / "source" / "negative"

# fixture stem -> diagnostic codes that must appear among the errors
EXPECTED_ERRORS = {
    "authority-expansion": ["MNE134"],
    "authority-type-mismatch": ["MNE134"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mncs_bin", help="path to the mns CLI binary")
    args = parser.parse_args()

    failures = []
    for stem, expected_codes in EXPECTED_ERRORS.items():
        source = NEGATIVE_DIR / f"{stem}.mncs"
        if not source.exists():
            failures.append(f"{stem}: fixture missing")
            continue
        completed = subprocess.run(
            [args.mncs_bin, "source-study", str(source), "--node-id", f"negative-{stem}"],
            capture_output=True,
            text=True,
        )
        try:
            payload = json.loads(completed.stdout[completed.stdout.find("{"):])
        except (ValueError, TypeError):
            failures.append(f"{stem}: source-study did not emit JSON (rc={completed.returncode})")
            continue
        error_codes = sorted({
            d.get("code") for d in payload.get("diagnostics", [])
            if d.get("severity") == "error"
        })
        missing = [code for code in expected_codes if code not in error_codes]
        if missing:
            failures.append(
                f"{stem}: expected errors {expected_codes}, got {error_codes}"
            )
        else:
            print(f"  rejected {stem}: {error_codes}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print("all negative fixtures rejected as required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
