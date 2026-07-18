#!/usr/bin/env python3
"""Audit a committed MoWE LeRobot-v3-style canonical archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.data import audit_canonical_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_canonical_archive(
        args.archive, verify_checksums=args.verify_checksums
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
