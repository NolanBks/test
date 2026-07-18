#!/usr/bin/env python3
"""Print a resolved MoWE-WAM config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.utils.config import dump_resolved_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(dump_resolved_config(cfg))
    router = cfg.get("router", {})
    if router:
        print(f"Router inputs: {router.get('inputs', 'TBD')}")
        print(f"Oracle labels: {router.get('use_oracle_predicates', False)}")
        if router.get("use_oracle_predicates", False):
            print("WARNING: oracle-predicate mode is analysis-only and non-deployable.")


if __name__ == "__main__":
    main()
