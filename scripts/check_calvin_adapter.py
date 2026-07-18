#!/usr/bin/env python3
"""Dependency-light smoke for CALVIN action and goal-boundary adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.benchmarks.calvin import CalvinActionAdapter
from mowe_wam.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/calvin_abc_d.yaml")
    parser.add_argument(
        "--synthetic-contract",
        action="store_true",
        help="Use a reversible synthetic action range without claiming CALVIN runtime readiness.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    action_cfg = dict(cfg["action"])
    if args.synthetic_contract:
        action_cfg.update(
            {
                "motion_q01": [-1.0] * 6,
                "motion_q99": [1.0] * 6,
                "gripper_open_value": -1.0,
                "gripper_closed_value": 1.0,
            }
        )
    adapter = CalvinActionAdapter.from_config(action_cfg)
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("NumPy is required for this smoke.") from exc
    raw = np.asarray(
        [[0.0] * 6 + [adapter.gripper_open_value], [0.1] * 6 + [adapter.gripper_closed_value]],
        dtype=np.float32,
    )
    shared = adapter.to_shared_action(raw)
    restored = adapter.from_shared_action(shared)
    maximum_error = float(np.abs(restored - raw).max())
    report = {
        "kind": "calvin_adapter_contract_smoke_not_benchmark",
        "benchmark": cfg["benchmark"],
        "action_contract": adapter.contract(),
        "roundtrip_max_abs_error": maximum_error,
        "passed": maximum_error <= 1e-6,
        "official_calvin_imported": any(name.startswith("calvin_") for name in sys.modules),
        "note": "This does not install CALVIN, run its simulator, or produce benchmark metrics.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
