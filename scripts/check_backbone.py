#!/usr/bin/env python3
"""Inspect the upstream OpenVLA-OFT checkout without running heavy evals."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="external/openvla-oft")
    args = parser.parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        raise SystemExit(f"Missing upstream repo: {repo}")

    required = ["README.md", "SETUP.md", "LIBERO.md"]
    missing = [name for name in required if not (repo / name).exists()]
    if missing:
        raise SystemExit(f"Missing upstream docs: {missing}")

    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    print(f"OpenVLA-OFT repo: {repo}")
    print(f"OpenVLA-OFT commit: {commit}")
    print("Found docs: README.md, SETUP.md, LIBERO.md")
    print("Smallest documented upstream LIBERO smoke template:")
    print(
        "cd external/openvla-oft && "
        "python experiments/robot/libero/run_libero_eval.py "
        "--pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial "
        "--task_suite_name libero_spatial --num_trials_per_task 1"
    )
    print("Not run locally: requires OpenVLA-OFT environment, LIBERO sim, GPU, and checkpoint download.")


if __name__ == "__main__":
    main()
