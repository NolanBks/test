#!/usr/bin/env python3
"""Audit the six-motor-plus-null leading-verb skill sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.data import ExpertSkillSidecar
from mowe_wam.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Accepted for parity with dataset preflight; not read by this audit.")
    parser.add_argument("--skill-config", default="configs/mowe_wam/skill_experts.yaml")
    parser.add_argument("--sidecar")
    parser.add_argument("--limit", type=int, help="Reserved for CLI compatibility; audit always covers the file.")
    args = parser.parse_args()
    cfg = load_config(args.skill_config)
    sidecar = ExpertSkillSidecar(args.sidecar or cfg["source_path"])
    report = sidecar.audit()
    report["configured_skills"] = cfg["skills"]
    report["data_root"] = args.data_root
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
