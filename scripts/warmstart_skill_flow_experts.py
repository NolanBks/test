#!/usr/bin/env python3
"""Stage 2: oracle per-timestep warm-start of temporal motor flow experts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training.flow_cli import run_flow_stage


if __name__ == "__main__":
    run_flow_stage("expert_warmstart", "configs/mowe_wam/warmstart_skill_flow_experts.yaml")
