#!/usr/bin/env python3
"""Stage 3: oracle-to-ST-Gumbel temporal skill routing joint training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training.flow_cli import run_flow_stage


if __name__ == "__main__":
    run_flow_stage("joint", "configs/mowe_wam/train_flow_wam_skill_moe.yaml")
