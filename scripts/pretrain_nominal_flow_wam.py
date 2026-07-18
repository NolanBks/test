#!/usr/bin/env python3
"""Stage 1: pretrain nominal 6D flow, gripper head, memory, and latent WAM."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training.flow_cli import run_flow_stage


if __name__ == "__main__":
    run_flow_stage("nominal_flow_pretrain", "configs/mowe_wam/train_nominal_flow_wam.yaml")
