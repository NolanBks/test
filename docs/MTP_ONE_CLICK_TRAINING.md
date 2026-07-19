# MoWE 8 卡一键训练脚本

`start_mtp.py` 是面向只能提交一个 Python 入口的 MTP/云训练平台的正式编排器。它会按 fail-closed 顺序自动完成：

1. 路径与 8 GPU 数量预检。
2. RLDS/skill config、formal feature-store checksum 与 8-rank assignment 审计。
3. 100-window raw/cache 等价性审计。
4. Stage 1：`0→2→25→100→1000→最多 50000`，保持同一训练合同精确恢复。
5. Stage 1 结束后，deployment validation 必须在至少 32 个不同 episode 上证明 H=4/8/16 future predictor 平均优于 `copy_current` 至少 10%，且任一 horizon 相对回退不得超过 5%；失败时禁止进入 Stage 2。该 5% 是单尺度容差，不会替代平均改善门槛。
6. Stage 2：`0→100→最多 50000`，使用通过质量门禁的 Stage 1 checkpoint 初始化。
7. Stage 3：`0→100→最多 50000`，使用 Stage 2 checkpoint 初始化。

三个阶段默认每 500 步验证一次。feature-store 验证按 episode identity 确定性均衡取样，每个 validation episode 固定一个窗口，并同时输出 diagnostic（GT action/oracle route）与 deployment（nominal action/predicted route）记录。早停只读取 deployment `total_loss`，并且要等 nominal-action 与 Stage 3 predicted-route 调度完成（默认 50000-step 合同下最早 step 35000）后才开始累计 patience。最佳 deployment checkpoint 单独保存为 `checkpoint_best.pt`，`checkpoint_latest.pt` 继续承担精确恢复。

任何数据审计、训练/验证质量、checkpoint 或分布式训练错误都会立即停止并保留 checkpoint、报告和日志，不会自动放宽阈值。

该一键入口不读取 `/proc`、cgroup、宿主机内存、进程 RSS、OOM event 或 GPU 显存统计，也不启动 soak、资源 runtime audit 或资源 readiness。平台资源观察与配额保护由云平台自身承担，不属于训练脚本流程。

## 1. 新服务器必须准备的内容

以下路径可以与旧服务器不同，但内容必须一致：

- MoWE 仓库。
- 完整 LIBERO RLDS 数据目录。
- 已完成的 formal feature store。
- 原始 `openvla/openvla-7b` snapshot，不能使用 LIBERO-finetuned OpenVLA-OFT。
- `facebook/dinov2-small` snapshot。
- `cot_file.json` skill sidecar。

feature store 可以挂载到新的绝对路径。OpenVLA 使用不可变 revision 与权重指纹校验；DINO 的新服务器路径由重新运行的 100-window 等价性审计进行实质验证。脚本在正式启动时仅通过 PyTorch 核对 8 张 CUDA 设备可见，不执行系统资源采样。

## 2. 本次重新训练：直接按此执行

旧的 `libero_original_openvla_h16_v1` 已经受单 episode validation 和过早停止影响。本次必须创建全新的 lineage：

```text
libero_original_openvla_h16_v2
```

保留旧 `v1` 目录作为历史证据，不要删除、覆盖、移动或复制其中的 Stage 1/2/3 checkpoint 到 `v2`。开始前先把当前修正版仓库完整同步到服务器；后续 dry-run、正式启动和中断恢复必须始终使用下面完全相同的路径、`run-id` 和训练参数。

### 2.1 先执行 dry-run

先用 `--dry-run` 检查所有路径和将要执行的命令；dry-run 不要求当前机器有 8 张 GPU：

```bash
cd /home/ma-user/work/algorithm/chaoxintao_2/MoWE/MoWE

python start_mtp.py \
  --repo-root /home/ma-user/work/algorithm/chaoxintao_2/MoWE/MoWE \
  --data-root /home/ma-user/work/algorithm/chaoxintao_2/MoWE/libero_cot_rlds \
  --feature-store /home/ma-user/work/algorithm/chaoxintao_2/MoWE/mowe_store/libero_h16_formal_4090 \
  --openvla-checkpoint /home/ma-user/work/algorithm/chaoxintao_2/MoWE/openvla-7b \
  --openvla-revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --dino-checkpoint /home/ma-user/work/algorithm/chaoxintao_2/MoWE/facebook-dinov2-small \
  --skill-sidecar /home/ma-user/work/algorithm/chaoxintao_2/MoWE/libero_cot_rlds/cot_file.json \
  --run-root-dir /home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs \
  --run-id libero_original_openvla_h16_v2 \
  --dry-run
```

dry-run 返回码必须为 0。随后执行以下检查；它只读取 launcher 生成的 JSON，不采集系统资源：

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("/home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs/libero_original_openvla_h16_v2")
state = json.loads((run / "launcher_state.json").read_text())
tasks = state["tasks"]
required = {
    "feature_store_audit",
    "feature_equivalence",
    "ddp_stage1_0_2",
    "ddp_stage1_2_25",
    "ddp_stage1_25_100",
    "ddp_stage1_100_1000",
    "ddp_stage1_1000_50000",
    "stage1_quality_gate",
    "ddp_stage2_0_100",
    "ddp_stage2_100_50000",
    "ddp_stage3_0_100",
    "ddp_stage3_100_50000",
}
missing = sorted(required - set(tasks))
assert not missing, f"dry-run 缺少任务: {missing}"
assert all(tasks[name]["status"] == "dry_run" for name in required)

for stage in ("stage1", "stage2", "stage3"):
    cfg = json.loads((run / "configs" / f"{stage}.json").read_text())
    distributed = cfg["training"]["distributed"]
    assert distributed["resource_monitoring"] is False
    assert "memory_guard_fraction" not in distributed
    assert "gpu_memory_guard_fraction" not in distributed

commands = "\n".join(
    " ".join(map(str, task.get("command", []))) for task in tasks.values()
).lower()
for forbidden in (
    "soak_mowe_feature_store",
    "audit_ddp_runtime",
    "audit_long_training_readiness",
    "long-run-readiness",
    "system-monitoring",
    "cgroup",
    "memory-guard",
    "gpu-memory",
):
    assert forbidden not in commands, forbidden

print("dry-run contract OK: v2 complete Stage 1 -> Stage 2 -> Stage 3, resource telemetry disabled")
PY
```

预期最后打印：

```text
dry-run contract OK: v2 complete Stage 1 -> Stage 2 -> Stage 3, resource telemetry disabled
```

### 2.2 正式启动

上述检查通过后，删除 dry-run 命令最后一行的 `--dry-run`，其他参数保持完全相同：

```bash
cd /home/ma-user/work/algorithm/chaoxintao_2/MoWE/MoWE

python start_mtp.py \
  --repo-root /home/ma-user/work/algorithm/chaoxintao_2/MoWE/MoWE \
  --data-root /home/ma-user/work/algorithm/chaoxintao_2/MoWE/libero_cot_rlds \
  --feature-store /home/ma-user/work/algorithm/chaoxintao_2/MoWE/mowe_store/libero_h16_formal_4090 \
  --openvla-checkpoint /home/ma-user/work/algorithm/chaoxintao_2/MoWE/openvla-7b \
  --openvla-revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --dino-checkpoint /home/ma-user/work/algorithm/chaoxintao_2/MoWE/facebook-dinov2-small \
  --skill-sidecar /home/ma-user/work/algorithm/chaoxintao_2/MoWE/libero_cot_rlds/cot_file.json \
  --run-root-dir /home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs \
  --run-id libero_original_openvla_h16_v2
```

不要添加 `--disable-system-monitoring`、soak、memory guard 或 readiness 参数；当前一键入口已经固定不执行这些操作。

### 2.3 中断后恢复

如果平台中断、进程退出或任务需要重新提交，直接再次执行上面的正式启动命令，不要添加 `--dry-run`，也不要修改任何参数。脚本会从 `v2` 当前 stage 的 `checkpoint_latest.pt` 精确恢复；不要手工指定旧 `v1` checkpoint。

只有以下情况应更换为新的 `run-id`，不能在 `v2` 上恢复：

- 修改 feature store、OpenVLA revision/权重、DINO、skill sidecar 或训练配置。
- 修改 `stage*-max-steps`、optimizer、学习率、solver、seed、window 或 routing schedule。
- 希望从头重新建立另一条实验 lineage。

如果平台要求显式 Python 环境，可增加：

```bash
--python /NEW_PATH/conda/envs/mowe/bin/python
```

若增加该参数，dry-run、首次正式启动和后续恢复三次都必须使用同一个 Python 路径。

### 2.4 当前 v2 训练结果

真实 `v2` 已完成三阶段训练，launcher 的 `pipeline.status=complete`。三个阶段均在调度完成后按 episode-balanced deployment validation 合规早停：Stage 1 最新 step 为 47,500、best 为 45,000；Stage 2/3 最新 step 均为 38,000、best 均为 35,500。Stage 1 best checkpoint 相对 `copy_current` 的结果为：

```text
H=4:  -3.89%
H=8: +37.68%
H=16: +57.68%
三尺度平均: +30.49%
```

Stage 1 的 `mowe_stage1_quality_gate_v3` 已在服务器通过。Stage 3 best checkpoint 的 H=4/8/16 相对改善进一步达到约 `+1.01% / +40.48% / +58.91%`，三尺度平均约 `+33.47%`。当前训练不需要继续增加 step，也不要新建 `v3`；进入第 9 节的 OSMesa simulator 评测。

训练完成状态可检查：

```bash
python -m json.tool \
  /home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs/libero_original_openvla_h16_v2/launcher_state.json
```

最终评测必须使用：

```text
/home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs/libero_original_openvla_h16_v2/ddp8/stage3/checkpoint_best.pt
```

不要使用 `checkpoint_latest.pt` 代替 best，也不要把离线训练 loss 当作 LIBERO success rate。Stage 3 predicted router 的离线 boundary recall 仍偏低，因此先跑 one-task smoke 和四-suite 5-trial gate，不能直接跳到大规模正式评测。

## 3. 必须传入的参数

| 参数 | 含义 | 是否可换路径 |
|---|---|---|
| `--repo-root` | MoWE 仓库根目录 | 可以 |
| `--data-root` | LIBERO RLDS 数据根目录 | 可以，但数据内容必须一致 |
| `--feature-store` | `formal_training_ready=true` 的正式 feature store | 可以，但不能混用/修改 shard |
| `--openvla-checkpoint` | 原始 OpenVLA-7B snapshot | 可以，权重指纹必须一致 |
| `--openvla-revision` | 40 位 immutable HF commit | 不可改变 |
| `--dino-checkpoint` | DINOv2-small snapshot | 可以，100-window equivalence 必须通过 |
| `--skill-sidecar` | `cot_file.json` | 可以，fingerprint 必须一致 |
| `--run-root-dir` | 所有新报告、日志、checkpoint 的父目录 | 可以 |
| `--run-id` | 当前正式 lineage 的唯一名字 | 新 lineage 使用新名字；恢复时必须保持不变 |

`--skill-config` 是可选参数。不传时，脚本会在当前服务器运行 RLDS audit，并自动生成带当前路径和数据 fingerprint 的 `skill_experts_h16.json`。跨服务器时推荐不传，让脚本重新生成。

## 4. 常用可选参数

一般保持默认值：

```text
--world-size 8
--cuda-devices 0,1,2,3,4,5,6,7
--stage1-max-steps 50000
--stage2-max-steps 50000
--stage3-max-steps 50000
--validation-freq 500
--early-stop-min-delta 1e-4
--early-stop-patience 5
--early-stop-min-steps 5000
--min-validation-episodes 32
--flow-solver-steps 4
--equivalence-samples 100
--long-save-freq 500
--long-log-freq 10
```

查看全部参数及脚本内注释：

```bash
python start_mtp.py --help
```

`--early-stop-min-steps` 是绝对下限；launcher 还会自动把有效下限提高到 deployment schedule 完成 step，因此默认 50000-step 合同实际不会早于 step 35000。数据等价性容差、Stage 1 质量门槛、episode 数和 imbalance ratio 不建议放宽。

## 5. 自动恢复

脚本会读取：

```text
<run-root-dir>/<run-id>/ddp8/stage1/checkpoint_latest.pt
<run-root-dir>/<run-id>/ddp8/stage2/checkpoint_latest.pt
<run-root-dir>/<run-id>/ddp8/stage3/checkpoint_latest.pt
```

任务被中断后，使用完全相同的 `run-root-dir`、`run-id`、模型合同和训练参数重新运行同一命令。脚本将：

- 跳过已经完成的 checkpoint step。
- 从当前 stage 的 `checkpoint_latest.pt` same-stage resume。
- 检查 Stage 2/3 保存的 predecessor identity；若前一阶段 checkpoint 已改变，拒绝把旧后续阶段静默续接到新 lineage。
- 不会从单卡 checkpoint 初始化正式 Stage 1。

不要在恢复时修改 `stage*-max-steps`、solver、seed、feature store、skill config 或模型 identity。只移动服务器路径时，仍需使用相同内容的 store/model/data，并让脚本重新生成跨服务器证据。

## 6. 输出目录

默认结构：

```text
<run-root-dir>/<run-id>/
  launcher_state.json
  configs/
    stage1.json
    stage2.json
    stage3.json
  reports/
    feature_store_audit.json
    feature_equivalence_100.json
    *.log
  ddp8/
    stage1/
      checkpoint_latest.pt
      checkpoint_best.pt
      train_log.jsonl
      validation_log.jsonl
      early_stopping.json
    stage2/
    stage3/
```

`launcher_state.json` 记录每项任务的状态、命令和日志路径，适合 MTP 平台断线后检查。

## 7. 脚本会主动停止的情况

- 不是 8 张可见 GPU，或 torchrun/DDP 启动失败。
- formal store 不完整、checksum 失败或 8-rank assignment 不完整。
- raw/cache equivalence 失败。
- checkpoint stage/step、same-stage schedule 或 predecessor 不合法。
- loss/gradient 或验证日志出现 NaN/Inf。
- validation 没有覆盖至少 32 个不同 episode，或 Stage 1 deployment future 质量门禁失败。

Stage 1 的 `copy_current` 质量门禁会阻止阶段切换；route accuracy、expert improvement 和 boundary 指标继续作为 Stage 2/3 诊断证据。`early_stopping.json` 中 `reason=validation_loss_plateau` 表示调度完成后的 deployment loss 平台期，`reason=max_steps` 表示跑满 50000 步。

这些停止条件是训练合同的一部分，不应通过更换 `run-id` 或放宽参数规避。

## 8. 完成边界

脚本完成表示 Stage 1/2/3 正式训练与训练期质量总结完成。它不会把训练指标当成 LIBERO success rate。当前 `v2` 的后续评测以第 9 节为准；该节已经按目标服务器“无 sudo、无 EGL、仅 OSMesa”的环境重写，不要再复制旧 runbook 中的 EGL 命令。

## 9. v2 LIBERO OSMesa 评测指南（无 sudo、无 EGL）

本节只评测已经训练完成的 `libero_original_openvla_h16_v2`，不修改 checkpoint、训练 config 或 execution 阈值。评测采用单 GPU 加 OSMesa CPU 离屏渲染，不请求 `sudo`，不安装系统包，不使用 EGL，也不执行 `/proc`、cgroup、RSS、OOM-event 或显存监控。

评测顺序固定为：

1. 路径、checkpoint metadata 和 OSMesa Python import 预检。
2. dependency-light action queue smoke。
3. `libero_spatial/task 0/1 trial` simulator smoke。
4. 四个 suite、每任务 5 trials 小样本门槛。
5. 四个 suite、每任务 50 trials、三个 seeds 的可恢复正式评测。

任何一层失败时停在该层排查，不要通过换 checkpoint、换 backbone、改 action statistics 或增加训练 step 绕过。

### 9.1 固定路径与 OSMesa 环境

每次新建 MTP 评测任务时，先完整执行下面这段。OSMesa 变量必须在启动任何 Python、MuJoCo、PyOpenGL 或 LIBERO 进程之前设置；PyOpenGL 一旦在当前进程选定平台，不能再从 EGL 热切换到 OSMesa。

```bash
set -o pipefail

export MOWE_PYTHON=/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/mowe/bin/python
export MOWE_REPO_ROOT=/home/ma-user/work/algorithm/chaoxintao_2/MoWE/MoWE
export MOWE_RUN_ROOT=/home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs/libero_original_openvla_h16_v2
export MOWE_EVAL_CONFIG="$MOWE_RUN_ROOT/configs/stage3.json"
export MOWE_S3_BEST="$MOWE_RUN_ROOT/ddp8/stage3/checkpoint_best.pt"
export MOWE_OPENVLA_SNAPSHOT=/home/ma-user/work/algorithm/chaoxintao_2/MoWE/openvla-7b
export MOWE_OPENVLA_REVISION=47a0ec7fc4ec123775a391911046cf33cf9ed83f
export MOWE_EVAL_ROOT="$MOWE_RUN_ROOT/evaluation/osmesa_stage3_best_step35500"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1

cd "$MOWE_REPO_ROOT"
mkdir -p "$MOWE_EVAL_ROOT"
```

不要设置 `MUJOCO_GL=egl` 或 `PYOPENGL_PLATFORM=egl`，不要运行 `sudo`、`apt`、`yum` 或系统库安装命令。若 MTP 为每次提交创建全新 shell，必须在每个提交命令开头重新设置以上变量。

### 9.2 只读预检

下面的检查只读取路径、metadata 和 Python 依赖，不启动 rollout：

```bash
"$MOWE_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

assert os.environ["MUJOCO_GL"] == "osmesa"
assert os.environ["PYOPENGL_PLATFORM"] == "osmesa"

required_files = {
    "evaluation config": Path(os.environ["MOWE_EVAL_CONFIG"]),
    "Stage 3 best checkpoint": Path(os.environ["MOWE_S3_BEST"]),
    "Stage 3 metadata": Path(os.environ["MOWE_S3_BEST"] + ".metadata.json"),
}
for label, path in required_files.items():
    assert path.is_file(), f"{label} 不存在: {path}"

snapshot = Path(os.environ["MOWE_OPENVLA_SNAPSHOT"])
assert snapshot.is_dir(), f"OpenVLA snapshot 不存在: {snapshot}"

metadata = json.loads(required_files["Stage 3 metadata"].read_text())
assert metadata["stage"] == "joint", metadata.get("stage")
assert metadata["step"] == 35500, metadata.get("step")

import OpenGL
import mujoco
from libero.libero import benchmark

print(
    {
        "osmesa_env": True,
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "libero_suites": sorted(benchmark.get_benchmark_dict()),
        "checkpoint_stage": metadata["stage"],
        "checkpoint_step": metadata["step"],
        "backbone_identifier": metadata["backbone_identifier"],
    }
)
PY

"$MOWE_PYTHON" scripts/eval_libero_temporal_skill.py --queue-smoke \
  > "$MOWE_EVAL_ROOT/variable_prefix_queue_smoke.json"

"$MOWE_PYTHON" -m json.tool \
  "$MOWE_EVAL_ROOT/variable_prefix_queue_smoke.json" >/dev/null
```

预检必须确认 `checkpoint_stage=joint`、`checkpoint_step=35500`，且 OSMesa/MuJoCo/LIBERO import 无异常。若错误仍提到 EGL，说明启动命令或平台环境残留了 EGL 变量；应新建干净的 MTP 任务并从 9.1 重新开始，而不是在已导入 PyOpenGL 的 Python 进程里修改变量。

### 9.3 one-task / 1-trial simulator smoke

先只运行 `libero_spatial` 的 task 0、trial 0。`--resume-results` 在文件不存在时会正常从头运行，在相同 checkpoint/seed 的文件已存在时会跳过已完成 episode，因此中断后可以原命令重跑。

```bash
set -o pipefail

"$MOWE_PYTHON" scripts/eval_libero_temporal_skill.py \
  --config "$MOWE_EVAL_CONFIG" \
  --policy-checkpoint "$MOWE_S3_BEST" \
  --backbone-checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --simulator \
  --task-suite libero_spatial \
  --task-id 0 \
  --trials 1 \
  --flow-seed 7 \
  --seed 7 \
  --output-jsonl "$MOWE_EVAL_ROOT/libero_spatial_task0_smoke.jsonl" \
  --summary-output "$MOWE_EVAL_ROOT/libero_spatial_task0_smoke_summary.json" \
  --resume-results \
  2>&1 | tee -a "$MOWE_EVAL_ROOT/libero_spatial_task0_smoke.log"
```

完成后执行合同检查：

```bash
"$MOWE_PYTHON" - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["MOWE_EVAL_ROOT"])
summary = json.loads((root / "libero_spatial_task0_smoke_summary.json").read_text())
episodes = [
    json.loads(line)
    for line in (root / "libero_spatial_task0_smoke.jsonl").read_text().splitlines()
    if line.strip()
]

assert summary["complete"] is True, summary
assert summary["episodes"] == summary["expected_episodes"] == 1, summary
assert summary["checkpoint_stage"] == "joint", summary
assert summary["checkpoint_step"] == 35500, summary
assert len(episodes) == 1, len(episodes)
episode = episodes[0]
assert episode["teacher_loaded"] is False
assert episode["video_saved"] is False
assert episode["view_order"] == ["primary", "wrist"]
assert episode["prefix_lengths"], "没有 policy query 记录"
assert all(1 <= int(value) <= 8 for value in episode["prefix_lengths"])
for weights in episode["current_view_weights"]:
    assert len(weights) == 2
    assert all(math.isfinite(float(value)) for value in weights)
    assert abs(sum(weights) - 1.0) <= 1e-4

print(
    {
        "smoke_contract_passed": True,
        "success_is_diagnostic_only": episode["success"],
        "actions_executed": episode["actions_executed"],
        "policy_queries": episode["policy_queries"],
        "prefix_lengths": episode["prefix_lengths"],
        "execution_reason_codes": episode["execution_reason_codes"],
    }
)
PY
```

单个 trial 成功或失败都不能作为模型性能结论。此层只要求：环境 reset/step 不崩溃、输出无 NaN/Inf、双视角权重合法、action queue 有记录、prefix 长度在 1～8、teacher 未加载、没有视频依赖。若出现全零动作、gripper 符号明显相反、相机方向错误或几乎每次 query 都立即 high-risk 停止，先停在 smoke 排查。

### 9.4 四 suite、每任务 5 trials 小样本门槛

smoke 合同通过后，按同一 checkpoint 和 seed 运行四个 suite。每个 suite 都使用独立 JSONL；原命令可恢复：

```bash
set -o pipefail

for suite in libero_spatial libero_object libero_goal libero_10; do
  "$MOWE_PYTHON" scripts/eval_libero_temporal_skill.py \
    --config "$MOWE_EVAL_CONFIG" \
    --policy-checkpoint "$MOWE_S3_BEST" \
    --backbone-checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
    --backbone-revision "$MOWE_OPENVLA_REVISION" \
    --simulator \
    --task-suite "$suite" \
    --all-tasks \
    --trials 5 \
    --flow-seed 7 \
    --seed 7 \
    --output-jsonl "$MOWE_EVAL_ROOT/${suite}_seed7_5trial.jsonl" \
    --summary-output "$MOWE_EVAL_ROOT/${suite}_seed7_5trial_summary.json" \
    --resume-results \
    2>&1 | tee -a "$MOWE_EVAL_ROOT/${suite}_seed7_5trial.log"
done
```

汇总检查：

```bash
"$MOWE_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MOWE_EVAL_ROOT"])
suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
total_episodes = 0
total_successes = 0
for suite in suites:
    path = root / f"{suite}_seed7_5trial_summary.json"
    summary = json.loads(path.read_text())
    assert summary["complete"] is True, (suite, summary)
    assert summary["checkpoint_stage"] == "joint", (suite, summary)
    assert summary["checkpoint_step"] == 35500, (suite, summary)
    total_episodes += int(summary["episodes"])
    total_successes += int(summary["successes"])
    print(suite, summary["successes"], summary["episodes"], summary["success_rate"])

print("overall", total_successes, total_episodes, total_successes / max(total_episodes, 1))
assert total_successes > 0, "四 suite 小样本 success 全零：停止正式评测并先排查 rollout 合同"
PY
```

进入正式评测前还要抽查四份 JSONL：执行长度不能几乎全部等于 suite 上限，prefix 不能几乎全部为 1～2 步，`execution_reason_codes` 不能几乎全部为 high-risk。若 success 全零，优先检查 action q01/q99、gripper sign、primary/wrist orientation、OpenVLA identity 和 OSMesa observation；不要先增加训练 step。

### 9.5 四 suite、50 trials、三个 seeds 正式评测

只有 5-trial gate 通过后才执行。正式主结果固定 seeds 为 `7/17/27`，每个 suite、每任务 50 个官方 initial states。每份 JSONL 独立绑定 checkpoint、suite、seed 和 flow seed；中断后重新执行整个循环会跳过已完成的 `(task_id, trial)`。

```bash
set -o pipefail

for eval_seed in 7 17 27; do
  for suite in libero_spatial libero_object libero_goal libero_10; do
    "$MOWE_PYTHON" scripts/eval_libero_temporal_skill.py \
      --config "$MOWE_EVAL_CONFIG" \
      --policy-checkpoint "$MOWE_S3_BEST" \
      --backbone-checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
      --backbone-revision "$MOWE_OPENVLA_REVISION" \
      --simulator \
      --task-suite "$suite" \
      --all-tasks \
      --trials 50 \
      --flow-seed "$eval_seed" \
      --seed "$eval_seed" \
      --output-jsonl "$MOWE_EVAL_ROOT/${suite}_seed${eval_seed}_50trial.jsonl" \
      --summary-output "$MOWE_EVAL_ROOT/${suite}_seed${eval_seed}_50trial_summary.json" \
      --resume-results \
      2>&1 | tee -a "$MOWE_EVAL_ROOT/${suite}_seed${eval_seed}_50trial.log"
  done
done
```

不要把 5-trial JSONL 改名后当作正式结果，也不要把不同 seeds 追加到同一个文件。`--resume-results` 会拒绝 checkpoint、suite、seed 或 flow seed 不一致的旧文件。

### 9.6 正式结果汇总

全部 12 个 suite/seed 任务完成后运行：

```bash
"$MOWE_PYTHON" - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["MOWE_EVAL_ROOT"])
suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
seeds = (7, 17, 27)
aggregate = defaultdict(lambda: {"episodes": 0, "successes": 0})

for seed in seeds:
    for suite in suites:
        path = root / f"{suite}_seed{seed}_50trial_summary.json"
        summary = json.loads(path.read_text())
        assert summary["complete"] is True, (suite, seed, summary)
        assert summary["checkpoint_stage"] == "joint", (suite, seed)
        assert summary["checkpoint_step"] == 35500, (suite, seed)
        aggregate[suite]["episodes"] += int(summary["episodes"])
        aggregate[suite]["successes"] += int(summary["successes"])

total_episodes = 0
total_successes = 0
for suite in suites:
    item = aggregate[suite]
    rate = item["successes"] / max(item["episodes"], 1)
    print(suite, item["successes"], item["episodes"], rate)
    total_episodes += item["episodes"]
    total_successes += item["successes"]

print("overall", total_successes, total_episodes, total_successes / max(total_episodes, 1))
PY
```

正式报告必须同时保存 12 份 episode JSONL、12 份 summary JSON 和对应日志。只有 `complete=true` 的 summary 可以进入结果表；离线 validation、one-trial smoke 和 5-trial gate 都不能替代正式 success rate。

### 9.7 OSMesa 环境的失败处理

- `ImportError` 或错误包含 `libOSMesa`：当前 Python 环境实际无法找到 OSMesa 用户态库。不要运行 sudo/apt；保留完整错误，由平台挂载已有 OSMesa 库或切换到已包含 OSMesa 的用户级环境后，从 9.1 新建任务重试。
- 错误仍包含 `EGL`：当前任务在设置变量前已经导入过 PyOpenGL/MuJoCo，或平台注入了 EGL 变量。新建干净任务，确保命令第一段就是 9.1。
- `Policy checkpoint ... missing`：模型权重仍在服务器训练目录；仅复制 `.metadata.json` 不够，必须保留真实 `checkpoint_best.pt`。
- backbone identity 不一致：必须继续使用本文固定的原始 OpenVLA snapshot 和 revision，不能换成 OFT/LIBERO-finetuned 权重。
- JSONL 已存在但合同不一致：不要删除或覆盖旧证据；换一个明确的新评测目录。相同合同的中断恢复才使用 `--resume-results`。
- simulator 全零但无 crash：先检查 action scale、gripper sign、双视角方向、等待步数和 high-risk prefix 分布；不要先重训或扩大训练步数。
