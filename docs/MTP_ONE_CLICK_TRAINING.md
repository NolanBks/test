# MoWE 8 卡一键训练脚本

`start_mtp.py` 是面向只能提交一个 Python 入口的 MTP/云训练平台的正式编排器。它会按 fail-closed 顺序自动完成：

1. 路径与 8 GPU 数量预检。
2. RLDS/skill config、formal feature-store checksum 与 8-rank assignment 审计。
3. 100-window raw/cache 等价性审计。
4. Stage 1：`0→2→25→100→1000→最多 50000`，保持同一训练合同精确恢复。
5. Stage 1 结束后，deployment validation 必须在至少 32 个不同 episode 上证明 H=4/8/16 future predictor 平均优于 `copy_current` 至少 10%，且任一 horizon 不得更差；失败时禁止进入 Stage 2。
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

脚本完成表示 Stage 1/2/3 正式训练与训练期质量总结完成。它不会把训练指标当成 LIBERO success rate。最终 checkpoint 仍需按 `docs/CLOUD_TRAINING_RUNBOOK.md` 的 LIBERO simulator smoke 和四-suite 正式评测执行。
