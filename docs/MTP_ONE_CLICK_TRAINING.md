# MoWE 8 卡一键训练脚本

`start_mtp.py` 是面向只能提交一个 Python 入口的 MTP/云训练平台的正式编排器。它会按 fail-closed 顺序自动完成：

1. 路径、8 GPU、cgroup v2 预检。
2. RLDS/skill config、formal feature-store checksum 与 8-rank assignment 审计。
3. 100-window raw/cache 等价性审计。
4. 同一节点、同一 boot、同一 cgroup 的 8-rank feature-store soak 和 8-GPU runtime audit。
5. Stage 1：`0→2→25→100→1000→50000`，在 step 100/1000 重新签发 checkpoint-bound readiness，并在 step 1000 检查 future predictor 相对 copy-current 的质量门槛。
6. Stage 2：`0→100→50000`，使用 Stage 1 最终 checkpoint 初始化，签发 Stage 2 readiness，完成后检查 expert/route promotion gate。
7. Stage 3：`0→100→30000`，使用 Stage 2 最终 checkpoint 初始化并签发 Stage 3 readiness。

任何审计、readiness、资源门槛或质量门槛失败时，脚本立即停止并保留 checkpoint、报告和日志，不会自动放宽阈值。

## 1. 新服务器必须准备的内容

以下路径可以与旧服务器不同，但内容必须一致：

- MoWE 仓库。
- 完整 LIBERO RLDS 数据目录。
- 已完成的 formal feature store。
- 原始 `openvla/openvla-7b` snapshot，不能使用 LIBERO-finetuned OpenVLA-OFT。
- `facebook/dinov2-small` snapshot。
- `cot_file.json` skill sidecar。

feature store 可以挂载到新的绝对路径。OpenVLA 使用不可变 revision 与权重指纹校验；DINO 的新服务器路径由重新运行的 100-window 等价性审计进行实质验证。旧服务器生成的 equivalence、soak、runtime/readiness 报告不能直接跨路径/节点复用。

正式运行前检查：

```bash
nvidia-smi -L
python -c 'import torch; print(torch.cuda.device_count())'
test -f /sys/fs/cgroup/cgroup.controllers
test -f /sys/fs/cgroup/memory.current
test -f /sys/fs/cgroup/memory.max
test -f /sys/fs/cgroup/memory.events
```

GPU 数必须是 8，cgroup v2 文件必须存在。

如果平台不向容器开放 `/proc` 或 cgroup、且无法取得该权限，可在启动命令追加：

```bash
--disable-system-monitoring
```

该显式降级模式不会读取 cgroup、`/proc` boot/cgroup identity 或宿主机内存/OOM 指标；仍强制检查 8 卡 CUDA、NCCL rank/GPU 绑定、GPU 显存、数据完整性、checkpoint 合同和阶段质量门槛。报告会标为降级资源证据，不能与带 cgroup 监控的正式节点证据混用。

## 2. 推荐启动命令

先用 `--dry-run` 检查所有路径和将要执行的命令；dry-run 不要求当前机器有 8 张 GPU：

```bash
cd /NEW_PATH/MoWE

python start_mtp.py \
  --repo-root /NEW_PATH/MoWE \
  --data-root /NEW_PATH/libero_cot_rlds \
  --feature-store /NEW_PATH/mowe_store/libero_h16_formal \
  --openvla-checkpoint /NEW_PATH/models/openvla-7b \
  --openvla-revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --dino-checkpoint /NEW_PATH/models/facebook-dinov2-small \
  --skill-sidecar /NEW_PATH/libero_cot_rlds/cot_file.json \
  --run-root-dir /NEW_PATH/outputs \
  --run-id libero_original_openvla_h16_v1 \
  --dry-run
```

确认输出正确后，删除最后一行的 `--dry-run`，使用相同参数正式启动：

```bash
python start_mtp.py \
  --repo-root /NEW_PATH/MoWE \
  --data-root /NEW_PATH/libero_cot_rlds \
  --feature-store /NEW_PATH/mowe_store/libero_h16_formal \
  --openvla-checkpoint /NEW_PATH/models/openvla-7b \
  --openvla-revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --dino-checkpoint /NEW_PATH/models/facebook-dinov2-small \
  --skill-sidecar /NEW_PATH/libero_cot_rlds/cot_file.json \
  --run-root-dir /NEW_PATH/outputs \
  --run-id libero_original_openvla_h16_v1
```

如果平台要求显式 Python 环境，可增加：

```bash
--python /NEW_PATH/conda/envs/mowe/bin/python
```

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
--stage3-max-steps 30000
--stage1-pilot-step 1000
--flow-solver-steps 4
--equivalence-samples 100
--soak-steps 10000
--long-save-freq 500
--long-log-freq 10
--disable-system-monitoring  # 仅当平台没有系统资源指标访问权限时加入
```

查看全部参数及脚本内注释：

```bash
python start_mtp.py --help
```

不建议放宽以下参数：feature/output/loss 容差、imbalance ratio、内存增长/斜率、GPU/cgroup guard、Stage 1/2 promotion gate。若真实运行失败，应保留报告并分析原因。

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
- 在需要继续长训练时，针对当前最新 checkpoint 重新签发 readiness。
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
    feature_store_soak_8rank.json
    ddp_runtime_8gpu.json
    readiness_stage*_step*.json
    stage1_quality_gate.json
    stage2_quality_gate.json
    stage3_quality_summary.json
    *.log
  ddp8/
    stage1/
      checkpoint_latest.pt
      train_log.jsonl
      validation_log.jsonl
    stage2/
    stage3/
```

`launcher_state.json` 记录每项任务的状态、命令和日志路径，适合 MTP 平台断线后检查。

## 7. 脚本会主动停止的情况

- 不是 8 张可见 GPU，或 rank/GPU 绑定不完整。
- 缺少 cgroup v2 指标。
- formal store 不完整、checksum 失败或 8-rank assignment 不完整。
- raw/cache equivalence 失败。
- soak 出现 OOM/OOM-kill 或内存增长/斜率超标。
- readiness 与 store/config/checkpoint/node identity 不匹配。
- checkpoint stage/step、same-stage schedule 或 predecessor 不合法。
- loss/gradient 出现 NaN/Inf，null residual 非零，residual norm 超过 0.5。
- Stage 1 step 1000 未达到 copy-current promotion gate。
- Stage 2 长训练未达到 route/expert promotion gate，因此不会启动 Stage 3。

这些停止条件是训练合同的一部分，不应通过更换 `run-id` 或放宽参数规避。

## 8. 完成边界

脚本完成表示 Stage 1/2/3 正式训练与训练期质量总结完成。它不会把训练指标当成 LIBERO success rate。最终 checkpoint 仍需按 `docs/CLOUD_TRAINING_RUNBOOK.md` 的 LIBERO simulator smoke 和四-suite 正式评测执行。
