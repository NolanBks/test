# CALVIN ABC RLDS 一键转换与三阶段训练

更新时间：2026-07-20

本手册对应仓库内已下载的数据：

```text
/Users/tt/Documents/MoWE/dataset/Calvin_rlds/
  calvin_abc-train.tfrecord-00000-of-00512
  ...
  calvin_abc-train.tfrecord-00511-of-00512
```

入口为 `start_mtp_calvin.py`。它与 LIBERO `start_mtp.py` 使用同一套 Stage 1→2→3、best checkpoint、same-stage resume、predecessor identity、episode-balanced deployment validation 和 Stage 1 quality gate；CALVIN 的 raw data、action statistics、feature store、skill config、checkpoint 和输出目录完全独立。

## 1. 已核对的本地数据合同

本地全量只读审计已经实际通过，当时报告写入 `outputs/calvin_local_data_audit.json`；该报告是非版本化本地产物，当前缺失时正式启动器会在 run 目录重建：

- 512/512 个 `calvin_abc` train shard，全部 SHA-256 与下载 metadata 一致；
- 17,870 个独立语言记录，1,071,807 帧；
- 785,887 个 H=16 有效窗口；
- 17,800 个 source episode id，其中 70 个 id 被多个独立语言记录复用；全局 join key 因而固定为 `(shard, record_index, source_episode_id, timestep)`；
- primary/wrist 分别为 `200×200×3` 与 `84×84×3`，action 为 `[T,7]`，state 为 `[T,15]`；
- 动作统计只来自 ABC train；D 环境没有进入审计、归一化或训练；
- CALVIN 专用可审计动词映射覆盖 `slide/sweep/toggle/take/store/remove/unstack/collapse` 及 `go/in ... <motor verb>`，六个 motor 类均存在，unknown ratio 为 0。

这只证明本地原始数据和标签合同通过，不代表 feature store、GPU 训练或 D 环境 simulator 已通过。

## 2. 一键命令

先在 8 GPU 节点设置实际路径。OpenVLA 必须是原始 `openvla/openvla-7b` 的固定 40 位 revision，不能使用任何 LIBERO/OFT finetuned 权重。

```bash
export MOWE_ROOT=/ABS/MoWE
export CALVIN_RLDS_ROOT=/ABS/Calvin_rlds
export OPENVLA_ROOT=/ABS/openvla-7b
export OPENVLA_REVISION=47a0ec7fc4ec123775a391911046cf33cf9ed83f
export DINO_ROOT=/ABS/facebook-dinov2-small
export CALVIN_STORE=/ABS/mowe_store/calvin_abc_rlds_h16
export MOWE_RUNS=/ABS/outputs
export MOWE_PYTHON=/ABS/miniconda3/envs/mowe/bin/python
```

### 2.1 先做 dry-run

```bash
cd "$MOWE_ROOT"

"$MOWE_PYTHON" start_mtp_calvin.py \
  --repo-root "$MOWE_ROOT" \
  --dataset-root "$CALVIN_RLDS_ROOT" \
  --feature-store "$CALVIN_STORE" \
  --openvla-checkpoint "$OPENVLA_ROOT" \
  --openvla-revision "$OPENVLA_REVISION" \
  --dino-checkpoint "$DINO_ROOT" \
  --run-root-dir "$MOWE_RUNS" \
  --run-id calvin_abc_original_openvla_h16_v2 \
  --python "$MOWE_PYTHON" \
  --dry-run
```

dry-run 不读 51 GB payload、不加载模型、不要求真正执行 CUDA，只检查 512 shard、路径和不可变训练合同，并写出：

```text
$MOWE_RUNS/calvin_abc_original_openvla_h16_v2/
  launcher_state.json
  configs/stage1.json
  configs/stage2.json
  configs/stage3.json
  reports/stage1_quality_gate.json
```

`launcher_state.json` 应包含以下任务：

```text
calvin_rlds_h16_audit
calvin_feature_conversion
feature_store_audit
feature_equivalence
ddp_stage1_0_2
ddp_stage1_2_25
ddp_stage1_25_100
ddp_stage1_100_1000
ddp_stage1_1000_100000
ddp_stage2_0_100
ddp_stage2_100_50000
ddp_stage3_0_100
ddp_stage3_100_50000
```

### 2.2 正式启动

dry-run 通过后，仅删除末尾 `--dry-run`，其余参数保持完全一致：

```bash
cd "$MOWE_ROOT"

"$MOWE_PYTHON" start_mtp_calvin.py \
  --repo-root "$MOWE_ROOT" \
  --dataset-root "$CALVIN_RLDS_ROOT" \
  --feature-store "$CALVIN_STORE" \
  --openvla-checkpoint "$OPENVLA_ROOT" \
  --openvla-revision "$OPENVLA_REVISION" \
  --dino-checkpoint "$DINO_ROOT" \
  --run-root-dir "$MOWE_RUNS" \
  --run-id calvin_abc_original_openvla_h16_v2 \
  --python "$MOWE_PYTHON"
```

启动器会依次执行：

1. 校验 512 个 RLDS shard、原始 OpenVLA identity、DINO 路径和 8 GPU 可见性；
2. 全量读取 ABC train，验证 shard SHA-256、schema、技能覆盖，并生成独立 action q01/q99 和 skill config；
3. 若 `$CALVIN_STORE/manifest.json` 不存在或未达到 formal contract，则用单卡 BF16 自动生成 H=16 feature store；converter 按 episode 可恢复；
4. 验证全部 feature shard checksum、expected/actual episode/frame/window counts 和 8-rank assignment；
5. 用 100 个真实窗口重新编码 raw RLDS，完成 feature/output/loss 等价性门禁；
6. Stage 1 按 `0→2→25→100→1000→最多 100000` 运行，same-stage 只改 `stop_step`；H=1/4/8/16 world loss 权重为 `0.25/1/1/1`，delta 使用按 batch/horizon 的 RMS 归一化与 magnitude-aware cosine；
7. Stage 1 同时保存 deployment total-loss 最优的 `checkpoint_best.pt` 与 future-quality 最优的 `checkpoint_best_mechanism.pt`；质量门和 Stage 2 predecessor 固定使用后者，不允许两种选择标准错位；
8. Stage 2 先到 100-step smoke，再到最多 50,000；Stage 3 同样执行，并严格绑定 predecessor semantic identity；
9. 三阶段早停只使用 episode-balanced deployment `total_loss`，且必须等待 action conditioning；CALVIN v2 Stage 1 最早从 step 70,000 开始累计、patience 为 10，Stage 3 还必须等待 predicted-route schedule 完成。

脚本与 LIBERO 入口一样关闭系统/进程/GPU-memory telemetry；资源配额与告警交由平台。数据完整性、等价性、NaN/Inf、checkpoint lineage 和质量门不会关闭。

## 3. 中断恢复与重新开 lineage

任意时刻中断后，使用 2.2 完全相同的命令重新执行。converter 会跳过已发布 episode，Stage 1/2/3 会从各自 `checkpoint_latest.pt` 精确恢复。若原始 backbone、DINO 与数据身份不变，已经通过 formal audit/equivalence 的 CALVIN feature store 可以复用，不需要因为 loss 改动重新编码。

2026-07-20 之前的 `calvin_abc_original_openvla_h16_v1` 是 50,000-step 旧 loss lineage。它必须保留用于比较，但不得在当前代码下续接；v2 的 horizon weighting、delta normalization/cosine、100,000-step scheduler 与 mechanism checkpoint 都属于新的恢复合同。

以下情况必须换新的 `--run-id`，不能续接旧 checkpoint：

- OpenVLA revision/fingerprint、DINO、CALVIN raw dataset 或 feature store 改变；
- action q01/q99、H=16 窗口、skill mapping、optimizer/LR/schedule、flow solver 或 stage max steps 改变；
- Stage 1/2 best predecessor 改变，而已有 Stage 2/3 来自旧 predecessor。

如果只改变日志频率、保存频率或同一 stage 的 `stop_step`，可按 same-stage 合同恢复。

## 4. 训练后官方 D 环境评测

训练完成不等于 CALVIN benchmark 完成。本地下载的 512 shard 只有 ABC train；D 环境、官方 CALVIN 仓库和 1,000-sequence LH-MTLC evaluator 必须另行安装并绑定固定 commit `fa03f01f19c65920e18cf37398a9ce859274af76`。

先做 one-sequence smoke，再执行正式 1,000 sequences。入口：

```bash
"$MOWE_PYTHON" scripts/eval_calvin_flow_wam.py --help
```

正式评测必须使用 Stage 3 `checkpoint_best.pt`、其保存的 execution config 和 CALVIN 独立 action adapter；不得复用 LIBERO statistics/checkpoint，也不得把离线 loss 或 adapter smoke 当作 simulator success rate。
