# MoWE-WAM 当前实现与执行计划

更新时间：2026-07-19（活动文档；完整历史规格已归档）

本文档是后续 Codex/agent 编程时默认读取的实现入口，只保留当前事实、关键合同、未完成门槛和下一步顺序。文档瘦身前的完整规格保存在 [`docs/history/IMPLEMENTATION_PLAN_FULL_2026-07-17.md`](docs/history/IMPLEMENTATION_PLAN_FULL_2026-07-17.md)，仅在追溯具体设计、旧任务或历史命令时按章节读取，禁止在新会话启动时默认全文加载。

当前执行范围不推进消融实验；先把原始 OpenVLA backbone、formal feature store、8 卡连续训练、LIBERO 和 CALVIN 主流程跑通。已有消融配置保留但不扩展，也不阻塞当前工程交付。

权威顺序：

1. `PROJECT_PLAN.md` 决定研究主张和论文方向。
2. 本文档决定当前如何实现和执行。
3. `ARCHITECTURE_RISKS.md` 决定停止条件与证据门槛。
4. `DEV_LOG.md` 记录实际完成状态；代码、checkpoint 和日志证据优先于计划文字。

## 0. 新会话快速接管

新 agent 先完成以下最小读取，不要全文读取历史归档：

1. `CODEX_PROJECT_RULES.md`。
2. `PROJECT_PLAN.md` 的“当前主线”“训练路线”“数据与 Benchmark 策略”“风险与停止条件”。
3. 本文档全文。
4. `ARCHITECTURE_RISKS.md` 的风险总表、进入长训练前门槛，以及与当前任务对应的风险章节。
5. `DEV_LOG.md` 的“当前状态快照”和最新 1～3 条记录。
6. 使用 `git status --short`、`rg --files` 和当前代码/config 验证文档，不从归档推断代码已经存在或实验已经通过。

历史定位示例：

```bash
rg -n '^#{1,4} |Stage 3|feature store|CALVIN|Definition of Ready' \
  docs/history/IMPLEMENTATION_PLAN_FULL_2026-07-17.md
rg -n '^## 2026-|readiness|Stage 2|8 卡' \
  docs/history/DEV_LOG_2026-07-08_to_2026-07-17.md
```

## 1. 当前主线与代码状态

当前主方法：

> Nominal Flow Policy + Verb-Seeded Residual Flow Skill Experts + MAW Routing

当前真实代码主线为 `flow_wam_skill_moe`：

```text
LIBERO/CALVIN episodic windows or mowe_feature_store_v1
  -> frozen non-leaky original openvla/openvla-7b visual/language context
  -> multi-scale memory
  -> nominal 6D motion flow + binary gripper head
  -> nominal-action-conditioned Latent WAM
  -> h_1...h_16 + multi-horizon future visual latent/delta
  -> [B,16,7] temporal skill router
  -> one shared residual-flow solve with per-token motor expert
  -> synchronous risk-gated execution: 8 / 4 / high-risk-boundary stop
```

旧 `legacy_predicate` 和 `regression_residual_moe` 路径保留为 baseline，不是默认研究主线，不应继续扩展为主模型。

### 已有实现

- LIBERO episodic sequence dataset、同 episode history/long-memory/future/action window、逐位置 skill sidecar join。
- 原始 `openvla/openvla-7b` non-leaky context、OFT-compatible 双视角加载/融合、DINO visual target 和 cache。
- nominal motion flow、独立 gripper head、Latent WAM、temporal router、六 motor residual-flow experts、null bypass。
- Stage 1/2/3 CLI/runtime、checkpoint/resume、日志、preflight、机制分析。
- 单机 DDP、episode-aware sharding/sampler、rank-0 I/O；通用 runtime 仍保留可选资源诊断，但 `start_mtp.py` 明确关闭且不调用任何系统/内存监控。
- `mowe_feature_store_v1`、canonical archive、可恢复 converter、结构/equivalence/soak/readiness 审计。
- feature-store validation 使用确定性的 episode-balanced sparse sampler；正式 launcher 同时记录 diagnostic 与 deployment 两种验证，早停只使用 deployment loss，并等待动作/路由课程调度完成。
- `checkpoint_best.pt`、Stage 1 `copy_current` 质量门和 Stage 2/3 predecessor identity 校验已实现；`checkpoint_latest.pt` 仍只用于精确 same-stage resume。
- 可恢复 LIBERO full-suite evaluator。
- CALVIN action/policy bridge、官方 NPZ 与 512-shard ABC RLDS reader/converter、feature-store configs、`start_mtp_calvin.py` 和官方 evaluator bridge。

### 已有证据边界

- 本地 synthetic/contract、CPU 2-rank Gloo DDP 和 checkpoint resume 已有通过记录；修改代码后仍应重跑相关测试。
- 云端 RTX 4090 已记录旧 LIBERO-OFT backbone 下的真实 Stage 1 step 0→100、resume，以及 Stage 2 oracle one-step/25-step coverage gate；这些结果只证明历史代码路径，不满足新的原始-backbone训练合同，也不能作为新 lineage 的 resume 起点。
- 已完成的一轮 8 卡 launcher 输出暴露出旧验证/早停合同问题：validation prefix 仅覆盖 1 个 episode，Stage 1/2/3 分别约在 11k/7.5k/5k 早停，Stage 3 尚未进入 predicted-route、各阶段尚未完成 nominal-action 课程；这些结果不能作为机制或正式模型质量证据。
- 新 `v2` lineage 已在真实 8 卡节点完成完整 Stage 1→2→3。Stage 1 latest/best 为 `47,500/45,000`，Stage 2 与 Stage 3 latest/best 均为 `38,000/35,500`；三阶段均在调度完成后的 78-episode deployment validation 上合规早停，launcher `pipeline.status=complete`。Stage 1 `mowe_stage1_quality_gate_v3` 已通过；Stage 3 best 的 H=4/8/16 相对 `copy_current` 改善约为 `+1.01%/+40.48%/+58.91%`。Stage 3 predicted router 的 boundary recall/F1 仍偏低，必须由 simulator 小样本门槛判断实际影响。
- 修正版一键入口的真实 8-GPU NCCL 三阶段训练已完成；LIBERO simulator smoke、四-suite success rate、future-shuffle 机制验证、CALVIN 正式训练与官方评测仍未完成。离线训练指标不能替代这些证据。
- 代码存在、CLI 可运行、mock/contract 测试通过，均不能替代上述真实证据。

## 2. 当前唯一优先执行链

除非用户明确改变研究方向，按以下顺序继续：

1. **冻结原始 backbone identity**：下载 [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) 的固定 Hugging Face revision 为只读本地 snapshot，记录 repo id、revision 和权重 fingerprint；主线禁止使用任何 LIBERO-finetuned OFT checkpoint。
2. **原始 backbone 双视角 smoke**：用现有 OFT-compatible loader 验证 primary/wrist processor、ordered multi-image forward、per-view pooling、language encoding、BF16 数值与显存；该 smoke 不训练模型。
3. **LIBERO 转换 smoke**：用原始 backbone 和 2 个 episode 生成 non-formal feature store/canonical archive，完成结构审计；smoke store 不得进入正式训练。
4. **LIBERO 全量转换**：仅从通过 smoke 的同一原始-backbone snapshot 生成满足 expected/actual episode、frame、window counts 的 formal store。
5. **等价性和数据审计**：至少 100 个真实窗口验证 raw original-backbone/store 的 views、language、DINO、actions、skills、完整 outputs/losses；冻结 8-rank suite/window/skill imbalance 上限。
6. **新 Stage 1 lineage 阶梯**：不恢复任何旧 OFT-backbone checkpoint；使用正式 50,000-step 配置从 step 0 新建 8 卡 lineage，依次验证 `0→2→25→100→1000`。same-stage resume 只改变 `stop_step`，从第一步起保持 `max_steps=50000` 和全部优化语义。
7. **Stage 1 连续训练**：从 step 100 连续训练到 1000，再继续到最多 50,000；早停不得早于 nominal-action 课程完成。
8. **机制门槛（已完成）**：选择 eligible deployment loss 最优的 `checkpoint_best.pt`，并在至少 32 个不同 validation episode 上要求 H=4/8/16 相对 `copy_current` 的平均改善至少 10%、任一 horizon 的相对回退不超过 5%；通过后才启动 Stage 2/3。`v2` 已通过并完成 Stage 2/3，最终选择 Stage 3 best step 35,500。
9. **正式评测（当前步骤）**：目标服务器无 sudo/EGL，仅使用 OSMesa。先完成 LIBERO one-task/1-trial smoke，再完成四 suite 每任务 5 trials gate；无 crash/NaN、success 非全零且执行前缀合理后，才运行四 suite × 每任务 50 trials × seeds `7/17/27` 的可恢复正式评测与机制分析。准确命令以 `docs/MTP_ONE_CLICK_TRAINING.md` 第 9 节为准。
10. **CALVIN 第二基准**：仅在 LIBERO 主线稳定后，以同一原始 backbone identity 重新生成 CALVIN 独立 store，完成 ABC 独立三阶段训练和 D 环境官方 1,000-sequence LH-MTLC 评测。

任何步骤失败时停在该层解决；不要通过放宽数据完整性、checkpoint、训练质量或泄漏门槛绕过。系统资源监控由平台侧负责，不进入一键脚本。

## 3. 不可漂移的实现合同

### 3.1 数据与标签

- `history_length=8`，`long_memory_slots=4`，`future_horizons=[1,4,8,16]`，`action_chunk_size=16`。
- 所有 history、future、target action 和 labels 必须来自同一 episode。
- 训练标签来自可审计的 per-timestep final-directive leading verb；整段 CoT、未来帧、simulator state、test split 不得进入部署输入或标签构造。
- routes 为六个 motor skills 加 `null_finish`；unknown 为 `-1` 并 mask，不得回退为 expert 0。
- `null_finish` 是 motion residual bypass，不是 episode termination；nominal action 仍保留。

### 3.2 动作与模型

- Frozen backbone 必须是 `openvla/openvla-7b` 的已签发 immutable revision；`checkpoint` CLI 优先级高于 `vla_path`，因此所有正式命令都必须显式传入该本地 snapshot，不能传入 `/hy-tmp/openvla-7b-oft-libero-all` 或 `moojink/*-oft-finetuned-*`。
- `external/openvla-oft` 只提供兼容的模型注册和 ordered multi-image loader，不代表使用 OFT-finetuned 权重。主线不加载 OpenVLA action head、proprio projector 或 LIBERO OFT continuous-action head。
- feature-store manifest、resolved config、训练 checkpoint 和评测记录必须绑定 base repo id、revision、snapshot/weight fingerprint 与 processor identity；仅记录一个可变 HF alias 或人工修改 manifest 不满足正式合同。
- 该身份合同由 `mowe_wam/backbones/openvla_identity.py` 实现：repo 必须为 `openvla/openvla-7b`，revision 必须是 40 位 commit，config/processor/weights 分别生成 SHA-256 fingerprint；同一快照允许挂载在不同绝对路径，身份不一致或命中 OFT/LIBERO-finetuned 标记时 fail closed。
- 旧 OFT-backbone feature store 与 Stage 1/2 checkpoint 不允许通过 `--allow-world-size-change` 迁移；该参数只授权 world-size 变化，不能授权 backbone 变化。
- 只有前 6 维 relative motion 进入 normalization、flow noise、solver 和 residual addition。
- residual target、sample 与诊断统一按 timestep 投影到 `max_residual_l2=0.5`，再与 nominal 相加并裁剪到 `[-1,1]`；checkpoint resume contract 锁定该值，避免中途改变 expert 修正尺度。
- 第 7 维 gripper 是独立 binary head/BCE；不得进入 flow 或被 residual expert 修改。
- Router 输出 `[B,16,7]`；query 固定为 `ActionMLP(A0[j]) + WorldProjection(h[j+1]) + PositionEmbedding(j)`。
- residual trunk/solver 每 chunk 只运行一次；禁止分别采样六个完整 chunks 后拼接。
- 推理采用无异步生成的同步 receding-horizon：每轮预测 16 步，默认提交前 8 步；若前 8 步内某个 skill 边界满足 caution 风险条件则提交 4 步；若满足 high-risk 条件则在该边界前停止，因此低于 4 步只会作为少数高风险结果出现。稳定的 Pick→Move→Place 边界允许跨越。
- 边界风险使用归一化 router entropy、Top-2 margin、相邻 6D normalized-motion jump 和 residual L2 联合判定。默认 caution/high 阈值分别为 entropy `0.55/0.75`、margin `0.20/0.10`、motion jump `0.60/0.90`、residual L2 `0.35/0.45`。
- 提交前缀耗尽后，必须基于最新观测同步重新生成；旧 chunk 未执行 suffix 一律丢弃，不做 temporal ensemble、motion stitching、旧 tail 复用或后台异步预测。
- 完整 `execution` 配置属于 checkpoint/readiness 恢复合同；正式评测默认使用 checkpoint 保存的阈值，不能在同一 lineage 中静默改写。
- teacher 只定义训练目标；推理不加载 teacher，不生成视频。

### 3.3 训练与恢复

- 三阶段入口：
  - `scripts/pretrain_nominal_flow_wam.py`
  - `scripts/warmstart_skill_flow_experts.py`
  - `scripts/train_flow_wam_skill_moe.py`
- Stage 2 只从 Stage 1 predecessor 初始化；Stage 3 只从 Stage 2 predecessor 初始化。same-stage 使用 `--resume`，跨 stage 使用 `--init-wam`。
- same-stage resume 必须保持 seed、stage、precision、完整 `max_steps`、optimizer/LR/schedule/loss/window/route contract 不变；只允许调整 `stop_step`、`save_freq`、`log_freq`。
- 正式 feature-store validation 每个 validation episode 确定性抽取一个窗口，同时输出 diagnostic（GT action；Stage 2/3 oracle route）和 deployment（nominal action + predicted route）记录；早停只读取 deployment `total_loss`，且至少覆盖 32 个不同 episode。
- 默认 50,000-step 合同下，early-stopping patience 只能从 action conditioning 与（Stage 3）predicted-route schedule 完成后的 step 35,000 开始累计；此前验证只用于诊断，不得消耗 patience。
- 每个阶段把 eligible deployment loss 最优状态保存为 `checkpoint_best.pt`；跨阶段初始化使用 best checkpoint，并把其 path-independent semantic identity 写入下一阶段 checkpoint。若 predecessor 改变，旧 Stage 2/3 checkpoint 必须 fail closed，不能拼接 lineage。
- 正式 DDP 合同为 world size 8、BF16、per-device batch 1、accumulation 1、effective global batch 8、NCCL、`num_workers=0`、`pin_memory=false`。
- `start_mtp.py` 生成的三阶段配置固定 `resource_monitoring=false`，不读取 `/proc`/cgroup、RSS、OOM event 或 GPU memory telemetry，也不运行 soak/runtime/readiness 资源审计；数据审计、等价性、checkpoint 和质量门继续 fail closed。

### 3.4 Benchmark 隔离

- LIBERO 与 CALVIN 必须分别绑定 dataset/store fingerprint、action statistics、backbone identifier、resolved config、checkpoint 和 evaluator commit。
- 两个 benchmark 可以共享同一 immutable 原始 OpenVLA base snapshot，但不能共享由它产生的 feature store、action statistics 或 MoWE checkpoint。
- 不得复用 LIBERO q01/q99 作为 CALVIN action contract。
- CALVIN D/validation 不得进入训练、normalization、cache、模型选择或 early stopping。
- 离线 accuracy、CLI smoke 和 adapter contract 不能替代 simulator success rate。

## 4. 当前代码导航

| 目的 | 主要文件 |
|---|---|
| Flow-WAM policy | `mowe_wam/models/flow_wam_policy.py` |
| Nominal/residual flow | `mowe_wam/models/action_flow.py`, `nominal_action_head.py`, `residual_experts.py` |
| WAM/router/memory | `latent_world_model.py`, `future_router.py`, `memory/multiscale_memory.py` |
| LIBERO raw dataset | `mowe_wam/data/libero_sequence_dataset.py`, `cot_skill_sidecar.py` |
| Feature store/archive | `mowe_wam/data/feature_store.py`, `canonical_archive.py`, `backbones/precomputed_features.py` |
| Runtime/DDP/readiness | `mowe_wam/training/flow_runtime.py`, `distributed.py`, `long_run_readiness.py` |
| LIBERO evaluator | `scripts/eval_libero_temporal_skill.py`, `mowe_wam/evaluation/libero_temporal_policy.py` |
| CALVIN | `mowe_wam/benchmarks/calvin/`, `scripts/convert_calvin_to_mowe_store.py`, `scripts/eval_calvin_flow_wam.py` |
| Contract tests | `tests/test_flow_contracts.py`, `test_flow_torch_contracts.py`, `test_flow_distributed.py`, `test_feature_store.py`, `test_long_run_readiness.py` |

定位接口时先使用 `rg`，不要仅依据本文档猜测签名。

## 5. 活动配置

### LIBERO

- 云服务器全流程：`docs/CLOUD_TRAINING_RUNBOOK.md`
- 单卡 Stage 1/2/3：`configs/mowe_wam/single_gpu_nominal_flow_wam_feature_store.yaml`、`single_gpu_warmstart_skill_flow_feature_store.yaml`、`single_gpu_train_flow_wam_feature_store.yaml`
- 8 卡正式 Stage 1：`configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml`
- Stage 2：`configs/mowe_wam/ddp8_warmstart_skill_flow_feature_store.yaml`
- Stage 3：`configs/mowe_wam/ddp8_train_flow_wam_feature_store.yaml`
- Skill taxonomy：`configs/mowe_wam/skill_experts.yaml`

### CALVIN

- Benchmark/action：`configs/mowe_wam/calvin_abc_d.yaml`
- 一键转换/审计/训练：`start_mtp_calvin.py`、`docs/MTP_CALVIN_ONE_CLICK_TRAINING.md`
- Stage 1：`configs/mowe_wam/ddp8_calvin_nominal_flow_feature_store.yaml`
- Stage 2：`configs/mowe_wam/ddp8_calvin_warmstart_skill_flow_feature_store.yaml`
- Stage 3：`configs/mowe_wam/ddp8_calvin_joint_flow_feature_store.yaml`

配置中的 `TBD`、空路径或本机路径必须在目标服务器显式解析。不得把示例路径写入正式 evidence/report。

## 6. 最小验证与云端命令

### 6.1 本地代码合同

```bash
/Users/tt/miniconda3/envs/mowe/bin/python -m compileall -q mowe_wam scripts tests
/Users/tt/miniconda3/envs/mowe/bin/python -m unittest discover -s tests -v
git diff --check
```

测试结果必须写入 `DEV_LOG.md`，不能复用旧日志声称当前 tree 已通过。

### 6.2 原始 backbone 双视角 preflight

先将 `openvla/openvla-7b` 的固定 revision 下载为只读本地 snapshot；`OPENVLA_REVISION` 必须替换为 Hugging Face 上核对后的完整 40 位 commit，不能写 `main`。实际 revision 和权重 fingerprint 会由代码写入转换报告，不能只记录可变 repo alias。随后用一个真实 LIBERO batch 验证当前 processor、双视角和 language context 路径，不执行 backward：

```bash
python scripts/preflight_flow_wam_training.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root /ABS/modified_libero_rlds \
  --checkpoint /ABS/models/openvla-7b-base-REVISION \
  --backbone-revision "$OPENVLA_REVISION" \
  --precision bf16
```

该 preflight 只证明原始 backbone 能按当前 context contract 前向运行；它不是训练或 benchmark 证据。当前实现会在加载前校验 repo/revision 并计算 config、processor、weights 指纹；任何旧 OFT/LIBERO-finetuned reference 都会直接失败。

### 6.3 LIBERO 两 episode 转换 smoke

以下 `/ABS/...` 必须替换为已核对的绝对路径：

```bash
# `TBD` 必须从当前数据/控制协议核对后替换为正数。
python scripts/convert_rlds_to_mowe_store.py \
  --config configs/mowe_wam/dataset_libero_mixture.yaml \
  --data-root /ABS/modified_libero_rlds \
  --checkpoint /ABS/models/openvla-7b-base-REVISION \
  --backbone-revision "$OPENVLA_REVISION" \
  --teacher-checkpoint /ABS/dinov2_checkpoint \
  --output /ABS/mowe_store_smoke \
  --canonical-output /ABS/mowe_archive_smoke \
  --canonical-fps TBD \
  --limit-episodes 2 \
  --device cuda \
  --precision bf16
```

该输出必须保持 `formal_training_ready=false`。随后先用 `--help` 核对各审计脚本的当前参数。

### 6.4 原始 backbone 的 8 卡 Stage 1 新 lineage

```bash
torchrun --standalone --nproc-per-node=8 scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --feature-store /ABS/formal_libero_feature_store \
  --output-dir /ABS/stage1_openvla_base_ddp8 \
  --max-steps 50000 \
  --stop-step 2

torchrun --standalone --nproc-per-node=8 scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --feature-store /ABS/formal_libero_feature_store \
  --resume /ABS/stage1_openvla_base_ddp8/checkpoint_latest.pt \
  --output-dir /ABS/stage1_openvla_base_ddp8 \
  --max-steps 50000 \
  --stop-step 25
```

继续到 100/1000/50000 时只改 `--stop-step`，`--max-steps` 始终保持 50000。完整一键参数以 `docs/MTP_ONE_CLICK_TRAINING.md` 为准。不得恢复旧 OFT-backbone checkpoint 或重定义 scheduler horizon。

## 7. 当前任务状态

| 任务 | 状态 | 下一证据 |
|---|---|---|
| 主线/legacy 隔离 | 已实现 | 保持 baseline 可导入 |
| LIBERO sequence/skill join | 已实现并有真实结构审计记录 | formal feature-store 全量计数 |
| 原始 OpenVLA/DINO/memory | loader/缓存代码已实现；现有真实训练记录来自旧 OFT backbone | 固定 base revision、双视角 smoke、重建 store、100-window raw/cache equivalence |
| Flow-WAM/router/experts | 已实现 | Stage 3 真实 optimizer/resume |
| 单机 DDP | 代码与 2-rank CPU contract 已完成 | 修正版一键入口 8-rank NCCL 实跑 |
| Feature store/archive | 核心代码与正式 store/checksum 完成 | 保持数据审计通过 |
| LIBERO evaluator | 代码完成 | one-task smoke 和四-suite 正式结果 |
| CALVIN adapter/converter/evaluator | 本地 512-shard ABC RLDS 全量数据/标签/checksum 审计通过；一键转换与三阶段 dry-run contract 完成 | 原始-backbone formal store、100-window equivalence、8-GPU 三阶段训练、D 环境官方结果 |
| 消融实验 | 本轮暂缓；已有配置保留 | 主训练与双 benchmark 稳定后再排期 |

## 8. 长训练 Definition of Ready

以下全部满足前，不批准连续长训练：

- formal feature store 的 expected/actual episode、frame、window counts 完全一致，全 shard checksum 通过。
- store、resolved config、checkpoint 与 evaluator 均绑定同一个已签发 `openvla/openvla-7b` immutable revision/weight fingerprint；不存在旧 OFT-backbone lineage 混入。
- 100 个真实窗口 raw/cache equivalence 通过，benchmark identity 与 store 一致。
- 8-rank episode union 完整且互斥；suite/window/skill imbalance 上限由真实 audit 后人工确认。
- checkpoint stage、world size、effective batch 和 same-stage schedule 合法；Stage 1 lineage 从原始 backbone step 0 建立，不发生 backbone migration。
- 一键 dry-run 展开完整 Stage 1→2→3 init/resume 链，且命令中不存在系统资源监控、soak 或 resource readiness 操作。
- Stage 1 future predictor 相对 `copy_current` 的改善达到继续训练的实验判断门槛。

## 9. 第一版 Definition of Done

- LIBERO Stage 1→2→3 在正式 store 和目标 GPU 合同下完成并可恢复。
- LIBERO 四 suite 正式 success rate、机制指标、主要 baselines/ablations 完成。
- predicted routing、expert coverage、boundary、null-zero、view fusion、latency 和资源指标可审计。
- CALVIN ABC→D 使用独立数据/action/checkpoint/evaluator contract 完成官方 LH-MTLC 评测。
- 所有论文表述遵守 `PROJECT_PLAN.md` 和 `ARCHITECTURE_RISKS.md` 的证据边界。

## 10. 归档索引

- [`docs/history/IMPLEMENTATION_PLAN_FULL_2026-07-17.md`](docs/history/IMPLEMENTATION_PLAN_FULL_2026-07-17.md)：旧完整 API、配置、Task 0～10、Smoke-first 命令和长训练设计。
- [`docs/history/DEV_LOG_2026-07-08_to_2026-07-17.md`](docs/history/DEV_LOG_2026-07-08_to_2026-07-17.md)：全部历史变更、实际命令、结果和失败记录。

查找后只读取命中章节附近内容，不要全文加载归档。

## 11. 更新规则

- 改动代码或项目文档后，在根 `DEV_LOG.md` 顶部快照同步“已验证/未验证/下一步”，并追加一条简短记录。
- 只有当前执行合同、任务状态或下一证据发生变化时才修改本文档；详细调试过程写入 `DEV_LOG.md`。
- 根 `DEV_LOG.md` 最多保留最近 10 条。超过后将最旧条目追加到按月历史文件，并保持链接。
- 不重复粘贴大段代码、完整日志或旧命令；使用文件路径、commit/checkpoint/report fingerprint 和归档章节定位。
- 归档只追加或新建快照，不把归档中的旧结论当作当前验证结果。
