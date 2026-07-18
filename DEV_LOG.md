# MoWE-WAM 当前状态与近期开发日志

更新时间：2026-07-18

本文件是新会话默认读取的活动日志，只保留当前快照和最近变更索引。2026-07-08 至文档瘦身前的完整逐条记录保存在 [`docs/history/DEV_LOG_2026-07-08_to_2026-07-17.md`](docs/history/DEV_LOG_2026-07-08_to_2026-07-17.md)。需要核对旧命令、输出或失败原因时，先用 `rg` 定位，再读取命中章节；禁止在启动时全文加载历史归档。

## 当前状态快照

### 当前主线

- 研究主线为 `Nominal Flow Policy + Verb-Seeded Residual Flow Skill Experts + MAW Routing`。
- 第一版唯一 frozen context backbone 已固定为 immutable revision 的原始 `openvla/openvla-7b`；`external/openvla-oft` 只作为双视角兼容 loader，不使用 LIBERO-finetuned OFT 权重。
- 当前优先级是在目标 8 卡节点通过 `start_mtp.py` 完成 8-rank soak、8-GPU runtime/readiness 与正式 Stage 1→2→3；LIBERO 正式训练/评测稳定后再推进 CALVIN。
- 旧 predicate/event 和 regression residual-MoE 代码只作为 baseline 保留。

### 当前代码事实

- Stage 1/2/3 Flow-WAM、DDP、feature store、canonical archive、readiness、LIBERO evaluator、CALVIN adapter/converter/evaluator 的代码路径已存在。
- `flow_wam_base.yaml` 已有 `vla_path=openvla/openvla-7b`，但 CLI `checkpoint` 会覆盖它；正式转换/训练/评测必须传入固定 revision 的原始模型本地 snapshot。
- 原始 backbone identity 已实现为 repo + 40 位 revision + config/processor/weights SHA-256 指纹；store、checkpoint、equivalence readiness 和 evaluator 已按该 identity fail closed，不再用服务器绝对路径作为语义身份。
- Residual expert 的训练 target、采样 endpoint 与诊断统一使用逐 timestep `max_residual_l2=0.5` 投影，该值已进入 same-stage resume contract。
- 主线已切换为 16-step prediction + synchronous risk-gated receding horizon：稳定时提交 8 步，caution 时 4 步，high-risk 时在边界前停止；每轮丢弃旧 suffix，不使用异步生成、stitching 或旧 tail。
- 工作树包含尚未提交的代码、配置、测试和文档修改。后续 agent 必须先运行 `git status --short` 并保留这些修改，不得 reset、checkout 或覆盖。
- 根 `IMPLEMENTATION_PLAN.md` 已压缩为当前执行合同；原完整实现计划和完整日志已移入 `docs/history/`。

### 已记录的验证

- 本地 synthetic/contract、feature-store、readiness、CPU 2-rank Gloo DDP/checkpoint resume 有通过记录。
- 云端 RTX 4090 已记录旧 LIBERO-OFT backbone 下的真实 Stage 1 step 0→100、same-stage resume，以及 Stage 2 oracle one-step/25-step coverage gate。
- 这些历史结果不满足新的原始-backbone identity，不能 resume 为新主线 checkpoint，也不能替代新 feature store/双视角 smoke。
- 上述是历史证据；当前 tree 发生修改后，相关测试必须按风险重新运行。
- 当前 16-step/risk-gated 修改已通过 synthetic joint forward/backward、8/4/边界停止专项合同、variable-prefix queue smoke 和本地 63 项完整单元/合同测试；这仍不是实际 GPU/simulator 成功率证据。
- 2026-07-18 单卡 A100 已完成新的原始 OpenVLA real-batch BF16 双视角 backward preflight：真实 H=16 RLDS/CoT、原始 7B 与 DINO 均已加载，`total_loss=4.308042526245117`，风险 gate 全部通过。两 episode H=16 feature-store smoke 及 checksum/shape 审计也已通过；partial smoke store 按合同保持 non-formal，不能训练。
- 4090 formal store `/hy-tmp/mowe_store/libero_h16_formal_4090` 已完成：1,693 episodes、273,465 frames、246,377 windows，expected/actual counts 和全 shard checksum/8-rank assignment audit 均通过。
- 最终真实 100-window mask-aware equivalence 已通过：100/100 pair、无 missing、`masks_match=true`、`max_feature_gate_error=0.00427994`、`max_output_gate_error=0.00221920`、`max_loss_gate_error=0.01294136`、`passed=true`。
- 单卡 A100 已分别完成原始-backbone Stage 1/2/3 step 0→100 工程验证：checkpoint stage/lineage 正确、loss/gradient 有限、Stage 2 六个 motor expert 与 Stage 3 router 梯度合同通过。它们不是正式 8 卡 lineage 的初始化 checkpoint。
- `start_mtp.py` 已实现单入口 8 卡正式训练编排与断点恢复，并配套 `docs/MTP_ONE_CLICK_TRAINING.md`；当前只有 dry-run/合同测试证据，尚未在真实 8 卡节点完整执行该 launcher。
- 对于 MTP 未开放 `/proc`/cgroup 的容器，launcher 现提供显式 `--disable-system-monitoring` 降级模式：跳过系统/cgroup/OOM/boot 资源证据，保留 CUDA、NCCL、GPU 显存、数据和训练质量门槛；其报告会明确标注为 degraded，不能与标准 cgroup 证据混用。
- raw/cache equivalence 的 raw matcher 现优先使用 formal store 已记录的精确 `(source_file_key, source_traj_index, step_id)`，只有旧 store 缺少该 provenance 时才回退图像摘要 `episode_id`；这修复跨服务器同计数/同 manifest 下的少量 image-fingerprint missing pair，不会放宽 100/100 gate。
- 真实 8×GPU Stage 1 step 0→2 首次运行在完成 step 1 后触发 DDP unused-parameter reducer 错误；参数索引已精确映射为 nominal trunk 的可选 `token_condition_projection` 与 Stage-1 无监督的 `world_model.route_world_head`，stage 配置现按真实梯度路径冻结它们，Stage 2/3 会重新启用 route-world head。

### 尚未完成

- 目标节点 CPU 8-rank continuous soak 和 8×A100/A800 NCCL/runtime/cgroup 证据。
- 原始-backbone新 lineage 的 step 0→2→25→100→1000→50000 DDP Stage 1 链路及 checkpoint-bound readiness。
- Stage 1 future predictor 明确优于 `copy_current`。
- Stage 3 真实 optimizer/resume、LIBERO 四 suite 正式结果和机制消融。
- CALVIN ABC 全量转换/训练及 D 环境官方 1,000-sequence 评测。

### 下一步

1. 将 repo、RLDS、formal store、OpenVLA、DINO 和 sidecar 挂载到目标 8 卡节点，先用 `start_mtp.py --dry-run` 核对全部解析路径与命令。
2. 使用完全相同的 `run-root-dir`/`run-id` 去掉 `--dry-run` 正式启动；launcher 必须先通过 node-bound soak/runtime/readiness，随后才从 Stage 1 step 0 建立正式 lineage。
3. 若平台没有系统资源指标权限，在 dry-run 和正式命令均追加 `--disable-system-monitoring`；这会降低资源证据等级，运行中应额外用平台控制台观察 CPU RAM、GPU memory 和 OOM。

## 最近关键变更索引

以下是归档中最近、仍影响当前执行合同的记录摘要。精确命令和证据以归档原条目为准。

| 日期 | 变更 | 当前影响 |
|---|---|---|
| 2026-07-17 | 同步 16-step prediction + 8/4/high-risk execution | 重建 H=16 store；旧 H=8 store/checkpoint 不兼容；部署不复用旧 suffix |
| 2026-07-17 | 实现原始 OpenVLA identity 与 residual 安全边界 | 旧 OFT/path-only 产物直接失败；正式转换必须提供 40 位 revision；residual L2≤0.5 |
| 2026-07-17 | 主线冻结为原始 `openvla/openvla-7b` | 重建 feature store，Stage 1 从 step 0 新建 8 卡 lineage；旧 OFT checkpoint 不可恢复 |
| 2026-07-17 | DDP 资源门禁 fail closed | 正式多卡缺少 cgroup-v2/CUDA 指标时不得启动 |
| 2026-07-17 | LIBERO expected/actual counts 进入 formal gate | `--limit-episodes` 或不完整 store 永远不能用于正式训练 |
| 2026-07-17 | resolved config 延后并原子发布 | 启动合同全部通过前不得留下伪就绪配置 |
| 2026-07-17 | 固定 LIBERO + CALVIN 双基准顺序 | LIBERO 在前，CALVIN 独立 action/data/checkpoint/evaluator |
| 2026-07-17 | 增加 8 卡长期训练证据聚合 | formal store、equivalence、soak、runtime、checkpoint 缺一不可 |
| 2026-07-17 | CALVIN raw/cache 等价性门禁 | CALVIN 使用独立 benchmark identity 和 100-window 报告 |
| 2026-07-17 | 清理 feature-store provenance/Stage 2 常数 loss | Stage 2 total loss 不混入冻结 nominal/gripper 与 oracle load-balance |
| 2026-07-17 | same-stage resume 完整语义锁定 | `max_steps` 等优化合同不可变，只可调整 `stop_step`/日志频率 |
| 2026-07-17 | shard-aware block shuffle | sampler block size 进入 checkpoint/readiness contract |
| 2026-07-17 | readiness 接入正式训练入口 | 超过 100 个未认证 steps 必须加载匹配 report |

## 2026-07-18 - 单卡 A100 原始 OpenVLA real preflight 与 feature-store smoke

### Goal

按 `docs/CLOUD_TRAINING_RUNBOOK.md` 验证新版原始 `openvla/openvla-7b`、H=16 双视角合同能在单张 A100 上读取真实 LIBERO RLDS、完成 backward，并写出可审计的非正式 feature-store smoke。

### Changed

- 修复 `OpenVLAContextAdapter`：在显式注册本地 OFT-compatible model/processor 后以 `trust_remote_code=false` 加载原始 snapshot，防止 checkpoint `auto_map` 覆盖多图 loader；新增该行为回归测试。
- 修复 real preflight 对 `--skill-sidecar` 的 config 覆盖；为 converter 增加同名显式 CLI 参数并在云手册两条 conversion 命令中使用，避免示例相对路径覆盖服务器实际 sidecar。
- 云手册变量已改为本机实际 `/hy-tmp` 数据、OpenVLA、DINO、store 路径，并记录实际依赖/fork revisions 与本次证据。

### Commands Run

```bash
python scripts/audit_flow_wam_rlds.py --data-root /hy-tmp/libero_cot_rlds \
  --skill-sidecar /hy-tmp/libero_cot_rlds/cot_file.json --max-horizon 16 \
  --output outputs/cloud_reports/libero_rlds_h16_audit.json

CUDA_VISIBLE_DEVICES=0 python scripts/preflight_flow_wam_training.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root /hy-tmp/libero_cot_rlds --checkpoint /hy-tmp/openvla-7b \
  --backbone-revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --teacher-checkpoint /hy-tmp/facebook-dinov2-small \
  --skill-sidecar /hy-tmp/libero_cot_rlds/cot_file.json --precision bf16 --backward

CUDA_VISIBLE_DEVICES=0 python scripts/convert_rlds_to_mowe_store.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root /hy-tmp/libero_cot_rlds --checkpoint /hy-tmp/openvla-7b \
  --backbone-revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
  --teacher-checkpoint /hy-tmp/facebook-dinov2-small \
  --skill-sidecar /hy-tmp/libero_cot_rlds/cot_file.json \
  --output /hy-tmp/mowe_store/libero_h16_smoke --encode-batch-size 8 \
  --episodes-per-shard 2 --limit-episodes 2 --device cuda:0 --precision bf16
```

### Result

- RLDS audit: 1,693/1,693 exact episode keys，273,465/273,465 annotation steps，246,377 H=16 valid windows，四 suite `trajectory_ids_contiguous=true`，gripper `non_binary=0`。
- 真实单卡 preflight: `status=preflight_passed`、`mode=real_batch`、`backward=true`，`total_loss=4.308042526245117`；真实原始 7B double-view loader、DINO teacher、RLDS batch 与可训练 MoWE 分支都成功执行，无 CUDA OOM。
- smoke store: conversion 退出码 0，2 episodes/253 frames/221 windows；all checksum verified，actions `[16,7]`、future targets `[4,16,384]`、views `[2,4096]`。partial expected-count mismatch 使 `valid=false` 且 `formal_training_ready=false`，这是 smoke 的预期 fail-closed 行为。
- 新增 OpenVLA identity regression 3 项与 Flow-WAM Torch contracts 15 项通过；`git diff --check` 通过。

### Issues

- 仍未进行 full formal conversion、100-window equivalence、8-rank soak、8-GPU runtime/readiness 或任何正式/长训练；本次单卡结果不能替代它们。
- `pip check` 仍会报告 OFT upstream 的非本次路径依赖（如 diffusers/FastAPI）以及精确 patch-level Torch 版本差异；当前 real loader/preflight 已实际通过，但开始 simulator/上游训练功能前应补齐或使用隔离的 pinned environment。

### Next

- 用户在单卡上执行云手册 7.2 的 full formal conversion（不得带 `--limit-episodes`），再执行 full checksum audit 与 100-window raw/cache equivalence。
- 仅当 formal store、equivalence、8-rank soak 和 8-GPU runtime/readiness 全部通过后，才在 8×A100 上从 Stage 1 step 0 创建正式 50,000-step lineage。

## 2026-07-18 - 4090 formal-store 重新开始交接

### Goal

记录用户从 A100 partial conversion 切换到单卡 4090 48 GiB 的决定，防止同一份正式 feature store 混合不同 GPU 生成的 BF16 特征。

### Changed

- `docs/CLOUD_TRAINING_RUNBOOK.md` 新增 4090 交接节，固定全新 formal output 为 `/hy-tmp/mowe_store/libero_h16_formal_4090`，列出可直接执行的完整 conversion 命令、OOM 降 batch 规则与最终计数。
- 明确已有 `/hy-tmp/mowe_store/libero_h16_formal` 是 A100 partial store（用户报告至少 `feature_episodes=100`），只能作为中断证据保留，不能在 4090 resume，也不能训练。

### Commands Run

```bash
find /hy-tmp/mowe_store -maxdepth 2 -type f \( -name manifest.json -o -name conversion_contract.json \)
git diff --check
```

### Result

- A100 partial formal directory 的 conversion contract 存在；未将其误记为 complete/formal-ready。
- 新 4090 命令不含 `--limit-episodes`，目标完整计数仍为 1,693 episodes、273,465 frames、246,377 windows。

### Issues

- 4090 full conversion 尚未执行，因而没有该硬件上的吞吐、显存峰值或完整 feature-store 证据。

### Next

- 在 4090 执行云手册 3.2 命令；完成后使用 `audit_mowe_feature_store.py --verify-all-checksums` 审计新目录。
- 仅在 formal count/checksum 通过后继续 100-window raw/cache equivalence；8 卡正式训练仍不得提前启动。

归档定位：

```bash
rg -n '^## 2026-|readiness|formal gate|same-stage|CALVIN' \
  docs/history/DEV_LOG_2026-07-08_to_2026-07-17.md
```

## 2026-07-17 - 压缩活动文档并保留完整历史

### Goal

降低新 Codex/agent 会话的默认 token 消耗，同时保留完整设计、命令、验证边界和历史可追溯性。

### Changed

- 将原 1,751 行 `IMPLEMENTATION_PLAN.md` 原样迁移为 `docs/history/IMPLEMENTATION_PLAN_FULL_2026-07-17.md`，并建立较短的根执行合同。
- 将原 1,860 行 `DEV_LOG.md` 原样迁移为 `docs/history/DEV_LOG_2026-07-08_to_2026-07-17.md`，并建立当前状态快照、近期索引和滚动日志。
- 后续启动规则改为读取活动摘要和任务相关风险；历史文档必须先搜索再按章节读取。
- 本次不修改模型、数据、训练、评测代码或配置。

### Commands Run

```bash
wc -lwm CODEX_PROJECT_RULES.md PROJECT_PLAN.md IMPLEMENTATION_PLAN.md ARCHITECTURE_RISKS.md DEV_LOG.md
rg -n '^#{1,4} ' IMPLEMENTATION_PLAN.md DEV_LOG.md ARCHITECTURE_RISKS.md
git diff --check
git check-ignore -v .ipynb_checkpoints/DEV_LOG-checkpoint.md
git status --short
```

### Result

- 三份完整活动文档和两个已过时中文快照均保留在 `docs/history/`；五个归档文件均非空，根文档链接目标均存在。
- `AGENTS.md` 已成为小型自动入口；`.ipynb_checkpoints/` 已加入忽略，避免过期副本进入搜索和 Git 状态。
- 活动入口由原来的 239,676 字符降至 71,015 字符（包含新增 `AGENTS.md`），下降约 70.4%；实际启动还只读取 Project/Risk 的任务相关章节。
- `git diff --check` 通过。本次是文档与 ignore 规则修改，因此未运行模型或训练测试。

### Issues

- 当前工作树还有大量既有未提交代码/配置修改；本次不判断其是否应提交。
- 历史归档中的状态只代表当时证据，不能自动视为当前 tree 已重新验证。

### Next

- 后续代码会话从 `IMPLEMENTATION_PLAN.md` 第 2 节的 LIBERO conversion smoke 继续。
- 若准备提交，应先单独审查当前已有的大量代码/config 变更；本轮未暂存或提交任何文件。

## 2026-07-17 - 主线冻结为原始 OpenVLA-7B backbone

### Goal

将项目与执行合同统一为原始 `openvla/openvla-7b` frozen context backbone，防止后续继续使用或恢复 LIBERO-finetuned OFT 权重产物。

### Changed

- `PROJECT_PLAN.md` 明确 LIBERO/CALVIN 共享同一 immutable 原始 base identity，但分别生成 feature store、action statistics 和 MoWE checkpoint。
- `IMPLEMENTATION_PLAN.md` 将执行链改为固定 revision、双视角 smoke、重建 store，并在 8 卡上从 Stage 1 step 0 建立新 lineage。
- `ARCHITECTURE_RISKS.md` 增加 backbone/cache/checkpoint identity 风险和 fail-closed 门槛。
- 旧 `/hy-tmp/openvla-7b-oft-libero-all` 训练结果保留为历史证据，但不能用于新主线 resume/readiness。

### Commands Run

```bash
rg -n "openvla-7b-oft-libero-all|openvla/openvla-7b|OpenVLA-OFT|backbone|checkpoint" \
  PROJECT_PLAN.md IMPLEMENTATION_PLAN.md ARCHITECTURE_RISKS.md DEV_LOG.md
git diff --check
```

### Result

- 活动项目、执行、风险和状态文档已采用同一原始-backbone合同。
- 本次仅修改文档，未运行模型、数据转换或训练测试；原始 checkpoint 双视角 smoke 和正式 store 重建仍待云端执行。

### Issues

- 当前代码仍主要以 checkpoint 路径字符串绑定 identity；repo revision、processor 和权重 fingerprint 的完整持久化仍需实现/核对。
- 当前工作树存在大量既有未提交修改，本次未覆盖或清理。

### Next

- 下载并固定 `openvla/openvla-7b` revision，完成 primary/wrist BF16 forward smoke。
- 用该 snapshot 生成两 episode smoke store；通过后再开始全量 conversion 和新 Stage 1 lineage。

## 2026-07-17 - 原始 OpenVLA 全链身份校验与 residual 边界

### Goal

把“使用原始 `openvla/openvla-7b`”从文档约定落实为转换、训练、恢复、readiness 和评测都会执行的代码合同，同时修复 residual expert 缺少显式幅值约束的风险。

### Changed

- 新增 `mowe_wam/backbones/openvla_identity.py`：只接受 `openvla/openvla-7b`、完整 40 位 commit revision、`model_type=openvla` 的本地 snapshot，并分别指纹化 config、processor/tokenizer 与全部权重文件；明显的 OFT/LIBERO-finetuned reference 直接拒绝。
- OpenVLA context adapter、LIBERO/CALVIN converter、真实 preflight、raw/cache equivalence 和两个 evaluator 接入相同身份解析；正式 CLI 必须提供 `--backbone-revision`。
- Feature-store runtime 以不可变 identity 为语义合同；相同 snapshot 可以挂载到不同本地路径。checkpoint/resolved config/readiness/evaluator 均持久化并比较完整 identity，旧 path-only checkpoint 不再进入新主线。
- 100-window equivalence 报告新增 `openvla_identity_sha256`，readiness 同时核对 store、benchmark 与 backbone identity。
- `FlowWAMSkillPolicy` 新增 `max_residual_l2=0.5`：训练 target、预测 residual 和 route-mode diagnostics 都按每 timestep 6D L2 投影；记录 target/sample clip fraction，并将该超参数锁入 same-stage resume contract。
- DDP 在新 PyTorch 使用 `forward_sync_buffers`，旧版本回退到 `broadcast_buffers`，保持 forward buffer sync 关闭且消除弃用警告。
- 项目/执行/风险文档标明当前暂缓消融，先完成原始 backbone、8 卡连续训练、LIBERO 与 CALVIN 主流程。

### Commands Run

```bash
/Users/tt/miniconda3/envs/mowe/bin/python -m compileall -q mowe_wam scripts tests
/Users/tt/miniconda3/envs/mowe/bin/python -m unittest discover -s tests -p 'test_*.py'
/Users/tt/miniconda3/envs/mowe/bin/python scripts/preflight_flow_wam_training.py --synthetic --backward
/Users/tt/miniconda3/envs/mowe/bin/python scripts/convert_rlds_to_mowe_store.py --help
/Users/tt/miniconda3/envs/mowe/bin/python scripts/preflight_flow_wam_training.py --help
/Users/tt/miniconda3/envs/mowe/bin/python scripts/eval_libero_temporal_skill.py --help
git diff --check
```

### Result

- `compileall` 通过。
- 62 个单元/合同测试全部通过，包含 2-rank Gloo DDP 参数一致性与 checkpoint resume、feature-store identity、readiness identity mismatch、原始 snapshot 指纹稳定性/权重敏感性和 residual L2 投影。
- synthetic joint forward/backward preflight 通过，total loss 有限；该结果不是实际 7B/GPU/benchmark 证据。
- 三个关键 CLI 均展示 `--backbone-revision`；正式 LIBERO converter 将其标为必填。
- `git diff --check` 通过。

### Issues

- 当前本机没有实际 7B 原始 snapshot 和 A100/A800，因此尚未执行权重加载、双视角 BF16、真实转换或 NCCL 8 卡验证。
- 尚未选定并记录正式 Hugging Face commit；目标服务器必须先固定完整 revision，不能使用 `main`。
- 非 Hugging Face cache 的完整本地权重副本在首次身份解析时需要顺序读取并计算 SHA-256；feature-store 长训练本身不会重复打开 7B 权重。

### Next

- 在目标服务器下载固定 revision 的原始 snapshot，运行双视角 preflight 和两 episode non-formal conversion smoke。
- 用同一 identity 完成 LIBERO formal store、100-window equivalence、8-rank soak 和 8-GPU runtime/readiness，再从 Stage 1 step 0 建立新 lineage。

## 2026-07-17 - 同步 16 步预测与风险门控执行

### Goal

在不采用异步生成、temporal ensemble 或跨 chunk stitching 的前提下，消除固定 1～3 步执行带来的频繁停顿，同时允许可靠的 Pick→Move→Place 边界在同一提交段中连续执行。

### Changed

- 主配置改为 `action_chunk_size=16`、`future_horizons=[1,4,8,16]`、router schedule 16；同步执行默认 8 步、caution 4 步、high-risk 边界前停止。
- 风险门控联合使用 router 归一化 entropy、Top-2 margin、相邻 6D normalized-motion jump 和 residual L2；默认 caution/high 阈值分别为 `0.55/0.75`、`0.20/0.10`、`0.60/0.90`、`0.35/0.45`。
- LIBERO/CALVIN adapter 只入队本轮已提交前缀；队列耗尽后使用最新观测同步重查，旧 16-step chunk 的所有未执行动作均丢弃。
- 训练与评测日志新增执行原因、边界位置/风险值、预测边界跨越率和 ground-truth boundary crossing 诊断；后者是行为统计，不再被命名为 overrun error。
- 项目、实现与风险文档统一为新合同；历史 H=8 formal-store 证据明确不能代替 H=16 重建和审计。

### Commands Run

```bash
conda run -n mowe python scripts/check_flow_wam_forward.py --synthetic --batch-size 2
conda run -n mowe python -m unittest discover -s tests
conda run -n mowe python scripts/eval_libero_temporal_skill.py --queue-smoke
python -m compileall -q mowe_wam scripts tests
git diff --check
```

### Result

- Synthetic joint forward/backward 输出 nominal/router/action shape 均为 16 positions，future latent 为 4 horizons；六个 motor experts 与 ST router 均得到有限梯度，null residual 保持精确为零。
- 专项测试覆盖 confident boundary→8、caution boundary→4、high-risk boundary→边界前 2 步停止。
- Variable-prefix smoke 在 observation 0/8/12 同步 query，证明 8/4/2 步前缀耗尽后才使用最新 observation 重查。
- 本地 63 项完整单元/合同测试通过；`compileall` 与 `git diff --check` 通过。

### Issues

- 风险阈值目前是 normalized-action 空间的工程初值，尚未用真实 LIBERO/CALVIN rollout 校准；正式实验必须报告 8/4/high-risk 分布、成功率、延迟和预测/真实边界跨越率。
- 旧 H=8 feature store、cache、checkpoint 和 readiness report 与新的 H=16/action-chunk-16 合同不兼容。
- 本轮没有实际 7B GPU、LIBERO simulator 或 CALVIN 官方评测证据。

### Next

- 使用固定原始 OpenVLA revision 重跑两 episode H=16 conversion smoke，再重建 LIBERO formal feature store 与 100-window equivalence。
- 在 one-task simulator smoke 中先检查 execution-reason histogram、每秒 query 数和控制平滑性，再决定是否仅微调四组风险阈值。

## 2026-07-17 - 单卡与 8 卡云训练全流程合同

### Goal

给目标云服务器提供从环境搭建、原始 backbone/RLDS 审计、feature-store 构建到 Stage 1/2/3 与 LIBERO 评测的可直接执行手册，并让单卡调试与 8 卡正式训练保持相同 effective global batch。

### Changed

- 新增 `docs/CLOUD_TRAINING_RUNBOOK.md`，所有关键命令显式传入 data/store、原始 OpenVLA snapshot/revision、DINO teacher、skill config、stage predecessor、output directory、max/stop steps 与 readiness report；每一阶段附 fail-closed 达标输出。
- 新增三份单卡 feature-store 配置：每卡 batch 1、accumulation 8、每阶段最多 100 个未认证 optimizer steps，只用于工程调试。
- 新增 `ddp8_nominal_flow_wam_feature_store_formal.yaml`：Stage 1 从第一步固定 `max_steps=50000`，与 Stage 2/3 的 8 卡 effective global batch 8 保持一致。
- Stage 2/3 feature-store config 的 teacher 路径改为 `TBD`，由 store identity 和显式 CLI 路径解析，避免不同服务器绝对路径导致伪 mismatch。
- RLDS audit/converter/preflight 的默认窗口合同统一为 H=16、chunk=16；真实 preflight 现在显式使用 CLI teacher checkpoint 和 immutable backbone revision。
- `IMPLEMENTATION_PLAN.md` 的 Stage 1 阶梯与活动配置导航同步为正式 50,000-step lineage，并链接云端手册。

### Commands Run

```bash
conda run -n mowe python -m unittest discover -s tests
conda run -n mowe python -m compileall -q mowe_wam scripts
conda run -n mowe python scripts/pretrain_nominal_flow_wam.py --help
conda run -n mowe python scripts/warmstart_skill_flow_experts.py --help
conda run -n mowe python scripts/train_flow_wam_skill_moe.py --help
git diff --check
```

### Result

- 6 份新增/活动 feature-store config 全部通过 inheritance load 和 `validate_flow_config`；单卡与 8 卡 effective global batch 均为 8，H=16/chunk=16 不漂移。
- 63 个单元/合同测试全部通过；`compileall`、三个 stage CLI 参数核对和 `git diff --check` 通过。
- 手册引用的 15 个脚本和 9 个配置路径全部存在。

### Issues

- 当前超参数只可称为 frozen v1 强默认值，尚无新 H=16 原始-backbone 真实长训练、8 卡 NCCL 或 simulator success-rate 证据，不能声称经验最优。
- 当前本机无法替代云端完成正式 store、100-window equivalence、8-rank soak、8-GPU runtime/readiness 或三阶段长训练。

### Next

- 在目标云服务器按 `docs/CLOUD_TRAINING_RUNBOOK.md` 顺序执行，先固定 OpenVLA revision 并完成真实双视角 preflight 与两 episode smoke store。
- 只有每一节达标条件通过后才进入下一阶段；首次失败时保留对应 report/log，停在该层定位。

## 2026-07-18 - MTP 单入口 8 卡正式训练编排器

### Goal

为只能提交一个 Python 文件的 MTP/云平台提供跨服务器可配置、可审计、可恢复的完整 LIBERO 8 卡 Stage 1→2→3 正式训练入口。

### Changed

- 将根 `start_mtp.py` 实现为 fail-closed 编排器：路径/GPU/cgroup 预检、RLDS/skill config、formal store 全 checksum/8-rank 审计、100-window equivalence、8-rank soak、8-GPU runtime、checkpoint-bound readiness 与三阶段训练全部串联。
- Stage 1 固定执行 `0→2→25→100→1000→50000` 并在 pilot 检查 future predictor 相对 copy-current；Stage 2 执行 `0→100→50000` 并在进入 Stage 3 前执行 route/expert gate；Stage 3 执行 `0→100→30000` 并输出训练质量摘要。
- 增加 checkpoint/state 幂等恢复、进程组信号转发、run-local resolved configs、MTP `--run_root_dir/--run_id` 兼容参数和非 8 卡可执行的 `--dry-run`。
- feature-store runtime/readiness 允许 DINO 在新服务器重挂载到不同绝对路径，同时保留 store 中原路径为 provenance；新路径仍必须重新通过 100-window equivalence。
- 新增 `docs/MTP_ONE_CLICK_TRAINING.md` 和 launcher/remount 回归测试。

### Commands Run

```bash
PYTHONPATH=tests:. python -m unittest \
  tests.test_start_mtp tests.test_feature_store tests.test_long_run_readiness \
  tests.test_feature_equivalence_audit tests.test_flow_contracts \
  tests.test_flow_torch_contracts
python -m compileall -q mowe_wam scripts tests start_mtp.py
python start_mtp.py --help
git diff --check
```

### Result

- 最终 100-window mask-aware 报告已核对为 `passed=true`，100/100 matched、无 missing、mask 与 feature/output/loss gates 全部通过。
- 单卡 Stage 1/2/3 checkpoint 均已核对到 step 100，作为工程验证保留但不会被 launcher 用作正式 Stage 1 初始化。
- launcher dry-run 能完整展开静态审计、节点证据、readiness、三阶段 init/resume 链；相关测试、compileall、help 和 whitespace 检查通过。

### Issues

- 尚未在真实 8×A100/A800 节点执行完整 launcher，因此不能声称 NCCL、cgroup 资源门禁或三阶段长训练已通过。
- launcher 完成训练后仍需单独执行 LIBERO simulator smoke 与四 suite 正式评测；训练日志不能替代 success rate。

### Next

- 在目标服务器替换所有绝对路径，先运行文档中的 `--dry-run` 命令；核对无误后使用相同 `run-root-dir`、`run-id` 去掉 `--dry-run` 启动正式训练。
- 首次真实失败时保留 `<run-root-dir>/<run-id>/launcher_state.json`、对应 report/log 和最新 checkpoint，在同一门槛定位，不放宽合同绕过。

## 2026-07-18 - MTP 无系统权限降级运行模式

### Goal

支持无法读取 `/proc`、cgroup 或宿主机资源指标且无法取得 sudo/平台权限的 8 卡容器，同时避免仅删除入口检查后在 soak/readiness/训练内部继续失败。

### Changed

- `start_mtp.py` 新增 `--disable-system-monitoring`（兼容别名 `--disable-cgroup-monitoring`），并统一传递到 run-local config、8-rank soak、8-GPU runtime、readiness 和所有 torchrun 子进程。
- 降级模式不读取 `/proc/self/status`、boot ID、cgroup membership、宿主机内存上限或 OOM events；soak/runtime/readiness 报告显式记录 monitoring disabled/degraded。
- 仍保留 8 GPU 可见性、NCCL rank/GPU 绑定、GPU memory guard、store checksum/equivalence、checkpoint/resume 和 Stage 1/2 质量门槛。
- 使用文档增加无系统权限平台的启动说明和证据边界。

### Commands Run

```bash
PYTHONPATH=tests:. python -m unittest tests.test_start_mtp tests.test_long_run_readiness
python -m compileall -q mowe_wam scripts tests start_mtp.py
python start_mtp.py --help
git diff --check
```

### Result

- launcher/degraded readiness 专项 8 项测试通过；compileall、help 与 whitespace 检查通过。
- dry-run 已验证生成配置为 `require_cgroup_metrics=false`，且 soak/runtime/readiness 命令收到对应降级参数。

### Issues

- 该模式无法检测容器 CPU RAM 接近上限或宿主机 OOM-kill 事件，资源证据弱于标准 cgroup 模式；真实 8 卡运行仍未执行。

### Next

- 在无权限服务器用同一命令追加 `--disable-system-monitoring` 重新 dry-run，再正式启动；同时用 MTP 控制台外部观察系统 RAM、GPU memory 和进程被杀事件。

## 2026-07-19 - 跨服务器 equivalence 精确 source identity 匹配

### Goal

修复 formal store 与重新挂载的同 fingerprint RLDS 在 raw/cache 100-window audit 中出现少量 image-derived `episode_id` missing pair 的问题。

### Changed

- raw LIBERO window 现在保留 sidecar overlay 已注入的 `source_file_key` 与 `source_traj_index`。
- feature-store window 暴露 converter 已存储的同一 provenance；equivalence audit 优先以 source identity 加 step 精确匹配，旧 store 才回退 image-derived episode ID，并在报告中写出匹配统计/缺失 source identity。

### Commands Run

```bash
PYTHONPATH=tests:. python -m unittest tests.test_feature_equivalence_audit tests.test_start_mtp
python -m compileall -q mowe_wam scripts tests start_mtp.py
git diff --check
```

### Result

- 6 项 equivalence/launcher 专项测试通过，新增 source-identity helper 回归覆盖。
- 用户服务器原报告的 95 个已比较窗口全部在 feature/output/loss gate 内；5 个 missing pair 是唯一失败原因。新 matcher 不放宽数值或 100-window 完整性要求。

### Issues

- 需要先将上述代码同步到目标 8 卡服务器，并在该节点实跑一次 equivalence；当前没有该节点的新报告。

### Next

- 先确认 `episodes.jsonl` 有 1,693 条 source provenance，再用单卡重跑同 seed 的 100-window audit；通过后以同一 run-id 重启 launcher。

## 2026-07-19 - 修复 Stage 1 第二步 DDP unused parameters

### Goal

处理真实 8 卡 Stage 1 在成功完成第一个 optimizer step 后，于第二次 forward 前报 `Expected to have finished reduction` 的错误。

### Changed

- 根据正式配置的 DDP 参数索引，确认 `52/53` 为 nominal trunk 的 `token_condition_projection`，该分支只供 residual expert 使用，nominal head 永不传 token condition，因此始终冻结。
- 确认 `168/169` 为 `world_model.route_world_head`；Stage 1 router 冻结且无 route loss，因此 Stage 1 冻结该 head，Stage 2/3 自动重新启用。
- 未开启全局 `find_unused_parameters=true`，避免隐藏未来新的断梯度问题和增加 DDP 遍历开销。
- 增加 Stage 1 全部 trainable 参数必须获得梯度、Stage 2/3 route-world head 恢复启用的合同测试。

### Commands Run

```bash
PYTHONPATH=tests:. python -m unittest tests.test_feature_equivalence_audit tests.test_start_mtp
python -m compileall -q mowe_wam scripts tests start_mtp.py
git diff --check
```

### Result

- 用户真实日志证明 8 卡 NCCL、effective batch 8、Stage 1 第一步有限 loss/gradients 均正常；失败参数在 8 ranks 完全一致，排除随机 rank 数据差异。
- 正式模型参数索引映射精确得到 `52/53=token_condition_projection`、`168/169=route_world_head`；equivalence/launcher 6 项专项测试、compileall 和 diff check 通过。

### Issues

- 当前本地容器的 CPU multiprocessing/large synthetic backward 测试被环境级进程阻塞提前终止，尚未替代目标节点真实 8 卡 step 0→2 重跑证据。

### Next

- 将修复同步到目标节点，备份或清理未到 step 2 的 Stage 1 失败目录，以相同 run-id 重启 launcher；必须先看到 step 2 checkpoint 正常写出，再由 launcher 自动继续 2→25。

## 日志维护规则

- 每次代码或核心文档发生有意义修改，更新顶部快照并在文件末尾追加一条简短记录。
- 每条保留 `Goal / Changed / Commands Run / Result / Issues / Next`；长日志写入文件并只记录路径、摘要和 fingerprint。
- 根文件最多保留最近 10 条详细记录。超过后，把最旧条目追加到按月归档，并在“最近关键变更索引”保留一句摘要。
- 不修改已经归档的实验结果；需要勘误时追加新记录并指向旧条目。

## 2026-07-19 - 三阶段改为 validation loss 早停

### Goal

按当前实验策略取消 Stage 1 copy-current、Stage 2 route/expert 和 Stage 3 推荐指标门槛，只按验证损失平台期或 50000 步结束各阶段。

### Changed

- 三阶段统一最大 50000 optimizer steps；默认每 500 步验证，至少训练 5000 步，`total_loss` 连续 5 次未改善 `1e-4` 时早停。
- 早停状态从 `validation_log.jsonl` 按 step 重建，same-stage resume 不重复消耗 patience；8 rank 广播停止决定并共同保存最新 checkpoint。
- launcher 接受经过合同校验的提前 checkpoint，并自动初始化下一阶段；删除旧 Stage 1/2 promotion gate 调用及 Stage 3 quality summary。
- 数据完整性、NaN/Inf、checkpoint/resume、CUDA/NCCL、显存与现有 smoke 正确性检查继续保留。

### Commands Run

```bash
python -m compileall -q start_mtp.py mowe_wam/training/flow_runtime.py tests/test_start_mtp.py tests/test_flow_torch_contracts.py
PYTHONPATH=tests:. python -m unittest tests.test_start_mtp tests.test_flow_torch_contracts.FlowTorchTests.test_validation_loss_early_stopping_is_resume_stable
```

### Result

- 专项 launcher 与 loss 早停测试通过。

### Issues

- loss-only 阶段完成不再证明 copy-current、router/expert 或 boundary 机制质量，最终 checkpoint 仍需 LIBERO simulator 评测。

### Next

- 在目标 8 卡节点用已有 Stage 1 step 1000 checkpoint 恢复，确认首个 validation 周期和 `early_stopping.json` 行为。

## 2026-07-18 - 等价性审计拆分连续输出与 gripper 诊断

### Goal

处理 mask-aware equivalence 重跑后仅由临界 binary gripper 翻转导致的报告失败。

### Changed

- 连续模型输出与 `gripper_logits` 进入 output gate；拼接后的 binary `actions` 与 `gripper_accuracy` 保留为离散诊断。
- loss gate 排除 `gripper_accuracy`，继续严格检查所有连续 scalar losses。
- readiness 现在要求 `max_output_gate_error` 和 `max_loss_gate_error` 分别不超过对应容差。

### Commands Run

```bash
PYTHONPATH=tests:. python -m unittest tests.test_feature_equivalence_audit tests.test_long_run_readiness tests.test_feature_store
python -m compileall -q mowe_wam scripts tests
git diff --check
```

### Result

- 17 项相关测试通过。
- 上一次报告的 feature gate 已通过；其 `max_output_abs_error=1.0` 和 `max_loss_abs_error=0.0625` 来自两个单步 gripper threshold flip，不代表连续缓存/模型输出错位。

### Issues

- 新 output/loss gate 合同尚未在 GPU 上重新生成 100-window 报告。

### Next

- 用同一 store、seed、sidecar 和容差重新运行 equivalence audit；新报告必须含 `max_output_gate_error`、`max_loss_gate_error` 并为 `passed=true`。

## 2026-07-18 - 修正 LIBERO raw/cache 等价性审计语义

### Goal

修复首次真实 100-window equivalence 将 masked padding 和 BF16/FP16 单点极值误作训练语义不一致的问题，同时保持 readiness fail closed。

### Changed

- `audit_feature_store_equivalence.py` 只在 raw/cache mask 同时有效的位置比较 short/long history；mask 本身必须完全一致。
- OpenVLA/language 使用 mean absolute 与 mean cosine distance；DINO 使用与 world-model target 一致的 Smooth-L1 与 mean cosine distance。所有 max-abs 仍保留为诊断值。
- readiness 明确要求 `mask_aware_training_metric_v1`、`masks_match=true` 和 `max_feature_gate_error <= feature_atol`；旧报告不能用于签发长训练 readiness。
- 新增 padding 排除与 DINO metric-selection 回归测试，并更新云端 runbook。

### Commands Run

```bash
PYTHONPATH=tests:. python -m unittest tests.test_feature_equivalence_audit tests.test_long_run_readiness
python -m compileall -q mowe_wam scripts tests
git diff --check
```

### Result

- 专项 equivalence 与 readiness 合同测试通过；compileall 和 diff whitespace 检查通过。
- 旧真实报告仍保持 `passed=false`，没有被修改或冒充新证据。

### Issues

- 修正后的 100-window GPU 审计尚未重跑，当前仍不能声称 equivalence 门槛通过。

### Next

- 使用相同 store、seed 和容差重跑 `audit_feature_store_equivalence.py`，确认新报告 `passed=true`、`masks_match=true` 后再进入下一门槛。
