# MoWE-WAM 实现计划（2026-07-17 完整归档）

> 本文件保存文档瘦身前的完整实现规格、历史任务说明和命令。当前执行入口为仓库根目录的 `IMPLEMENTATION_PLAN.md`；除非根文档明确要求追溯，本文件不应在新会话启动时全文读取。

更新时间：2026-07-17（单机多卡 DDP 代码已实现；真实 8 卡与节点内存门禁待云端验证）

本文档是 Codex 编写项目代码时的主要参考。研究主张以 `PROJECT_PLAN.md` 为准，风险边界以 `ARCHITECTURE_RISKS.md` 为准，真实进展以 `DEV_LOG.md` 为准。

## 0. 当前状态与迁移决策

### 0.1 当前实现状态

仓库当前已有：

- `LiberoSequenceDataset`、multi-suite episodic windows、联合 action bounds 与 latent collator；
- OpenVLA-OFT non-leaky visual/language context API、instruction token cache 与冻结视觉 pooled-feature LRU；
- 冻结 DINOv2 `VisualTargetEncoder` 与按 episode/timestep 去重、分片、延迟读取的 teacher cache；
- `MultiScaleMemoryEncoder`、`NominalActionHead`、6 层 `LatentWorldModel`；
- legacy `FutureGroundedRouter`、5 个 legacy `ResidualActionExperts` 与完整 `LatentWAMPolicy`；
- Stage 1/Stage 2 训练入口、AMP、checkpoint/resume、schedule、JSONL 机制日志、preflight 与分析脚本；
- 旧 `WorldPredicateHead`、`WorldTransitionHead`、predicate/event scripts，作为 legacy baseline 保留。
- 独立 `flow_wam_skill_moe` 主线：6D rectified-flow nominal head、binary gripper head、`[B,8,128]` world tokens、`[B,8,7]` temporal router、六 motor residual-flow heads与无参数 null bypass；
- `episode_metadata.file_path + deterministic global trajectory index + timestep` TensorFlow sidecar overlay、三阶段 configs/CLI/runtime、阶段 checkpoint 守卫、variable-prefix queue、flow preflight、日志分析与 contract tests。
- LIBERO `primary+wrist` 双视角 sequence/online-memory/evaluator 契约、OpenVLA-OFT ordered multi-image 编码、1,065,344 参数的语言条件 scalar View Fusion、primary-only DINO target，以及 `flow_wam_skill_components_v2` checkpoint view contract。
- 三个 Flow-WAM stage 共用的原生 PyTorch DDP runtime：`torchrun` rank/device 初始化、episode 级 RLDS 分片、梯度累积 `no_sync()`、rank-0-only 日志/验证/checkpoint、跨 rank 指标与资源汇总，以及受控 world-size checkpoint 迁移。
- 长训练热路径使用的 `mowe_feature_store_v1`、LeRobot 风格 canonical archive、episode-aware DDP sampler、可恢复 converter、expected/actual completion gate 与 continuous memory soak。
- 第二主评测 CALVIN 的独立 action/policy adapter、ABC-only dataset audit/converter、三阶段 DDP8 feature-store 配置和官方 evaluator bridge；这些代码通过本地 contract test，但真实 CALVIN 数据、训练和 simulator 仍未验证。

deterministic action-regression + Top-2 residual MoE 已降为 `regression_residual_moe` baseline，新版 flow 路径以独立类和训练入口实现，未覆盖或删除旧 baseline。CPU synthetic forward/backward 已验证 nominal/oracle/ST/predicted shapes、六 expert 梯度、ST action-to-router 梯度、null-zero、1～3 步 prefix、checkpoint/resume；legacy lightweight smoke 仍通过。真实四-suite RLDS 结构审计、sidecar exact lookup、联合 6D motion statistics、gripper canonicalization、完整 primary-view DINO cache 和双视角真实训练链路已通过。云端 RTX 4090 已完成 Stage 1 step 0→100（`max_steps=1000` scheduler）及恢复验证，Stage 2 one-step/25-step oracle coverage gate 也已完成；但 future predictor 仍未优于 copy-current。单机多卡 DDP 已有代码、配置和本地 2-rank Gloo 参数/checkpoint/rank-0 I/O contract test，但尚未在 8 张 A100/A800 上运行 NCCL、真实数据或长期压力门禁，不能视为真实 8 卡通过。

### 0.2 迁移规则

- 不删除旧 predicate/event 模块；它们保留为 `legacy_predicate` baseline。
- 不继续扩展旧 `WorldTransitionHead` 为主模型。
- 新主路线使用独立的 `flow_wam_skill_moe` variant、数据契约、config 和训练入口；当前 `latent_wam_residual_moe` 作为 regression baseline 保留。
- 默认训练入口已实现为 `scripts/pretrain_nominal_flow_wam.py`、`scripts/warmstart_skill_flow_experts.py` 与 `scripts/train_flow_wam_skill_moe.py`；只有完成真实 one-batch optimizer step 后，才可称对应真实训练路径通过。
- 不修改 `external/openvla-oft/`。
- 新主路线不得要求 simulator state、predicate JSONL 或 transition label cache；仅允许训练期、可审计的逐位置 `expert_skill_labels`，且标签不得由未来帧、test episode、simulator state 或整段 CoT 部署输入构造。

### 0.3 目标最小闭环

```text
LIBERO multi-suite episodic windows
    → non-leaky OpenVLA visual/language context
    → multi-scale memory
    → nominal flow sample A0
    → nominal-action-conditioned Latent WAM
    → future visual latent / delta
    → per-token future-grounded skill schedule [B, 8, 7]
    → one shared residual flow solve with per-token expert adapters/heads
    → final 8-step candidate chunk
    → execute current-skill prefix, capped at 3 steps
```

### 0.4 双 Benchmark 交付顺序

1. 先在 LIBERO feature store 上通过 8 卡 2/25/100-step 与 continuous memory soak，完成 Stage 1→2→3；只有 Stage 1 future predictor 相对 copy-current 明显改善后才继续后续长训练。
2. 用 Stage 3 LIBERO checkpoint 完成 one-task smoke 和四-suite 可恢复正式评测，建立基础成功率与机制消融主表。
3. 单独审计并转换 CALVIN `task_ABC_D/training`，从 ABC 数据生成 action statistics、skill taxonomy 和 formal feature store；不得复用 LIBERO q01/q99 或把 D/validation 写入训练产物。
4. 使用 CALVIN 专用 Stage 1→2→3 配置完成小步门禁与连续训练，再由官方 evaluator 在 D 环境运行 1,000 条五子任务 sequence。
5. LIBERO 与 CALVIN 分别绑定 dataset fingerprint、backbone identifier、action contract、resolved config、checkpoint 和 evaluator commit；结果表不混用 checkpoint，也不以离线 window accuracy 替代 simulator 指标。

## 1. 默认参数

```yaml
data:
  observation_views: [primary, wrist]
  dataset_names:
    - libero_spatial_no_noops
    - libero_object_no_noops
    - libero_goal_no_noops
    - libero_10_no_noops
  history_length: 8
  long_memory_slots: 4
  future_horizons: [1, 4, 8]
  action_dim: 7
  motion_dim: 6
  gripper_dim: 1
  action_chunk_size: 8
  # Aligns with external/openvla-oft LIBERO NUM_ACTIONS_CHUNK=8.
  # This is the prediction horizon, not a requirement to execute all 8 steps.

skill_experts:
  num_motor_experts: 6
  num_routes: 7
  motor_names: [pick_grasp, place_release, move_transport, open_close, turn_rotate, push_pull]
  null_route: null_finish
  unknown_label: -1
  source: cot_final_directive_leading_verb_v1
  deployment_uses_labels: false
  require_all_classes: true
  supervision: per_timestep
  assume_sidecar_timestep_aligned: true
  boundary_label_policy: direct_per_timestep_no_extra_mask
  boundary_policy: execute_before_first_predicted_boundary

flow:
  dimensions: motion_only
  formulation: conditional_rectified_flow
  solver: euler
  num_inference_steps: 4
  deterministic_warmstart_seed: 7
  share_residual_trunk_across_experts: true
  share_nominal_and_residual_weights: false

teacher:
  name: dinov2_vits14
  frozen: true
  spatial_tokens: 16
  target_dim: 384
  inference_enabled: false
  target_views: [primary]

view_fusion:
  type: language_conditioned_scalar_attention
  view_order: [primary, wrist]
  num_views: 2
  score_hidden_dim: 128
  initialization: uniform_zero_score

world_model:
  hidden_dim: 512
  layers: 6
  heads: 8
  mlp_ratio: 4
  dropout: 0.0
  route_world_steps: 8
  route_world_dim: 128
  predict_uncertainty: false

router:
  num_routes: 7
  schedule_length: 8
  top_k: 1
  joint_estimator: straight_through_gumbel_softmax
  gumbel_temperature_start: 1.0
  gumbel_temperature_end: 0.1
  oracle_route_warmup_ratio: 1.0
  predicted_route_start_ratio: 0.20
  predicted_route_end_ratio: 0.70

execution:
  prediction_horizon: 8
  min_steps: 1
  max_steps: 3
  stop_before_first_skill_change: true

backbone:
  freeze_backbone: true
  context_source: pre_action_context
  num_images_in_input: 2
  lora_enabled: false

loss_weights:
  flow_nominal: 1.0
  flow_expert: 1.0
  gripper_bce: 1.0
  route: 1.0
  world: 1.0
  delta: 0.5
  load_balance: 0.01
  residual: 0.001
  endpoint: 0.0
```

## 2. 目标目录结构

只列出新版需要新增或重点修改的文件；已有 legacy 文件继续保留。

```text
mowe_wam/
├── backbones/
│   ├── openvla_oft_adapter.py          # 修改：non-leaky context API
│   └── visual_target_encoder.py        # 新增：冻结 DINO/VAE teacher
├── data/
│   ├── libero_predicate_dataset.py     # 保留 legacy
│   ├── libero_sequence_dataset.py      # 新增：episode window + suite mixture
│   ├── latent_wam_collator.py           # 新增：multi-scale batch collator
│   └── expert_skill_labels.py           # 新增：训练期谓语标签适配/审计
├── memory/
│   ├── event_memory.py                 # 保留 legacy
│   └── multiscale_memory.py            # 新增：short/long latent memory
├── models/
│   ├── world_transition.py             # 保留 legacy
│   ├── predictive_router.py            # 保留 legacy
│   ├── action_flow.py                  # 新增：shared flow trunk, time/noise, solver
│   ├── nominal_action_head.py          # 修改：nominal flow velocity head
│   ├── latent_world_model.py           # 新增
│   ├── future_router.py                # 新增
│   ├── residual_experts.py             # 修改：label-selected residual flow heads
│   ├── view_fusion.py                   # 新增：语言条件双视角 scalar attention
│   └── latent_wam_policy.py             # 新增：完整主模型 wrapper
├── training/
│   ├── losses.py                       # 保留 legacy API
│   ├── latent_losses.py                # 修改：flow + route + world losses
│   └── schedules.py                    # 修改：action condition / oracle-to-predicted routing
├── analysis/
│   ├── future_prediction.py            # 新增
│   └── routing_diagnostics.py          # 新增
└── evaluation/
    └── libero_temporal_policy.py        # 新增：variable-prefix action queue adapter

configs/mowe_wam/
├── dataset_libero_mixture.yaml          # 新增
├── skill_experts.yaml                   # 新增：六 motor + null taxonomy/source/coverage gate
├── train_nominal_flow_wam.yaml          # 新增
├── warmstart_skill_flow_experts.yaml    # 新增
├── train_flow_wam_skill_moe.yaml        # 新增，最终主 config
└── ablations/
    ├── latent_history_only.yaml
    ├── latent_behavior_prior.yaml
    ├── latent_nominal_conditioned.yaml
    ├── latent_no_memory.yaml
    ├── latent_short_memory_only.yaml
    ├── flow_dense_residual.yaml
    ├── flow_oracle_skill.yaml
    ├── flow_predicted_skill.yaml
    ├── flow_shuffled_skill.yaml
    ├── latent_future_shuffle.yaml
    └── latent_copy_current.yaml

scripts/
├── inspect_latent_sequence_dataset.py   # 新增
├── cache_visual_targets.py              # 新增，可选
├── inspect_skill_experts.py              # 新增
├── check_flow_wam_forward.py             # 新增
├── pretrain_nominal_flow_wam.py          # 新增
├── warmstart_skill_flow_experts.py       # 新增
├── train_flow_wam_skill_moe.py           # 新增，最终主入口
├── preflight_flow_wam_training.py        # 新增
├── analyze_flow_wam_logs.py              # 新增
└── eval_libero_temporal_skill.py          # 新增：每 1～3 步重新查询 policy
```

## 3. 数据契约

### 3.1 Sequence Sample

`LiberoSequenceDataset` 必须在 raw episode/trajectory 层先构造窗口，再调用图像预处理；禁止从已打乱的单步 samples 伪造历史。

```python
sample = {
    "episode_id": str,
    "step_id": int,
    "dataset_name": str,
    "language": str,

    "history_pixel_values_primary": Tensor, # [K-1,C,H,W]
    "history_pixel_values_wrist": Tensor,   # [K-1,C,H,W]
    "pixel_values_primary": Tensor,         # [C,H,W]
    "pixel_values_wrist": Tensor,           # [C,H,W]
    "history_mask": Tensor,         # [K]
    "history_actions": Tensor,      # [K-1, action_dim]

    "long_history_pixel_values_primary": Tensor, # [M,C,H,W]
    "long_history_pixel_values_wrist": Tensor,   # [M,C,H,W]
    "long_history_actions": Tensor, # [M, action_dim]
    "long_history_mask": Tensor,    # [M]

    "current_raw_pixel_values": Tensor, # primary-only DINO input
    "future_raw_pixel_values": Tensor,  # primary-only [num_horizons,3,H,W]
    "future_mask": Tensor,          # [num_horizons]
    "future_horizons": Tensor,      # [num_horizons]

    "target_actions": Tensor,       # [chunk_size, action_dim]
    "target_motion": Tensor,        # [chunk_size, 6], normalized relative motion
    "target_gripper": Tensor,       # [chunk_size, 1], canonical absolute binary target
    "proprio": Tensor | None,       # current generic robot state only

    # Training-only supervision. Never passed to deployment wrapper or memory.
    "expert_skill_labels": Tensor,  # [chunk_size] int64 in [0, 6], unknown/padding = -1
    "expert_skill_mask": Tensor,    # [chunk_size] bool; true independently per valid timestep
    "expert_label_source": list[str], # length chunk_size; raw_annotation | audited_mapping | unknown
}
```

默认时间定义：

```text
current observation: I_t
known executed actions: a_(t-K+1:t-1)
target/nominal chunk: a_(t:t+7)
future images: I_(t+1), I_(t+4), I_(t+8)
```

Action contract：

```text
target_motion  = target_actions[..., :6]   # relative, normalized, flow target
target_gripper = target_actions[..., 6:7]  # absolute binary, no motion normalization
```

训练内部 gripper 使用 OpenVLA-OFT 数据侧 canonical `0/1` 语义；环境需要的 `[-1,+1]` 和 OpenVLA sign inversion 只在 LIBERO evaluation adapter 内处理。禁止将环境侧 gripper 值送入 BCE target，也禁止对 gripper 做 residual addition。

### 3.2 Episode 边界

- 所有 history、long history、future 和 target action 必须来自同一个 episode。
- `primary` 与 `wrist` 必须来自同一 timestep，分别预处理、按固定顺序编码；任一视角缺失即明确失败，不静默复制另一个视角。
- episode 起点不足的 history 用 padding + mask；默认不复制第一帧伪装真实历史。
- episode 终点不足的 future/action chunk 默认跳过；可配置 padding，但必须有有效 mask。
- `episode_id` 只用于索引和 cache，不作为模型输入。
- 不再要求外部 transition label JSONL。
- 第一版按配置假设 sidecar timestep 与 action timestep 已对齐，`expert_skill_labels[j]` 直接由 `t+j` annotation 最后一句首谓语构造。不得用 chunk 起点标签复制填满 8 个位置，不得扫描整段 CoT 猜测，也不得使用 object pose、contact/collision state 或 test episode。该对齐是显式设计假设，不实现额外 join audit gate，也不得写成已经实证验证。
- 不做边界 `±1` mask、soft target 或 label debounce。跨越 Pick→Move、Move→Place 等边界的 chunk 保留训练；只对 unknown/padding 位置置 `-1/false`。

### 3.3 Long-memory Sampling

第一版避免 stateful dataloader，使用可复现的 sparse prefix sampling：

```python
def sample_long_history_indices(step_id: int, slots: int = 4) -> list[int]:
    """从 [0, step_id-history_length) 中按时间均匀采样，缺失位置用 mask。"""
```

Instruction 始终作为独立 task token；long history 只提供 episode 早期状态 summary，不引入 subtask label。

### 3.4 Multi-suite Sampling

- 支持四 suite 按配置权重混合，默认按有效 window 数量采样。
- 保存每个 suite 的样本数和有效窗口数。
- 只为前 6 维 relative motion 计算联合 LIBERO normalization statistics；第 7 维 gripper 保持 canonical absolute binary 表示，不进入 motion statistics。
- `dataset_name` 只用于日志和 normalization 检查，默认不输入 router。

### 3.5 Verb-seeded skill 数据审计

在实现任何 flow expert 前，新增 `mowe_wam/data/expert_skill_labels.py` 与 `scripts/inspect_skill_experts.py`。默认 taxonomy 是配置项而非硬编码：

```yaml
pick_grasp: 0
place_release: 1
move_transport: 2
open_close: 3
turn_rotate: 4
push_pull: 5
null_finish: 6
unknown: -1
```

首谓语映射表：

```text
pick/grasp/grab/lift                       → pick_grasp
place/put/release/set/stack                → place_release
move/carry/bring/position/align/approach   → move_transport
open/close                                → open_close
turn/rotate                               → turn_rotate
push/pull                                 → push_pull
finish/stop/done/check/hold                → null_finish
otherwise                                 → unknown (-1)
```

当前 `cot_final_directive_leading_verb_v1` 实现对 `libero_cot_rlds` 273,465 条 directive 的离线审计为：`place_release` 92,320、`pick_grasp` 81,137、`null_finish` 49,276、`move_transport` 30,112、`open_close` 8,974、`turn_rotate` 5,395、`push_pull` 4,262、unknown 1,989；该结果取代早期临时解析器快照。Labeler 只保留最后一句和 leading verb，并把版本、映射、输入字段、sidecar fingerprint 记录到 dataset fingerprint。overlay 必须在 upstream standardization 前读取 `traj_metadata.episode_metadata.file_path`，并用 `file_path + deterministic global trajectory index + timestep` 精确查找；为复现 dlimp 的 trajectory index，TFDS 读取固定为 `shuffle_files=false`、`num_parallel_reads=16`。真实四-suite 审计已验证 1,693/1,693 个 episode key 与 273,465/273,465 个 timestep key 存在，unknown 为 1,989；这只是结构关联证据，`alignment_verified=false` 仍表示没有把语义时序对齐当作实证结论。第一版不增加边界 `±1` mask、soft target、debounce 或额外 alignment gate。任何 motor expert 零覆盖或未知率异常时停止 warm-start。

`null_finish` 从第一版起就是无参数 residual bypass，而不是 residual expert：其 route probability 仍参与 CE，但 6D motion residual velocity/state/endpoint 必须精确为零；最终仍使用 nominal motion 与 nominal gripper。它不表示 stop、done 或 episode termination，也不得以一个可学习 finish head 代替该契约。

## 4. 模型接口

### 4.1 OpenVLA Adapter

修改：`mowe_wam/backbones/openvla_oft_adapter.py`

新增 API：

```python
class OpenVLAOFTAdapter(nn.Module):
    def encode_pooled_views(self, primary_pixel_values, wrist_pixel_values):
        """按 [primary,wrist] ordered multi-image contract 返回 [B,2,D]。"""

    def encode_images(self, pixel_values, *, return_tokens: bool = True):
        """只运行冻结视觉分支，不能读取 labels/action target。"""

    def encode_language(self, language_or_input_ids, attention_mask=None):
        """返回 instruction tokens；同一 episode 可缓存。"""

    def extract_context_features(self, batch: dict) -> dict[str, Tensor]:
        """
        返回 current/history 的 `[B,...,2,D]` ordered view features 与 language tokens。
        禁止传 labels；禁止使用 last_action_hidden。
        """
```

迁移约束：

- `extract_features()` 保留给 legacy path。
- 新 `latent_wam` variant 只能调用 `extract_context_features()`。
- 若 upstream 不能单独暴露视觉分支，在本地 adapter 中包装已有 vision backbone；不得修改 upstream。
- 单元检查必须断言 latent path 的 model inputs 不含 `labels`。
- 主线 `FlowWAMSkillPolicy` 必须在冻结 adapter 之外调用共享 `LanguageConditionedViewFusion`；legacy latent baseline 可继续使用固定 view mean。
- `score_head` 零初始化，初始权重严格为 `[0.5,0.5]`；current/history/long weights 与 entropy 写入训练诊断。

### 4.2 Visual Target Encoder

新增：`mowe_wam/backbones/visual_target_encoder.py`

```python
class VisualTargetEncoder(nn.Module):
    def __init__(
        self,
        name: str = "dinov2_vits14",
        target_dim: int = 384,
        num_spatial_tokens: int = 16,
        checkpoint: str | None = None,
        freeze: bool = True,
    ): ...

    @torch.no_grad()
    def encode(self, pixel_values: Tensor) -> Tensor:
        """返回 [B, num_spatial_tokens, target_dim]。"""
```

规则：

- 永久 `eval()`，全部参数 `requires_grad=False`。
- Teacher transform 与 policy image augmentation 分离。
- DINOv2 ViT-S/14 默认保留原生 384 维，通过固定 `4×4` spatial average pooling 得到 16 tokens；target path 不放可学习 projector。
- 当前帧和未来帧使用同一 teacher 计算 delta target。
- 支持从 cache 读取；cache miss 可选择在线计算或明确失败。
- 推理 wrapper 不实例化 teacher。

### 4.3 Multi-scale Memory

新增：`mowe_wam/memory/multiscale_memory.py`

```python
class MultiScaleMemoryEncoder(nn.Module):
    def forward(
        self,
        history_visual_tokens,
        history_actions,
        history_mask,
        long_visual_tokens,
        long_actions,
        long_mask,
        language_tokens,
    ) -> dict[str, Tensor]:
        """返回 short_memory、long_memory、memory_context。"""

class OnlineMemoryState:
    def reset(self): ...
    def append(self, visual_tokens, executed_action): ...
    def tensors(self): ...
```

第一版 online state 保存当前 episode 的已观测图像与动作，并在每次 query 时重建与训练完全一致的短期连续 history 和均匀稀疏 episode-prefix indices；episode reset 必须清空状态，且不得存未来 teacher token。默认最多保留 600 个已观测 step。训练和推理必须共享相同的时间方向、索引与 mask 语义。

### 4.4 Shared Action-Flow Trunk 与 Nominal Flow Head

新增：`mowe_wam/models/action_flow.py`；修改：`mowe_wam/models/nominal_action_head.py`。

```python
class ActionFlowSampler:
    def sample(self, velocity_fn, condition, *, num_steps: int, seed: int | None) -> Tensor:
        """只在 normalized 6D motion space 积分，返回 [B, chunk, 6]。"""

class NominalActionHead(nn.Module):
    def __init__(self, context_dim, hidden_dim=512, motion_dim=6, chunk_size=8): ...
    def motion_velocity(self, noisy_motion, flow_time, context, memory_context) -> Tensor:
        """返回 nominal motion velocity，[B, chunk_size, 6]。"""
    def gripper_logits(self, context, memory_context) -> Tensor:
        """返回 shared binary gripper logits，[B, chunk_size, 1]。"""
    def sample(self, context, memory_context, *, seed=None, num_steps=4) -> dict[str, Tensor]: ...
```

返回字典固定包含 `nominal_motion [B,8,6]`、`gripper_logits [B,8,1]`、`nominal_actions [B,8,7]`。只有 motion 采用 conditional rectified flow：`x_s=(1-s)eps+s*A_motion*`，velocity target 为 `A_motion*-eps`；gripper 采用 `BCEWithLogits`。Nominal/residual motion 必须共享 normalization、flow-time embedding、noise distribution 与 solver schedule。Residual path 共享一个轻量 motion-flow trunk，六个 motor experts 仅保留独立 FiLM/adapter/velocity head；`null_finish` 没有 head。禁止先独立采样六个 `[B,8,6]` residual chunks 后拼接。warm-start 用固定 seed/solver 得到 `A0_motion`，随后对 residual target stop-gradient。

### 4.5 Latent World Model

新增：`mowe_wam/models/latent_world_model.py`

```python
class LatentWorldModel(nn.Module):
    def forward(
        self,
        current_context,
        short_memory,
        long_memory,
        action_condition,
        horizon_mask=None,
    ) -> dict[str, Tensor]:
        """
        return {
            "world_belief": ...,       # [B, 512]
            "route_world_tokens": ..., # [B, 8, 128] == h_1...h_8
            "future_latents": ...,     # [B, 3, 16, target_dim]
            "delta_latents": ...,      # [B, 3, 16, target_dim]
            "log_variance": ...,       # optional
        }
        """
```

实现要求：

- 6 层 causal Transformer，hidden 512，8 heads，FFN 2048。
- Nominal action token 编码 `A0_motion[j]` 与 `sigmoid(gripper_logits[j])`；模型沿 action prefix 输出 `h_1...h_8`，shape `[B,8,128]`。
- `h_1/h_4/h_8` 分别连接到原有 future/delta spatial predictor；其余 `h_k` 不增加高维视觉 target。
- Future latent head 与 delta head 独立投影。
- 不包含 decoder，不输出 RGB。
- 参数统计脚本必须单独报告 WAM 参数量。

### 4.6 Future Router

新增：`mowe_wam/models/future_router.py`

```python
class FutureGroundedRouter(nn.Module):
    def forward(
        self,
        world_belief,
        future_latents,
        delta_latents,
        route_world_tokens,
        memory_context,
        nominal_action_tokens,
        uncertainty=None,
        route_mode: str = "predicted",  # oracle | st_gumbel | predicted
    ) -> dict[str, Tensor]: ...
```

输出：

```python
{
    "router_logits": Tensor,    # [B, chunk_size, 7]
    "router_probs": Tensor,     # [B, chunk_size, 7]
    "route_gates": Tensor,      # [B, chunk_size, 7], oracle/ST-hard/predicted one-hot
    "route_indices": Tensor,    # [B, chunk_size], hard Top-1
    "current_skill": Tensor,    # [B], exactly route_indices[:, 0]
    "route_source": str,        # oracle | st_gumbel | predicted
}
```

第一版 router query 固定为简单相加版：

```python
action_q = self.action_mlp(nominal_action_tokens)        # [B, 8, D]
world_q = self.world_proj(route_world_tokens)            # [B, 8, D], h_1...h_8
pos_q = self.position_embedding.weight[None, :8, :]      # [1, 8, D]
queries = action_q + world_q + pos_q
router_logits = self.route_head(queries, pooled_future_context, memory_context)
```

即 action index `j=0...7` 对齐 `h_(j+1)`。第一版明确不加入 `h_(j+1)-h_j`、pre/post-state pair 或 transition-difference MLP。所有 nominal action tokens 在规划时已知；skill annotation 只能作为训练 target，不能进入 query/input。

训练时 `route_mode="oracle"` 从 batch 的逐位置 labels 产生 one-hot gate，但仍必须计算 logits 以训练 masked CE。joint 使用 `route_mode="st_gumbel"`：forward 为 hard per-token one-hot，backward 通过退火后的 soft Gumbel probabilities。部署始终为 `route_mode="predicted"`，逐位置 hard Top-1，并拒绝携带 label 字段。Router 当前 context skip 最大 128 维；future/delta 分支不得被删除或默认为零。

### 4.7 Residual Experts

修改：`mowe_wam/models/residual_experts.py`，保留旧 regression MoE 兼容。

```python
class ResidualFlowExperts(nn.Module):
    def velocity(
        self,
        expert_context,
        nominal_motion,
        noisy_residuals,
        flow_time,
        route_gates,
    ) -> Tensor:
        """一次 shared solve 内按 timestep gate 组合 6 个 motor velocity heads；null route 返回零。"""
    def sample(self, expert_context, nominal_motion, route_gates, *, seed, num_steps) -> Tensor: ...
```

最终输出必须包含：

```python
{
    "nominal_motion": A0_motion,       # [B,8,6]
    "gripper_logits": g0_logits,       # [B,8,1]
    "residual_motion": R_motion,       # [B,8,6]
    "motion_actions": motion_final,    # [B,8,6]
    "actions": A_final,                # [B,8,7]
    "route_indices": route_indices,
    "current_skill": route_indices[:, 0],
}
```

实现必须将 hard `null_finish` 位置的 6D residual solver state 初始化为零，而不是 Gaussian noise，并在每个 ODE step 后把 state 与 velocity 都保持为零；endpoint 再断言对应 motion residual 为零。`null_finish` 最终输出仍拼接 nominal gripper prediction，不得触发停止。训练期为支持 ST Gumbel 梯度，可以并行计算六个轻量 head 的 velocity 后由 `route_gates[..., :6]` 加权；共享 motion trunk 只计算一次。

不要用 detached boolean clamp 切断 ST 路由梯度。定义 `motor_gate = route_gates[..., :6].sum(-1, keepdim=True)`，其 hard forward 在 motor/null 位置分别为 1/0，但 backward 保留 soft Gumbel 梯度：

```text
r_0      = motor_gate * eps_R
v        = sum_i route_gates[..., i] * v_i
r_(k+1)  = motor_gate * (r_k + dt * v)
```

这样 null 的 forward state/velocity/endpoint 精确为零，同时 joint 阶段 action loss 仍能对 motor-vs-null route logits 提供 ST 梯度。

### 4.8 完整 Policy Wrapper

新增：`mowe_wam/models/latent_wam_policy.py`

```python
class LatentWAMPolicy(nn.Module):
    def forward(
        self,
        batch: dict,
        *,
        action_condition_mode: str = "scheduled",
        route_mode: str = "predicted",
        flow_seed: int | None = None,
        compute_teacher_targets: bool = True,
    ) -> dict[str, Tensor]: ...

    def predict_action(self, observation, memory_state) -> tuple[Tensor, OnlineMemoryState, dict]: ...
```

训练输出还应包含 current/future teacher targets、action-condition source 和 action-distance gate，便于 loss 与日志检查。部署 metadata 至少返回 `route_indices[8]`、首个预测边界和 `execution_steps`；policy 只向环境交付 `actions[:execution_steps]`。

执行长度算法固定为：

```python
def execution_steps(route_indices: Tensor, max_steps: int = 3) -> Tensor:
    """每个 batch item 返回 1..3；停在第一个与 route[0] 不同的位置之前。"""
```

若 schedule 为 `[Pick, Pick, Pick, Move, Move, Place, Place, Place]`，则本轮只交付前三个 Pick 动作；重新观测后重新生成完整 `A0`、WAM belief 和 schedule，而不是缓存并继续执行旧 chunk 的 Move/Place 部分。

现有 OpenVLA-OFT LIBERO evaluator 用 action queue 在队列为空时重新查询，并默认一次返回 8 步。新 adapter 必须允许 `predict_action` 返回长度为 1～3 的 prefix；队列耗尽后立即重查。不要修改 `external/openvla-oft/`，也不要把 `NUM_ACTIONS_CHUNK=8` 误当成必须执行满 8 步。测试必须记录 query id，证明旧 chunk 的未执行后缀没有进入下一轮队列。

## 5. Loss 与训练调度

### 5.1 Loss API

修改：`mowe_wam/training/latent_losses.py`；新增：`mowe_wam/training/flow_matching.py`。

```python
def future_latent_loss(pred, target, mask): ...
def delta_latent_loss(pred_delta, target_delta, mask): ...
def conditional_flow_matching_loss(pred_velocity, target_velocity, timestep_mask=None): ...
def gripper_binary_loss(gripper_logits, binary_targets, timestep_mask=None): ...
def masked_route_cross_entropy(router_logits, labels, timestep_mask, class_weights=None): ...
def residual_regularization(residuals, mask=None): ...
def flow_wam_skill_losses(outputs, batch, weights, schedule_state): ...
```

总损失：

```text
L_flow_nominal(v_0(A_s), A_motion* - eps_A)
+ sum_j motor_mask[j] * L_flow_expert(sum_i gate[j,i] v_i(R_s)[j], R_motion*[j] - eps_R[j])
+ L_gripper_bce(gripper_logits, target_gripper)
+ sum_j label_mask[j] * L_route(router_logits[j], expert_skill_labels[j])
+ L_future_cosine + 0.5 L_future_smooth_l1
+ 0.5 [L_delta_cosine + 0.5 L_delta_smooth_l1]
+ 0.01 L_load_balance                 # predicted/soft route 阶段才启用
+ 0.001 L_residual
+ lambda_endpoint * L1(motion_final, target_motion)  # warm-start 默认 0，joint 可选弱项
```

其中 `A_s=(1-s)eps_A+sA_motion*`，`R_motion*=A_motion*-stopgrad(A0_motion)`，`R_s=(1-s)eps_R+sR_motion*`。Flow/noise/residual/endpoint L1 一律只作用于 6D motion；gripper 只进入 BCE，禁止进入 residual addition。所有 mask 在 reduction 前按 timestep 应用。Route CE 对有效 motor/null labels 计算；expert flow loss 只对有效 motor labels 计算。unknown/padding 不得分配给 expert 0；cross-boundary chunk 不整体丢弃，且不做边界特殊 mask。`L_load_balance` 排除 null/unknown。`null_finish` 的零 motion residual 是结构约束，不是靠 residual loss 学出来的。

### 5.2 Action-condition Schedule

新增：`mowe_wam/training/schedules.py`

```python
def action_condition_probability(step: int, max_steps: int) -> float:
    """返回使用 nominal condition 的概率。"""

def action_distance_gate(nominal, target, beta: float) -> Tensor:
    """主要基于 6D motion distance 返回 detach 后的 [B] world-loss weight。"""
```

默认：

- 0%～30%：ground-truth condition；
- 30%～70%：线性增加 nominal condition；
- 70%～100%：80% nominal / 20% ground truth。

### 5.3 Router Schedule

- **Warm-start**：`route_mode=oracle`。每个有效 timestep 用其 label 选择对应 motor head；null 位置为零 residual，unknown 位置不产生 expert/route loss。chunk 可以包含多个 skill。
- **Joint**：从 oracle gates 退火到 per-token ST Gumbel gates；forward 维持 hard routing，backward 允许 route/action 联合优化。每个阶段分别记录 oracle、ST 与 hard-predicted endpoint error，不能只报混合结果。
- **Deployment**：`route_mode=predicted`，输入 schema 中不得出现 label。
- Top-2 不是第一版 schedule 的一部分；主线始终是每个 timestep 一个 hard route。
- checkpoint 必须保存 action-condition、flow solver、noise seed policy 与 oracle-to-predicted route schedule state；resume 后不能重置。

## 6. 实现任务

## Task 0：锁定新版 variant 与 legacy 边界

状态：已完成配置与类级隔离；legacy lightweight smoke 保持通过。

Goal：

确保配置、model builder 和训练脚本能够显式区分 `legacy_predicate`、`regression_residual_moe` baseline 与 `flow_wam_skill_moe` 主线。

Files：

- `scripts/train_mowe_wam.py`
- `configs/mowe_wam/train_predictive_memory_router.yaml`
- `configs/mowe_wam/train_flow_wam_skill_moe.yaml`（已新增）
- 新版 configs

Done Criteria：

- 旧 config 仍可解析，但打印 legacy warning。
- 新 config 不要求 `transition_label_path`。
- 新 variant 在未实现时明确报出“not implemented”，不能静默回退到 L1 regression 模型。
- 任何主训练日志都记录 `model.variant`。

## Task 1：实现 LIBERO episodic sequence dataset

Goal：

从 upstream episodic RLDS 流构造 multi-suite、multi-horizon、同 episode 窗口。

Files：

- `mowe_wam/data/libero_sequence_dataset.py`
- `mowe_wam/data/latent_wam_collator.py`
- `scripts/inspect_latent_sequence_dataset.py`
- `configs/mowe_wam/dataset_libero_mixture.yaml`
- `mowe_wam/data/expert_skill_labels.py`（已新增）
- `scripts/inspect_skill_experts.py`（已新增）

Done Criteria：

- Mock 和一个真实 episode 均能打印 K=8、M=4、H=[1,4,8] shape。
- 边界 mask 正确，不跨 episode。
- 四 suite mixture 能报告有效 window 数。
- 不读取 predicate/transition JSONL。
- label report 能输出六个 motor skills + null route 的覆盖、逐位置未知比例、来源与 dataset fingerprint；第一版按 timestep direct mapping，不增加对齐 gate、边界软标签或 debounce。
- 四 suite audit 能验证 1,693 个 episode、273,465 个 annotation timestep、259,921 个有效 H=8 window，并记录稳定 manifest fingerprint；exact key 成功不得改写为语义 alignment 已验证。

## Task 2：实现 non-leaky OpenVLA context

状态：已实现 pre-action instruction prompt、结构化 target-field 隔离、instruction cache 与逐帧视觉 pooled-feature LRU；真实 7B 端到端显存/吞吐仍待 GPU gate。

Goal：

提取视觉/语言 context，不把 action target token 或 labels 输入 latent path。

Files：

- `mowe_wam/backbones/openvla_oft_adapter.py`
- `scripts/check_backbone.py`

Done Criteria：

- API 返回 current/history visual tokens 与 language tokens。
- 测试在传入 labels 时拒绝 latent context path，或显式忽略并断言未送入模型。
- `last_action_hidden` 只存在于 legacy 分支。
- 冻结 language feature 按 instruction 缓存；重叠训练窗口和部署重规划中的相同预处理帧按精确像素 hash 复用 pooled visual feature，LRU 容量由 config 限制。

## Task 3：实现 VisualTargetEncoder 与 cache

状态：已实现 `latent_teacher_feature_cache_v2`；每个 episode/timestep 只存一个 float16 feature，按 shard 写入并通过 manifest index 延迟读取，避免全量特征驻留内存或单文件 checkpoint。

Goal：

用冻结 DINOv2 输出 16 个 spatial target tokens，并支持可验证 cache。

Files：

- `mowe_wam/backbones/visual_target_encoder.py`
- `scripts/cache_visual_targets.py`

Cache metadata：

```json
{
  "teacher_checkpoint": "facebook/dinov2-small",
  "transform_hash": "...",
  "transform_id": "...",
  "image_resolution": [224, 224],
  "spatial_tokens": 16,
  "target_dim": 384,
  "future_horizons": [1, 4, 8],
  "dataset_fingerprint": "...",
  "skill_sidecar_fingerprint": "...",
  "storage_contract": "one_float16_feature_per_episode_timestep"
}
```

Done Criteria：

- 同一图像重复编码输出一致。
- Teacher 无梯度、始终 eval。
- Cache mismatch 明确失败，不静默复用。
- Cache 输出为 `manifest.json + features-*.pt`，按 timestep 去重且单 shard 大小可配置；训练端 LRU 延迟载入 shard。
- 推理构建 model 时可完全不加载 teacher。

## Task 4：实现 MultiScaleMemory

Goal：

编码短期连续 history 和稀疏 episode-prefix summaries，并提供与训练索引一致的在线 episode memory。

Files：

- `mowe_wam/memory/multiscale_memory.py`
- `scripts/check_flow_wam_forward.py`（已新增；旧 synthetic checker 保留 baseline）

Done Criteria：

- Episode reset 后无前一 episode 状态。
- Masked padding 不影响有效 memory。
- 所有 memory token 时间戳不晚于当前 `t`。
- 训练 sample 与在线 buffer 的 shape/顺序一致。

## Task 5：实现 Nominal Flow Policy 与 LatentWorldModel

状态：本地 synthetic、真实 RLDS + DINO lightweight context，以及云端完整 OpenVLA 7B Stage 1 forward/backward、单 optimizer step 和 resume 均已完成；外部采样峰值约 15,980 MiB。

Goal：

先完成不含 router 的 Stage 1 前向和 loss。

Files：

- `mowe_wam/models/action_flow.py`（已新增）
- `mowe_wam/models/nominal_action_head.py`
- `mowe_wam/models/latent_world_model.py`
- `mowe_wam/training/latent_losses.py`
- `scripts/pretrain_nominal_flow_wam.py`（已新增）
- `configs/mowe_wam/train_nominal_flow_wam.yaml`（已新增）

Done Criteria：

- `nominal_motion` 为 `[B,8,6]`，`gripper_logits` 为 `[B,8,1]`，拼接后的 `A0` 为 `[B,8,7]`。
- Fixed-seed flow sampler 对同一输入输出一致；不同 seed 的 output shape/范围正确。
- nominal motion-flow velocity 对应 rectified-flow target，且 6D motion normalization 在 noise、solver 和 endpoint 上保持一致；gripper 不进入 flow/noise/residual。
- Future/delta 为 `[B,3,16,384]`。
- `route_world_tokens` 为 `[B,8,128]`，且 `h_1/h_4/h_8` 分别驱动三个 spatial future/delta predictions。
- gripper BCE finite；canonical `0/1` 与 evaluator 的 binarize/sign conversion 有独立测试。
- WAM 参数量单独报告且约 20M～30M。
- 一个真实 batch 可完成 forward/backward 和 checkpoint resume。
- 日志区分 GT-conditioned 与 nominal-conditioned world loss。

## Task 6：实现 Temporal Label-Seeded Router 与 Residual Flow Experts

状态：oracle/ST/predicted、simple query、六 expert 梯度、null-zero、exact sidecar/RLDS join、online-memory parity、variable-prefix unit test、one-task simulator adapter 与可恢复 full-suite evaluator 已完成；真实 LIBERO rollout 待验证。

Goal：

将 predicted future change 转为 `[B,8,7]` per-token coarse-skill schedule；先以训练期逐位置 leading-verb labels 预训练六个 motor residual-flow heads，再让 router 预测当前与未来位置的 skills。

Files：

- `mowe_wam/models/future_router.py`
- `mowe_wam/models/residual_experts.py`
- `mowe_wam/models/latent_wam_policy.py`
- `mowe_wam/evaluation/libero_temporal_policy.py`
- `scripts/warmstart_skill_flow_experts.py`（已新增）
- `scripts/eval_libero_temporal_skill.py`（已新增 offline/prefix、one-task simulator loop 与逐 task/trial JSONL 恢复的 full-suite evaluator；真实 simulator rollout 待验证）
- `configs/mowe_wam/warmstart_skill_flow_experts.yaml`（已新增）

Done Criteria：

- Oracle route 按 timestep 选择 label-matched motor head，且每类、每位置 residual flow loss 可分别记录。
- `motion_final=A0_motion+R_motion` 与拼接后的 `[B,8,7] A_final` shape 正确；residual target 只使用 6D motion；只进行一次 shared residual flow solve。
- gripper 只来自 shared binary head，不被任何 residual expert 修改。
- 每个有覆盖的 motor expert 都获得有限梯度；unknown 位置不更新 expert/route loss；null 位置 motion residual 精确为零但 nominal action 保留。
- Router query 严格为 `ActionMLP(A0[j]) + WorldProjection(h_(j+1)) + PositionEmbedding(j)`；测试确认没有 delta-h/pre-post-state 分支。
- Oracle、ST Gumbel 与 hard-predicted schedules 均可前向；部署 forward 明确拒绝 label。
- `[Pick, Pick, Pick, Move, ...]` 的执行器只返回前三步，且每次重规划后丢弃旧 chunk 未执行后缀。
- LIBERO variable-prefix queue adapter 在 1/2/3 步后分别重新 query，且不执行旧 query 的剩余动作。
- 每个环境 step 都必须写入 online memory，即使 policy queue 尚未耗尽；query 时的 history/prefix indices 与训练采样一致。
- checkpoint 中联合 6D motion `q01/q99` 必须在送入 LIBERO 前反归一化；canonical gripper `0/1` 再转换为 LIBERO `+1/-1`。
- feature-store checkpoint 在评测时必须把 backbone 显式重建为 online frozen OpenVLA，而不能保留 `precomputed_features` mode；正式入口在加载大模型前校验 CLI 指定的 backbone 与 checkpoint 内绑定的 `backbone_identifier` 完全一致。full-suite 只接受 Stage 3 joint 且 source dataset 为 LIBERO 的 checkpoint，防止换用形状兼容但语义不同的 OpenVLA 或静默串用 CALVIN action contract。
- full-suite 每个 `(task_id, trial)` 立即 append JSONL，恢复时绑定 checkpoint、suite、seed/flow-seed 并拒绝重复 key；summary 记录 per-task/overall success、官方 initial-state trial 数与 upstream OpenVLA-OFT commit。
- Router 不接受完整 unprojected OpenVLA feature。

## Task 7：实现 Oracle-to-ST-Predicted 联合训练入口

状态：三入口、分组学习率、warmup-cosine、AMP、schedule、JSONL、checkpoint/RNG resume、resolved metadata sidecar 与 Stage 1→2→3 predecessor guard 已实现；云端真实 Stage 1 optimizer step/resume、Stage 2 one-step 与 25-step oracle coverage gate 已通过，Stage 3 真实 optimizer step 仍待完成。

Goal：

支持 nominal flow pretrain、oracle per-token expert warm-start、oracle-to-ST-Gumbel joint 训练、AMP、resume 和诊断日志。

Files：

- `scripts/pretrain_nominal_flow_wam.py`（已新增）
- `scripts/warmstart_skill_flow_experts.py`（已新增）
- `scripts/train_flow_wam_skill_moe.py`（已新增）
- `configs/mowe_wam/train_flow_wam_skill_moe.yaml`（已新增）
- `mowe_wam/training/schedules.py`

CLI：

```bash
python scripts/train_flow_wam_skill_moe.py \
  --config configs/mowe_wam/train_flow_wam_skill_moe.yaml \
  --data-root /PATH/TO/modified_libero_rlds \
  --checkpoint /PATH/TO/OPENVLA_CHECKPOINT \
  --init-wam /PATH/TO/STAGE2_CHECKPOINT \
  --output-dir outputs/train/flow_wam_skill_moe
```

必须支持：

- `--max-steps`
- `--limit-batches`
- `--grad-accumulation-steps`
- `--save-freq`
- `--log-freq`
- `--resume`
- `--precision bf16|fp16|float32`
- `--teacher-cache`
- `--skill-expert-config`
- `--route-mode oracle|scheduled|predicted`
- `--flow-solver-steps`

Done Criteria：

- Optimizer 只包含 `requires_grad=True` 参数。
- OpenVLA 和 teacher 保持 eval/frozen。
- Resume 恢复 optimizer、scheduler、scaler、action-condition、solver/noise policy 和 oracle-to-predicted route schedule。
- Stage 2 只接受 Stage 1 checkpoint，Stage 3 只接受 Stage 2 checkpoint；same-stage resume 单独校验。checkpoint 内嵌 taxonomy/audit、action statistics、teacher/cache contract 与 backbone identifier。
- same-stage resume 还必须保持 seed、完整 `max_steps`、precision、optimizer/LR/warmup/min-LR/grad-clip、loss weights、router/action-conditioning schedule、window contract、route mode 与 ablation 不变；`stop_step`、日志/保存频率可调整。这样 102→104→125→200→500→1000 只改变临时退出边界，不会悄悄重定义 scheduler 或训练目标。Stage predecessor init 不受此同阶段限制，允许按 Stage 1→2→3 切换优化配置。
- 一个真实 optimizer step 后产生 resolved config、JSONL log 和 latest checkpoint。

## Task 8：实现 preflight 与风险日志

状态：synthetic backward、“真实四-suite RLDS + 真实 DINO + lightweight context”以及云端完整 OpenVLA 7B Stage 1 batch backward/optimizer/resume preflight 已通过；Stage 2/3 与 8 卡路径仍由各自真实门禁单独验证。

Goal：

在长训练前一次性检查 `ARCHITECTURE_RISKS.md` 中可自动化的硬门槛。

Files：

- `scripts/preflight_flow_wam_training.py`（已新增；旧 latent preflight 保留 baseline）

检查项：

- episode/window/horizon mask；
- 6D motion normalization 与 1D canonical binary gripper 分离；
- context input 不含 labels；
- expert label 来源、逐位置 mask、六 motor + null 覆盖和 test/future leakage；
- cache fingerprint；
- teacher frozen；
- nominal/residual 6D flow velocity、solver endpoint、gripper logits/BCE、`[B,8,128]` world tokens 与 WAM/router/expert shapes；
- expert gradient coverage；
- finite loss/backward；
- peak CUDA memory（若有 GPU）。

Done Criteria：

- 失败时指出具体 risk ID。
- 不把 preflight 通过写成 benchmark 成功。
- `--real-data-lightweight` 只能证明真实数据/sidecar/teacher/loss/backward 链路，不得写成 OpenVLA 7B 已验证。

## Task 9：实现分析与机制消融

状态：flow JSONL analyzer、future/history/behavior/dense/oracle/predicted/shuffled-skill/copy-current configs、route-source 对比、逐位置熵、skill×位置误差、router 分支强度与 oracle/ST/hard paired diagnostics 已实现并通过 mock/contract tests；真实机制曲线与 benchmark 对比待训练日志生成后完成。

Goal：

证明增益来自 verb-seeded skill warm-start 与 future-grounded temporal routing，而非参数量、标签泄漏或历史长度。`instruction_only_router` 暂不作为第一版实现门槛。

Files：

- `mowe_wam/analysis/future_prediction.py`
- `mowe_wam/analysis/routing_diagnostics.py`
- `scripts/analyze_flow_wam_logs.py`（已新增）
- ablation configs

必须输出：

- horizon-wise cosine/Smooth L1/delta error；
- current-copy baseline；
- motion-magnitude buckets；
- motor expert usage/gradient coverage、per-position router entropy；
- 六 motor + null 的 label 覆盖、逐位置 confusion、每类 nominal/residual flow loss 与 endpoint error；
- oracle schedule vs ST schedule vs hard predicted schedule vs shuffled-route gap；
- current-skill accuracy、future-position accuracy、boundary precision/recall/F1、schedule edit distance；
- execution steps 分布、真实边界越界率、重规划频率和 null-zero violation count；
- nominal/final/target 6D motion distance 与 gripper accuracy；
- motion residual magnitude、`h_1...h_8` norm/variance；
- future-shuffle/mask 后 router change；
- latency 与参数量。

Done Criteria：

- Analysis 可在 mock log 上运行。
- 不制造 success rate。
- 每个主风险至少对应一个可记录指标或明确人工检查。

## Task 10：Benchmark fine-tuning 接口

状态（2026-07-17）：CALVIN action adapter、双视角/goal policy bridge、官方 sequence/subtask reset bridge、Stage 3 checkpoint/action-statistics guard、严格 ABC language-segment reader/audit、单次扫描 canonical+feature 双层 converter、三阶段 DDP8 配置和 dependency-light smoke 已实现。canonical archive 支持 CALVIN static `200×200` 与 gripper `84×84` 的独立 camera shape。官方数据全量审计/转换、ABC 训练、官方环境安装和 1,000-sequence 真实评测尚未完成。

Goal：

为已确定的第二主 benchmark **CALVIN** 实现独立 adapter，使 MoWE-WAM 能在不污染 LIBERO action/data contract 的前提下完成官方 LH-MTLC ABC→D 训练和评测。L-CALVIN 只作为标准 CALVIN 验证后的可选扩展；BOSS 不再是第一版候选。

Files：

```text
mowe_wam/benchmarks/calvin/action_adapter.py
mowe_wam/benchmarks/calvin/policy_adapter.py
mowe_wam/benchmarks/calvin/custom_model.py
configs/mowe_wam/calvin_abc_d.yaml
scripts/eval_calvin_flow_wam.py
scripts/check_calvin_adapter.py
mowe_wam/benchmarks/calvin/dataset.py
scripts/audit_calvin_training_data.py
scripts/convert_calvin_to_mowe_store.py
scripts/audit_calvin_feature_store_equivalence.py
configs/mowe_wam/ddp8_calvin_nominal_flow_feature_store.yaml
configs/mowe_wam/ddp8_calvin_warmstart_skill_flow_feature_store.yaml
configs/mowe_wam/ddp8_calvin_joint_flow_feature_store.yaml
```

Proposed API：

```python
class CalvinActionAdapter:
    def to_shared_action(self, calvin_action): ...
    def from_shared_action(self, shared_action): ...

class CalvinPolicyAdapter:
    def reset(self): ...
    def step(self, obs, goal): ...
```

Contract：

- 对接官方 `CustomModel` 生命周期：当前官方代码在每个 subtask 前调用 `reset()`、每个环境步调用 `step(obs, goal)`。本地 evaluator bridge 必须在 environment sequence reset 处额外调用 `reset_sequence()`；subtask `reset()` 默认只清 goal/queue、不清 episode memory，并提供 `preserve_memory_across_subtasks=false` 消融。
- 将 `rgb_static` / `rgb_gripper` 明确映射为 primary/wrist view，并记录 resize、crop、颜色空间、view order 和语言 tokenizer fingerprint。
- 在 action adapter 中单独配置并测试 relative/absolute Cartesian mode、world/gripper frame、position/rotation scaling、Euler/axis-angle 转换、gripper sign 和 30 Hz 控制契约；禁止直接复用 LIBERO 数值而不做 round-trip/rollout 验证。
- MoWE 可内部预测 H=8 action chunk，但对 CALVIN 环境仍逐步输出；沿用 boundary-capped 1～3 步执行和队列耗尽重查，`goal` 变化或预测 skill boundary 必须立即丢弃旧后缀。
- 训练数据只读取 A/B/C train split；D 环境仅用于官方 ABC→D evaluation，不生成 teacher cache、不参与 action normalization 或模型选择。
- reader 只接受官方 `task_ABC_D/training` 和 `lang_annotations/auto_lang_ann.npy` 的 inclusive frame spans，明确拒绝 `validation`/D 与其他 split root；相机、`robot_obs[15]`、`rel_actions[7]`、gripper `close=-1/open=+1` 均逐帧校验。
- 6D motion q01/q99 只从 ABC language-annotated unique training frames 计算；转换后统一为 normalized motion + canonical `0=open,1=closed`。CALVIN audit 的 segment/transition/window counts 同样写入 feature completion contract；actual 不一致或 `--limit-segments` 产物必须带 `formal_training_ready=false`，训练 runtime 会拒绝该 smoke-only/incomplete store。
- CALVIN language segment 是一个 goal-bounded training episode，不能跨 annotation span 构造 H=8 window；coarse skill 由当前语言指令 leading verb 生成并先审计 unknown ratio、六 motor 类覆盖与 class weight。真实审计不通过时停止，不通过修改阈值或伪造 null label 放行。
- converter 在同一次 segment materialization 中先从原始相机 tensor 生成正式 OpenVLA/DINO feature，再把相同 raw static/gripper frame 写入 canonical H.264；两类 writer 独立按 episode 恢复。canonical manifest 为每路 camera 单独记录 shape，不能把 wrist resize 到 static 尺寸后冒充原始归档。
- CALVIN 必须使用 `audit_calvin_feature_store_equivalence.py` 从官方 ABC NPZ 重新编码抽样 segment，独立重建窗口并比较双视角/language/DINO/action/skill、完整模型输出与 loss；统一 readiness gate 会校验 equivalence report 的 benchmark identity，禁止拿 LIBERO 报告替 CALVIN 放行。
- 正式结果由官方评测器产生，至少记录 average sequence length、连续完成 1/2/3/4/5 个子任务的 success rate、per-task success、失败子任务位置、policy query/prefix histogram、seed、checkpoint/backbone 和评测器 commit；本地 bridge 在每条官方 sequence 返回时采集这些诊断，最终 summary 原子替换。
- formal loader 只接受 Stage 3 joint checkpoint，CLI 指定的 frozen OpenVLA 必须与 checkpoint 绑定的 `backbone_identifier` 一致，且 checkpoint 内 CALVIN q01/q99 必须与 action adapter 完全一致；这会明确拒绝换用另一个 backbone 或直接拿 LIBERO-normalized checkpoint 跑 CALVIN。

Done Criteria：

- LIBERO path 稳定后再开始。
- 固定并记录 CALVIN repo commit、数据版本、PyBullet/依赖环境和官方 evaluation sequence 配置；先完成官方 baseline/custom-model 空适配 smoke。
- [已通过本地测试] action transform round-trip、gripper sign、camera mapping、sequence/subtask reset、goal 切换和 stale-action-queue contract。
- [代码与合成官方 schema 已通过] ABC→D 训练/评测隔离可审计；reader 拒绝 D，partial store 拒绝进入训练。真实官方数据仍需生成审计报告，确认 D 不进入 feature cache、normalization、early stopping 或 checkpoint selection。
- 用小 checkpoint 完成至少一个官方五子任务 sequence 的端到端 rollout，再运行完整官方评测；日志中没有伪造或用离线 proxy 替代的 success 指标。
- 标准 CALVIN 门禁完成前不启动 L-CALVIN，也不声称 CALVIN 已经接入。

官方评测命令模板（未运行）：

```bash
python scripts/eval_calvin_flow_wam.py \
  --calvin-root /path/to/calvin \
  --dataset-path /path/to/task_ABC_D \
  --config configs/mowe_wam/calvin_abc_d.yaml \
  --flow-checkpoint /path/to/calvin_stage3/checkpoint_latest.pt \
  --backbone-checkpoint /path/to/openvla_calvin_checkpoint \
  --eval-log-dir outputs/eval/calvin_abc_d
```

该 wrapper 继续使用官方 `get_sequences(NUM_SEQUENCES=1000)`、task oracle、environment、360-step subtask limit 和 `print_and_save`；只增加明确的 model sequence-reset hook。正式实验必须绑定 `configs/mowe_wam/calvin_abc_d.yaml` 中记录的官方 commit，并同时运行 per-subtask-memory-reset 消融。

CALVIN ABC 训练数据准备与三阶段命令模板（未在真实 CALVIN 数据或 8 卡上运行）：

```bash
python scripts/audit_calvin_training_data.py \
  --dataset-root /data/task_ABC_D \
  --output outputs/preflight/calvin_abc_training_audit.json \
  --skill-config-output outputs/preflight/calvin_skill_experts.json

python scripts/convert_calvin_to_mowe_store.py \
  --dataset-root /data/task_ABC_D \
  --checkpoint /models/openvla-calvin \
  --teacher-checkpoint /models/facebook-dinov2-small \
  --output /data/mowe_calvin_abc_features_v1 \
  --canonical-output /data/mowe_calvin_abc_lerobot_v3 \
  --canonical-fps 30 \
  --audit-output outputs/preflight/calvin_abc_training_audit.json \
  --skill-config-output outputs/preflight/calvin_skill_experts.json \
  --encode-batch-size 16 --episodes-per-shard 96 \
  --canonical-episodes-per-chunk 32 --precision bf16

python scripts/audit_calvin_feature_store_equivalence.py \
  --config configs/mowe_wam/ddp8_calvin_nominal_flow_feature_store.yaml \
  --benchmark-config configs/mowe_wam/calvin_abc_d.yaml \
  --store /data/mowe_calvin_abc_features_v1 \
  --dataset-root /data/task_ABC_D \
  --checkpoint /models/openvla-calvin \
  --teacher-checkpoint /models/facebook-dinov2-small \
  --samples 100 \
  --output outputs/preflight/calvin_abc_feature_equivalence_100.json

torchrun --standalone --nproc-per-node=8 scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_calvin_nominal_flow_feature_store.yaml \
  --feature-store /data/mowe_calvin_abc_features_v1 \
  --checkpoint /models/openvla-calvin \
  --skill-expert-config outputs/preflight/calvin_skill_experts.json
torchrun --standalone --nproc-per-node=8 scripts/warmstart_skill_flow_experts.py \
  --config configs/mowe_wam/ddp8_calvin_warmstart_skill_flow_feature_store.yaml \
  --feature-store /data/mowe_calvin_abc_features_v1 \
  --checkpoint /models/openvla-calvin \
  --skill-expert-config outputs/preflight/calvin_skill_experts.json \
  --init-wam /path/to/calvin_stage1/checkpoint_latest.pt
torchrun --standalone --nproc-per-node=8 scripts/train_flow_wam_skill_moe.py \
  --config configs/mowe_wam/ddp8_calvin_joint_flow_feature_store.yaml \
  --feature-store /data/mowe_calvin_abc_features_v1 \
  --checkpoint /models/openvla-calvin \
  --skill-expert-config outputs/preflight/calvin_skill_experts.json \
  --init-wam /path/to/calvin_stage2/checkpoint_latest.pt
```

三份配置默认 Stage 1/2/3 scheduler horizon 分别为 20k/15k/10k steps、每卡 batch 1、accumulation 1、BF16、validation/save 每 500 step。它们是针对 feature-store + 冻结 backbone 的保守启动值，不是已经由 CALVIN 曲线证明的最优值；真实训练先用 `--stop-step` 完成 2/25/100-step 门禁，再保持同一 `max_steps` 连续运行，并按 validation/future-predictor/router 结果决定是否延长。

CALVIN 训练同样必须先生成带显式 imbalance limits 的 `audit_mowe_feature_store.py --world-size 8 --verify-all-checksums` 报告、8-rank 10k-step soak 和目标节点 `audit_ddp_runtime.py` 报告，再用 `audit_long_training_readiness.py` 合并证据。Stage 1 从头训练时不传 checkpoint；Stage 2/3 分别用对应 config、`--checkpoint-mode init` 和前一阶段 checkpoint 重新生成 readiness 报告。统一 gate 会从 manifest 推断 `calvin_abc_d` identity，并拒绝 LIBERO equivalence/action/checkpoint contract。

## 7. 训练配置建议

### 7.1 Stage 1：Nominal flow + WAM pretrain

```yaml
training:
  batch_size: 1
  grad_accumulation_steps: 8
  max_steps: 50000
  precision: bf16
  learning_rate:
    memory: 0.0001
    nominal_flow: 0.0001
    gripper_head: 0.0001
    world_model: 0.0001
  weight_decay: 0.01
  max_grad_norm: 1.0
```

### 7.2 Stage 2：Oracle temporal-skill residual-flow warm-start

```yaml
training:
  batch_size: 1
  grad_accumulation_steps: 8
  max_steps: 50000
  precision: bf16
  learning_rate:
    memory: 0.00001
    nominal_flow: 0.0  # frozen
    gripper_head: 0.0  # frozen
    world_model: 0.00001
    router: 0.0001
    residual_flow_experts: 0.0001
  weight_decay: 0.01
  max_grad_norm: 1.0
loss_weights:
  flow_nominal: 0.0  # frozen component; keep raw metric but exclude from total_loss
  gripper_bce: 0.0   # frozen component; keep raw metric but exclude from total_loss
  flow_expert: 1.0
  route: 1.0
  world: 1.0
  delta: 0.5
  load_balance: 0.0  # oracle route in Stage 2
  residual: 0.001
  endpoint: 0.0
```

Stage 2 仍记录 nominal/gripper 原始指标用于诊断 predecessor checkpoint，但二者已冻结，不能以权重 1 混入 `total_loss`，否则 validation 曲线会被不可优化的常数项污染。六个 feature-store configs 也显式将 `data_root`、`skill_sidecar_path` 和 `tf_frame_parallel_calls` 设为 `null`；正式训练 provenance 只读取 manifest 中的 benchmark-specific source/annotation contract，不保留未使用的 RLDS/TF 路径。

### 7.3 Stage 3：ST-Gumbel temporal-route joint fine-tuning

从 Stage 2 checkpoint 恢复；`nominal_flow` 与 `gripper_head` 以 `1e-5` 量级解冻、WAM `1e-5`、router 和 residual flow heads `1e-4`。`oracle_route_probability` 从 1 退火到 0，替换为温度从 1.0 退火到 0.1 的 per-token ST Gumbel gates，`L_route` 保持开启。以上均是启动默认值，不是已验证的最佳超参数；真实训练步数应根据有效 window 数、类别覆盖、epoch 和云端预算调整。

## 8. 日志与 Checkpoint 契约

每条训练日志至少包含：

```json
{
  "step": 0,
  "stage": "nominal_flow_pretrain|expert_warmstart|joint",
  "total_loss": 0.0,
  "nominal_flow_loss": 0.0,
  "expert_flow_loss": 0.0,
  "gripper_bce_loss": 0.0,
  "gripper_accuracy": 0.0,
  "route_ce_loss": 0.0,
  "world_cosine_loss": 0.0,
  "world_smooth_l1_loss": 0.0,
  "delta_loss": 0.0,
  "load_balance_loss": 0.0,
  "residual_loss": 0.0,
  "endpoint_loss": 0.0,
  "motion_endpoint_l1": 0.0,
  "action_condition_source": "ground_truth|nominal",
  "action_distance_gate_mean": 0.0,
  "router_entropy": 0.0,
  "route_source": "oracle|st_gumbel|predicted",
  "oracle_route_probability": 1.0,
  "gumbel_temperature": 1.0,
  "route_label_coverage": [0, 0, 0, 0, 0, 0, 0],
  "route_accuracy_by_position": [0, 0, 0, 0, 0, 0, 0, 0],
  "current_skill_accuracy": 0.0,
  "boundary_f1": 0.0,
  "motor_expert_usage": [0, 0, 0, 0, 0, 0],
  "null_route_usage": 0.0,
  "null_motion_zero_violation_count": 0,
  "execution_steps_mean": 0.0,
  "skill_boundary_overrun_rate": 0.0,
  "nominal_motion_target_l1": 0.0,
  "final_motion_target_l1": 0.0,
  "motion_residual_norm": 0.0,
  "route_world_token_norm": 0.0,
  "route_world_token_variance": 0.0,
  "future_horizon_errors": {},
  "view_order": ["primary", "wrist"],
  "current_view_weights_mean": [0.5, 0.5],
  "current_view_entropy_mean": 0.693,
  "view_weights_by_current_skill": {},
  "cache_fingerprint": "..."
}
```

Checkpoint 必须保存：

- model variant 与 resolved config；
- trainable model states；
- nominal 6D motion-flow、shared gripper head 与 `h_1...h_8` world-token heads；
- optimizer/scheduler/scaler；
- global step；
- action-condition schedule state；
- flow formulation, solver steps, solver implementation id and deterministic seed policy；
- oracle-to-ST-Gumbel routing schedule、temperature 与 RNG state；
- `flow_wam_skill_components_v2` 的 trainable View Fusion state，以及 observation views、teacher target views、view order、fusion config 与 `num_images_in_input=2`；
- expert-skill taxonomy、leading-verb mapping/version、per-position coverage summary、默认 aligned 标志与 sidecar fingerprint；
- 6D motion normalization statistics 与 gripper canonical/sign-conversion contract；
- teacher/cache metadata；
- backbone checkpoint identifier；
- distributed contract、各 rank RNG 与 episode-aware sampler state；
- `checkpoint_latest.pt` 先完整写入同目录临时文件再原子替换。metadata sidecar 在切换 generation 前移除，因此中断时读取端回退到 checkpoint 内嵌 metadata，不会把旧 sidecar 配给新 checkpoint。

## 9. Smoke-first 命令顺序

以下命令只是进入训练循环的最低检查，不代表性能验证：

```bash
PYTHONPYCACHEPREFIX=/tmp/mowe_flow_pycache \
python3 -m compileall -q mowe_wam scripts

python scripts/inspect_latent_sequence_dataset.py \
  --data-root /PATH/TO/modified_libero_rlds \
  --dataset-name libero_spatial_no_noops \
  --limit 2

python scripts/inspect_skill_experts.py \
  --data-root /PATH/TO/modified_libero_rlds \
  --skill-config configs/mowe_wam/skill_experts.yaml \
  --limit 20

python scripts/audit_flow_wam_rlds.py \
  --data-root /PATH/TO/modified_libero_rlds \
  --output outputs/preflight/flow_wam_rlds_all_suites_audit.json

python scripts/cache_visual_targets.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root /PATH/TO/modified_libero_rlds \
  --processor-checkpoint /PATH/TO/OPENVLA_CHECKPOINT \
  --output /PATH/TO/DINO_TEACHER_CACHE \
  --shard-size 4096

python scripts/check_flow_wam_forward.py \
  --synthetic --batch-size 2

python scripts/preflight_flow_wam_training.py \
  --real-data-lightweight \
  --data-root /PATH/TO/modified_libero_rlds \
  --backward

python scripts/preflight_flow_wam_training.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root /PATH/TO/modified_libero_rlds \
  --checkpoint /PATH/TO/OPENVLA_CHECKPOINT \
  --backward

python scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root /PATH/TO/modified_libero_rlds \
  --checkpoint /PATH/TO/OPENVLA_CHECKPOINT \
  --max-steps 1 --limit-batches 1

python scripts/warmstart_skill_flow_experts.py \
  --config configs/mowe_wam/warmstart_skill_flow_experts.yaml \
  --data-root /PATH/TO/modified_libero_rlds \
  --checkpoint /PATH/TO/OPENVLA_CHECKPOINT \
  --init-wam /PATH/TO/STAGE1_CHECKPOINT \
  --max-steps 1 --limit-batches 1

python scripts/train_flow_wam_skill_moe.py \
  --config configs/mowe_wam/train_flow_wam_skill_moe.yaml \
  --data-root /PATH/TO/modified_libero_rlds \
  --checkpoint /PATH/TO/OPENVLA_CHECKPOINT \
  --init-wam /PATH/TO/STAGE2_CHECKPOINT \
  --max-steps 1 --limit-batches 1

# Stage 3 完成后的可恢复 LIBERO full-suite 正式入口；先用单 task/trial smoke。
python scripts/eval_libero_temporal_skill.py \
  --simulator --all-tasks \
  --task-suite libero_spatial --trials 50 \
  --policy-checkpoint /PATH/TO/STAGE3_CHECKPOINT \
  --backbone-checkpoint /PATH/TO/OPENVLA_CHECKPOINT \
  --output-jsonl outputs/eval/libero_spatial/episodes.jsonl \
  --summary-output outputs/eval/libero_spatial/summary.json \
  --resume-results --seed 7 --flow-seed 1701
```

## 10. 长训练前 Definition of Ready

- [x] 新 dataset 使用真实 ordered episodes 构造窗口；读取顺序固定为 `shuffle_files=false`、`num_parallel_reads=16`，窗口级 buffered shuffle 在 join 后进行。
- [x] 四 suite 的 episode/transition/H=8 window 数、联合 6D motion stats 与 gripper binary distribution 已记录。
- [x] Latent path 的 backbone call boundary 不含 action、skill 或 teacher target；contract test 已覆盖。
- [x] Teacher/cache fingerprint 可复现；sharded cache smoke 与 metadata mismatch gate 已通过。
- [x] `expert_skill_labels` 的来源、leading-verb 映射、逐位置未知比例与六 motor + null 覆盖已记录；1,693 个 episode key 和 273,465 个 timestep key 精确匹配。语义 timestep alignment 仍为显式假设，不增加边界特殊 mask 或对齐 gate。
- [x] 双视角真实 RLDS + primary-only 真实 DINO + lightweight context forward/backward 通过；该项不替代完整 backbone gate。
- [x] dataset、online memory、OpenVLA adapter 与 LIBERO evaluator 均使用同 timestep 的 `[primary,wrist]`；权重 shape/sum、neutral initialization、gradient 与 paired LRU 有 contract tests。
- [x] Stage 1 一个真实 7B batch forward/backward 通过；RTX 4090 外部 1 秒采样峰值约 15,456 MiB。
- [x] 轻量/合成 Stage 1 contract 输出 `[B,8,6]` motion、`[B,8,1]` gripper logits 与逐步 `h_1...h_8`，且 gripper BCE/accuracy finite；真实 7B batch 仍由上一项单独把关。
- Stage 2 oracle routing 下每个有覆盖类别的 residual flow expert 获得梯度，且 endpoint error 有限。
- [x] ST Gumbel 与 hard-predicted temporal routing 可运行且部署 batch 不读取 label；logger 可在同一 log step 记录 oracle/ST/hard paired diagnostics。
- [x] Current-copy baseline 已实现为逐 horizon 训练诊断与 analysis-only config。
- [x] 日志代码覆盖第 8 节的逐位置路由、skill/position、world horizon、执行、branch norm、梯度、延迟与参数量字段，并通过 analyzer mock test。
- [ ] 正式长训练 backend 已切换为 map-style `mowe_feature_store_v1`；训练进程不 import TensorFlow、不解码视频、不实例化冻结 OpenVLA 7B。
- [ ] 至少 100 个真实窗口通过 RLDS/raw-backbone 与 feature-store 的特征、输出和 loss 等价性审计。
- [ ] episode-aware 8-rank assignment 的集合互斥、union 完整、window/suite/skill 负载和 sampler resume 已通过真实数据审计。
- [ ] CPU-only 与 GPU 8-rank continuous soak 证明 `anon`/working set 在 warmup 后平台化，且 `memory.events.oom_kill` 不增加。
- `ARCHITECTURE_RISKS.md` 的自动化硬门槛全部通过。

### 10.1 单机 8 卡 DDP 启动契约

- 启动方式固定为 `torchrun --standalone --nproc-per-node=8`；A100/A800 使用 NCCL + BF16，不引入 FSDP/DeepSpeed。
- `configs/mowe_wam/ddp8_nominal_flow_wam.yaml` 固定每卡 batch 1、accumulation 1，因此有效全局 batch 为 8；`max_steps=1000` 始终是 scheduler horizon，`--stop-step` 只截断本次进程。
- 从旧单卡 checkpoint 迁移必须显式传 `--allow-world-size-change`，且 checkpoint 与当前配置的 effective global batch 必须相同；迁移保持参数、optimizer、scheduler 与 step 连续，不承诺逐样本顺序/RNG 一致。
- DDP 下 `num_workers=0`；在确定性 `_traj_index` 与 CoT sidecar exact join 后、frame transform 前按 episode 分片。feature-store writer 为每个 episode 预计算 target-chunk skill histogram，sampler 使用 deterministic suite/window/skill minimax 分配完整 episode，既保持 ranks 间 episode 不重叠，也降低罕见 expert 只集中于单 rank 的风险；validation 仅由 rank 0 用未分片集合和底层模型执行。
- rank 0 独占目录、resolved config、JSONL 与 checkpoint 写入。checkpoint 保存无 `module.` 前缀组件、distributed contract、各 rank RNG 和 sampler state，并使用临时文件原子替换保护最近恢复点；日志保存跨 rank 汇总指标、episode IDs、RSS、cgroup 与 GPU allocated/reserved 峰值。
- cgroup working set 80%、GPU peak allocated/reserved 85% 与“启动后不得新增 cgroup OOM/OOM-kill”是 runtime 硬门禁，不只是日志字段。DDP 配置默认 `require_cgroup_metrics=true`；current/max/working-set/oom/oom_kill 或 CUDA total/peak 任一不可测时在模型加载前 fail closed。门禁在启动基线、model/data setup、DDP wrap、每个日志步及 rank-0 validation 后由全部 ranks 共同执行，取最坏 rank；Gloo 控制归约固定用 CPU tensor，NCCL 用本地 CUDA tensor。
- 8 卡长期训练前必须依次通过 102-step smoke、102→104 resume、25-step 和 100-step 压力门禁；GPU 峰值须低于单卡容量 85%，cgroup 实际占用须低于上限 80%，且无 rank 数据重叠、NaN/Inf、OOM kill、NCCL/TF/checkpoint 死锁。

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/audit_ddp_runtime.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml \
  --output outputs/preflight/ddp8_runtime_audit.json

torchrun --standalone --nproc-per-node=8 \
  scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam.yaml \
  --data-root /hy-tmp/libero_cot_rlds \
  --checkpoint /hy-tmp/openvla-7b-oft-libero-all \
  --teacher-cache /hy-tmp/mowe_dino_cache_v2_complete \
  --resume outputs/train/stage1_extended_1000_20260716/checkpoint_latest.pt \
  --output-dir outputs/train/stage1_extended_1000_ddp8 \
  --max-steps 1000 --stop-step 102 \
  --save-freq 1 --log-freq 1 \
  --allow-world-size-change --precision bf16
```

### 10.2 长训练数据层迁移：LeRobot 风格归档 + MoWE Feature Store

实现状态（2026-07-17）：`mowe_feature_store_v1` 核心训练路径已经实现，包括 durable episode-boundary staging、只读 mmap Dataset、precomputed-backbone boundary、episode-aware DDP sampler、sampler cursor checkpoint、结构审计、等价性审计脚本和 CPU soak 脚本。Feature writer 对未满 shard 的 pending episode 原子保存索引与 checksum，重启后先恢复；LIBERO converter 用 sidecar-joined source file/traj identity 在 frame transform 前过滤 feature/canonical 都已持久化的 episode，避免恢复时重解码/重编码。LeRobot-v3-style canonical writer 已实现为低维 Parquet + primary/wrist H.264 MP4 + relational episode offsets，并与 feature-store converter 共用一次 RLDS 扫描；本地真实 PyArrow/FFmpeg 的 staging→resume→commit→checksum audit 已通过。三个 Flow-WAM stage 均已有 8 卡 feature-store 配置。真实 RLDS 全量双层转换、100-window 等价性、8-rank CPU/GPU continuous soak 尚未运行，因此仍不得称为真实长训练门禁已通过。

#### 10.2.1 决策与目标

单机 8 卡长期训练不得长期依赖八个独立 TensorFlow/dlimp/RLDS pipeline。扩大节点 RAM 到 512 GiB 或 940 GiB 只能增加安全余量，不能消除 TensorFlow anonymous memory、allocator fragmentation、并发 frame transform 和 rank-0 额外 validation pipeline 的累积风险。

正式长训练数据层采用两层结构：

```text
现有 RLDS + CoT sidecar
    -> 单进程、确定性、可恢复的离线转换器
        -> LeRobot v3 风格 canonical archive
           (Parquet metadata/actions + primary/wrist MP4 + episode offsets)
        -> MoWE feature store
           (paired OpenVLA pooled features + DINO targets + actions/skills)
            -> map-style PyTorch Dataset
            -> episode-aware DDP sampler
            -> 8 卡连续训练
```

其中：

- RLDS 继续作为 source of truth、转换输入和回归审计路径，但不再进入正式长训练进程。
- canonical archive 负责保留原始多模态数据，便于可视化、重新生成特征、未来数据增强和生态兼容。
- feature store 是 Stage 1/2/3 的训练热路径；它不得 import TensorFlow、不得解码视频、不得在每个 rank 实例化冻结 OpenVLA 7B。
- feature-store manifest 是冻结 OpenVLA/DINO identifier 的权威来源；训练节点不需要存在 7B 权重。CLI 可用 `--checkpoint` / `--teacher-checkpoint` 显式声明并与 manifest 交叉校验，配置为 `TBD` 时 runtime 从 manifest 填充 resolved config/checkpoint metadata，禁止把 `TBD` 写进正式训练 checkpoint。
- checkpoint 仍按固定频率保存以容灾，但训练不再依赖每 25 step 主动退出/重启来释放 CPU 内存。

参考实现原则：LeRobotDataset v3 将低维时序信号存入 Parquet，将各 camera 视频存入分片 MP4，并用 relational episode metadata/offset 恢复 episode 视图；其 file-based chunking 避免 episode-per-file 的文件系统压力。MoWE 只采用这一 canonical layout 思路，训练特征使用更适合定长高维 tensor 随机访问的 memory-mapped arrays。

官方参考：

- `https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx`
- `https://github.com/huggingface/lerobot/blob/main/docs/source/porting_datasets_v3.mdx`
- `https://docs.pytorch.org/docs/stable/data.html`

#### 10.2.2 Canonical archive 契约

已实现目录：

```text
mowe_lerobot_v3/
  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/chunk-000/file-000.parquet
  data/chunk-000/file-000.parquet
  videos/primary/chunk-000/file-000.mp4
  videos/wrist/chunk-000/file-000.mp4
```

frame-level Parquet 至少包含：

```text
episode_index: int32
frame_index: int32
timestamp: float64
dataset_name/suite_id
task_id: int32
action: fixed[7] float32
skill_id: int8
skill_valid: bool
source_traj_index/source_file_key
```

episode metadata 至少包含：

```text
episode_id
suite_id
task_id
length
global_frame_start/global_frame_end
data_shard
primary_video_shard/wrist_video_shard
source RLDS manifest fingerprint
sidecar fingerprint
train/validation partition
```

转换器必须从原始 RLDS frame 直接计算正式训练特征，再进行 MP4 编码；不得从有损 MP4 重新生成正式 OpenVLA/DINO cache。若要求像素级可逆审计，应使用无损编码或额外保留原始 frame checksum。

实现文件与提交语义：

- `mowe_wam/data/canonical_archive.py`：episode 独立 staging、Parquet/MP4 chunk writer、统计汇总、manifest/checksum 与结构审计。
- `scripts/convert_rlds_to_mowe_store.py`：同一 deterministic RLDS/sidecar pass 先从原始 tensor 计算 OpenVLA/DINO feature，再将相同原始 primary/wrist frame 送入 canonical writer。
- `scripts/audit_mowe_canonical_archive.py`：核对 committed chunks、Parquet row counts、episode/video offsets、metadata 与可选全量 SHA256。
- chunk payload 只有在 data Parquet、episode Parquet 和两路 MP4 完整写入后才发布 `meta/chunks/chunk-*.json` commit marker；崩溃发生在 marker 前时安全重写该 chunk，发生在 marker 后时清理重复 staging record。
- `source_traj_index` 与 `source_file_key` 在 TensorFlow sidecar exact join 后作为 transport-only 字段保留到两类 store，不进入模型输入。
- PyArrow 与 `imageio-ffmpeg` 仅属于离线转换依赖；import `mowe_wam.data` 或 feature-store 训练不会导入 PyArrow、FFmpeg wrapper 或 TensorFlow。

#### 10.2.3 MoWE feature store 契约

当前主配置满足 `freeze_backbone=true`、`image_aug=false`。因此每个 timestep 可以离线缓存与在线冻结 backbone 数值等价的上下文：

| 字段 | Shape / dtype | 用途 |
|---|---|---|
| `openvla_view_features` | `[2, 4096]`, FP16 | primary+wrist 必须成对经过 fused vision backbone 后再按 view pooling |
| `dino_tokens` | `[16, 384]`, FP16 | current 与 H=1/4/8 future target |
| `action` | `[7]`, FP32 | history action 与 `[t:t+8]` action target |
| `skill_id` | scalar INT8 | per-timestep expert label，unknown=`-1` |
| `task_id` | scalar INT32 | instruction/language feature lookup |
| `episode/frame` | integers | window、partition、sharding 与审计 |

建议布局：

```text
mowe_features_v1/
  manifest.json
  episodes.jsonl
  windows.npy
  tasks.json
  tasks/task-00000.npy
  shards/
    shard-000/
      openvla_views.npy
      dino_tokens.npy
      actions.npy
      skills.npy
    shard-001/
      ...
```

约束：

- shard 必须在 episode 边界切分；建议每 shard 64～128 episodes 或约 1～2 GiB。
- 高维定长 tensor 使用 `.npy` memory map 或等价的只读、定长、可校验格式；禁止训练时 `torch.load` 整个 Python dict shard。
- 当前热路径用小型 `episodes.jsonl` 保存约 1,693 个 episode 的 metadata/offset，用逐 task `.npy` 保存 pooled language feature，避免为训练进程增加 PyArrow 依赖；canonical archive 仍按 10.2.2 的 Parquet 契约实现。`windows.npy` 只存 `(episode_index, step_id)`。
- 所有 shard 先写同目录临时文件，完成 shape/count/checksum 校验后原子 rename；converter 必须支持按已完成 shard resume。
- `manifest.json` 必须记录 OpenVLA checkpoint、processor/transform hash、`num_images_in_input=2`、view order、DINO checkpoint/transform、RLDS manifest、sidecar fingerprint、action statistics、conversion code version 和 feature dtype。

按当前 273,465 个 timesteps 粗估：OpenVLA 双视角 FP16 features 约 4.17 GiB，DINO FP16 tokens 约 3.13 GiB，actions/labels/index 小于 0.1 GiB，总训练 feature store 约 7.3～8 GiB。一个 K=8、long slots=4、future horizons `[1,4,8]` 的 sample 约读取 0.235 MiB，不含小量 Python/tensor metadata。

#### 10.2.4 训练模型接入

已新增 `backbone.mode=precomputed_features`：

- 训练时不构造 `OpenVLAOFTAdapter.model`，Dataset 直接输出 `current_visual_views`、`history_visual_views`、`long_history_visual_views` 和 pooled `language`。
- `FlowWAMSkillPolicy` 从现有 trainable View Fusion 开始执行；nominal flow、memory、WAM、router 和 experts 的结构/参数不变。
- 推理、LIBERO rollout 和在线部署仍使用真实 OpenVLA 7B，从 raw primary/wrist images 在线产生相同上下文。
- feature store 是冻结 backbone 的等价缓存，不得改变论文中的模型输入语义，也不得缓存 action-conditioned/action-answer hidden states。
- 如果未来启用 image augmentation、解冻 vision/language backbone 或训练 proprio projector，必须回退 raw-image path 或重新定义 cache；不得静默复用旧 feature store。

必须增加新旧路径等价性测试：随机抽取至少 100 个真实窗口，同时运行当前 `RLDS -> processor -> OpenVLA/DINO` 路径和 feature-store 路径，比较 view features、language feature、完整 model outputs 与 losses；FP16/BF16 tolerance 和最大偏差写入审计报告。

#### 10.2.5 Map-style window dataset 与 DDP sampler

已新增 `MoWEFeatureWindowDataset(torch.utils.data.Dataset)`，以 `(episode_index, step_id)` 构造与当前 `build_episode_windows()` 完全一致的索引：

```text
current             = t
short history       = [max(0,t-7), ..., t-1]
long history        = sparse_prefix_indices(max(0,t-7), 4)
action target       = [t, ..., t+7]
skill labels        = [t, ..., t+7]
DINO future targets = [t+1, t+4, t+8]
```

第一版不直接使用会补重复 index 的默认 `DistributedSampler(drop_last=false)`。已新增 `EpisodeAwareDistributedSampler`：

- 先按 suite 分组，再按 episode window count 做确定性负载均衡。
- 一个 episode 永久只属于一个 rank；八 rank episode union 必须等于完整 train partition。
- rank 内用 `seed + epoch` 生成 window permutation；保存/恢复 sampler epoch、cursor 和 RNG state。
- 训练仍以 optimizer `max_steps` 为 horizon；每 rank 的 sampler 可循环，但同一全局 epoch 内不得跨 rank 复制 episode。
- 记录每 rank episode count、window count、suite/skill coverage 和 assignment fingerprint。
- checkpoint 额外保存各 rank sampler `epoch/cursor/assignment_fingerprint`；同 world size 精确恢复，world-size 迁移重新分配 episode，不伪称样本顺序连续。

#### 10.2.6 DataLoader 与 CPU 配置

feature-store 第一轮配置：

```yaml
data:
  backend: mowe_feature_store_v1
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  prefetch_factor: null
  max_open_feature_shards: 2
  sampler_shuffle_block_size: 256
```

因为单 sample 只有约 0.235 MiB，先以 `num_workers=0` 建立最低 RSS 基线。sampler 在每个 episode 内确定性混洗，再按 shard 聚合为 256-window block 并全局混洗 block；这样保留跨 shard 的块级随机性，同时避免 `max_open_feature_shards=2` 下逐 sample 随机访问造成 mmap LRU 频繁开关。block size 写入 assignment fingerprint、checkpoint sampler state 和 same-stage resume contract，恢复训练时不得静默改变。只有 GPU data wait 明显时，才依次尝试每 rank `num_workers=1`、`prefetch_factor=2`、`persistent_workers=true`；未经压力门禁不得直接扩到 4～8 workers/rank。

推荐进程环境：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2
```

validation 与 training 共享同一个只读 FeatureStore/manifest，只使用不同 window index，不再构造第二套 TensorFlow pipeline。rank 0 可在原进程验证；若仍观察到 working-set 增长，则退回独立 validation job，但不得要求训练每 25 step 重启。

#### 10.2.7 cgroup 内存观测与门禁

memory-mapped file page cache 会计入 `memory.current`，但可由内核回收；不能继续只用 `memory.current / memory.max` 判断泄漏。资源日志至少记录：

```text
memory.current
memory.max
memory.stat: anon
memory.stat: file
memory.stat: inactive_file
memory.stat: slab_reclaimable
memory.events: high / oom / oom_kill
每 rank VmRSS
```

额外计算：

```text
working_set_estimate = memory.current - inactive_file
```

长训练门禁：

- 512 GiB 与 940 GiB 节点都以实际 container cgroup limit 为准，不以宿主机 `free` 总量替代。
- CPU-only 8-rank dataloader soak 至少覆盖多次 sampler reshuffle；`anon` 和 working-set 在 warmup 后必须同时通过首尾绝对增长与线性回归斜率门禁，不允许用短时抖动掩盖持续线性增长。
- cgroup v2 的 `anon`、working-set、`oom` 与 `oom_kill` 指标缺失时脚本必须失败，禁止把未测量的本地 smoke 当作内存门禁；`oom/oom_kill` 必须保持不变，`high` 增量必须记录并解释。
- GPU 训练 25-step、100-step 后继续做至少 500/1000-step continuous soak；不得仅依据短段外推批准长训练。
- 目标是 512 GiB 节点也能稳定连续训练；940 GiB 仅作为 page cache/系统余量，不作为掩盖泄漏的必要条件。

#### 10.2.8 实施任务与验收顺序

1. [代码完成，真实数据待运行] `scripts/convert_rlds_to_mowe_store.py` 已能单进程读取 RLDS、完成 sidecar exact join、先生成 OpenVLA/DINO/language features、再写 canonical Parquet+MP4，并按 episode/chunk 独立恢复两类输出。恢复时以 source file/traj key 在 frame transform 前过滤两类 writer 都已持久化的 episode；`--limit-episodes` 产物写入 `formal_training_ready=false` 并由 runtime 拒绝。真实 LIBERO 全量双层转换尚未执行。
2. [代码完成，真实数据待运行] converter 从 TFDS statistics 固定 expected episode/frame/window counts，feature writer finalize 与 actual counts 精确比较；提前结束即非 formal。`scripts/audit_mowe_feature_store.py` 继续核对 partition、shape/checksum、多 rank assignment、target-skill union，以及逐 rank window/suite/skill imbalance ratio；真实 1,693 episodes / 273,465 timesteps / 259,921 windows 尚待云端审计。首次真实 audit 先不设 imbalance 阈值，人工核对真实分布后将可接受上限写入正式 runbook；不得凭空选择一个使报告刚好通过的阈值。
3. [代码完成] `MoWEFeatureWindowDataset`、两 shard mmap LRU、完整 window contract 和无 TensorFlow hot-path tests 已实现。
4. [代码完成，真实门禁待运行] `precomputed_features` backbone boundary 与 `scripts/audit_feature_store_equivalence.py` 已实现；100 个真实窗口尚未执行。
5. [代码完成] episode-aware、suite/window/target-skill-aware deterministic minimax DDP sampler、assignment report、cursor/fingerprint checkpoint/resume 已实现并通过合成数据测试。rank 内顺序采用默认 256-window 的 shard-aware block shuffle，减少两 shard mmap LRU 的文件开关；策略与 block size 写入 fingerprint/resume contract。新 store 在 episode metadata 预存 target-skill histogram；旧 store 可从只读 skill mmap 回退重建并在 audit 中产生迁移 warning。
6. [代码完成，真实门禁待运行] `scripts/soak_mowe_feature_store.py` 已记录 RSS、anon/file/inactive_file/working set 与 cgroup events，并用首尾增长、每千 step 线性斜率、最少 post-warmup 采样点及 `oom/oom_kill` 零增量共同判定；缺少 cgroup v2 指标会 fail closed。单 rank/8 rank 长 soak 尚未执行。
7. [本地 2-rank 已完成，真实 8 卡待运行] 小型 feature-store 上的完整 Flow-WAM Gloo DDP step N→N+1 checkpoint/resume 已通过，包含 optimizer/scheduler、两 rank RNG/sampler cursor 和 rank-0 单写连续日志；下一步运行 8 卡 step 100→102 和 102→104。
8. [代码完成，真实证据待生成] `scripts/audit_long_training_readiness.py` 会合并 formal/count/checksum、8-rank assignment、人工确认后的 imbalance limits、100-window equivalence、10k-step/rank CPU soak、8-GPU runtime 与 checkpoint stage/world-size/effective-batch 证据，并要求 feature audit、soak 与训练 config 使用相同的 shard-aware sampler strategy/block size。报告额外绑定 resolved training/store/skill contract 与 checkpoint lineage 的 SHA-256 attestation；feature-store 的同阶段 checkpoint lineage 累计计划运行超过 100 个未认证 optimizer steps 时，训练入口强制要求 `--long-run-readiness-report` 并重新计算上述契约，不能通过反复分成 100-step 任务绕过。任一报告缺失、被修改、对应旧 config/store/checkpoint、指标为 `None`、阈值被放宽或 world size 不是 8 都会失败。
9. 通过 25/100/500/1000-step continuous gate 后，才将 feature-store backend 标记为正式长训练路径。
10. 保留 RLDS backend 作为回归/转换路径；feature-store 验证失败时可回退，但不得把回退路径直接用于未分段长训练。

双层转换命令模板（未在真实 LIBERO 上运行；`LIBERO_CONTROL_HZ` 必须按数据生成配置核对，禁止猜测）：

```bash
export LIBERO_CONTROL_HZ=TBD

python scripts/convert_rlds_to_mowe_store.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root /hy-tmp/libero_cot_rlds \
  --checkpoint /hy-tmp/openvla-7b-oft-libero-all \
  --teacher-checkpoint /hy-tmp/facebook-dinov2-small \
  --output /hy-tmp/mowe_features_v1 \
  --canonical-output /hy-tmp/mowe_lerobot_v3 \
  --canonical-fps "${LIBERO_CONTROL_HZ}" \
  --episodes-per-shard 96 \
  --canonical-episodes-per-chunk 32 \
  --encode-batch-size 16

python scripts/audit_mowe_feature_store.py \
  --store /hy-tmp/mowe_features_v1 \
  --world-size 8 --verify-all-checksums \
  --output outputs/preflight/mowe_feature_store_ddp8_audit.json

python scripts/audit_mowe_canonical_archive.py \
  --archive /hy-tmp/mowe_lerobot_v3 --verify-checksums
```

正式训练证据与启动命令（未在真实 8 卡节点运行）：

```bash
# 1) 真实 raw path 与 feature-store 至少 100 个窗口等价。
python scripts/audit_feature_store_equivalence.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --store /hy-tmp/mowe_features_v1 \
  --data-root /hy-tmp/libero_cot_rlds \
  --checkpoint /hy-tmp/openvla-7b-oft-libero-all \
  --teacher-checkpoint /hy-tmp/facebook-dinov2-small \
  --samples 100 \
  --output outputs/preflight/mowe_feature_store_equivalence_100.json

# 2) 先读取无阈值审计中的真实 imbalance，再由人工确认上限并重跑。
export MAX_WINDOW_IMBALANCE=REVIEW_AFTER_FIRST_AUDIT
export MAX_SUITE_IMBALANCE=REVIEW_AFTER_FIRST_AUDIT
export MAX_SKILL_IMBALANCE=REVIEW_AFTER_FIRST_AUDIT
python scripts/audit_mowe_feature_store.py \
  --store /hy-tmp/mowe_features_v1 --world-size 8 \
  --verify-all-checksums \
  --max-window-imbalance-ratio "${MAX_WINDOW_IMBALANCE}" \
  --max-suite-imbalance-ratio "${MAX_SUITE_IMBALANCE}" \
  --max-skill-imbalance-ratio "${MAX_SKILL_IMBALANCE}" \
  --output outputs/preflight/mowe_feature_store_ddp8_reviewed_audit.json

# 3) 8 个 CPU rank 连续读取；训练热路径不得 import TensorFlow。
torchrun --standalone --nproc-per-node=8 scripts/soak_mowe_feature_store.py \
  --store /hy-tmp/mowe_features_v1 \
  --steps 10000 --warmup-steps 1000 --sample-every 250 \
  --output outputs/preflight/mowe_feature_store_soak_ddp8.json

# 4) 在目标 A100/A800 节点验证 8 rank、GPU 绑定和 cgroup/GPU guard。
torchrun --standalone --nproc-per-node=8 scripts/audit_ddp_runtime.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml \
  --output outputs/preflight/mowe_ddp8_runtime_audit.json

# 5) 合并所有证据。旧单卡 step-100 checkpoint 迁移必须显式授权；
#    readiness 仍会拒绝 effective global batch 不等于 8 的迁移。
python scripts/audit_long_training_readiness.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml \
  --store /hy-tmp/mowe_features_v1 \
  --feature-audit outputs/preflight/mowe_feature_store_ddp8_reviewed_audit.json \
  --equivalence-report outputs/preflight/mowe_feature_store_equivalence_100.json \
  --soak-report outputs/preflight/mowe_feature_store_soak_ddp8.json \
  --ddp-runtime-audit outputs/preflight/mowe_ddp8_runtime_audit.json \
  --checkpoint outputs/train/stage1_extended_1000_20260716/checkpoint_latest.pt \
  --checkpoint-mode resume --allow-world-size-change \
  --output outputs/preflight/mowe_stage1_ddp8_long_run_readiness.json

# 6) 单卡 step 100 -> DDP step 102。feature manifest 绑定 OpenVLA/DINO identifier，
#    因此训练热路径无需加载 7B/teacher，也不需要再次传 --checkpoint。
torchrun --standalone --nproc-per-node=8 scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml \
  --feature-store /hy-tmp/mowe_features_v1 \
  --resume outputs/train/stage1_extended_1000_20260716/checkpoint_latest.pt \
  --output-dir outputs/train/stage1_extended_1000_ddp8_feature_store \
  --max-steps 1000 --stop-step 102 --save-freq 1 --log-freq 1 \
  --allow-world-size-change --precision bf16

# 7) 从新 DDP checkpoint 连续 102 -> 104；同 world size 不再传迁移授权。
torchrun --standalone --nproc-per-node=8 scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml \
  --feature-store /hy-tmp/mowe_features_v1 \
  --resume outputs/train/stage1_extended_1000_ddp8_feature_store/checkpoint_latest.pt \
  --output-dir outputs/train/stage1_extended_1000_ddp8_feature_store \
  --max-steps 1000 --stop-step 104 --save-freq 1 --log-freq 1 --precision bf16
```

102→104 通过后使用同一 output/checkpoint 连续推进，`stop_step` 依次设为 `125`（相对原 step 100 共 25 步）和 `200`（共 100 步）；不要重新使用旧单卡 checkpoint，也不要改变 `max_steps=1000`。每次进程启动计划推进不超过 100 steps 时会被标记为 `bounded_smoke`，无需 readiness report。

在 step 200 检查通过后，用同一组 feature/equivalence/soak/runtime 证据和最新 DDP checkpoint 重新生成 attestation，然后尽量一次连续运行到 step 1000：

```bash
python scripts/audit_long_training_readiness.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml \
  --store /hy-tmp/mowe_features_v1 \
  --feature-audit outputs/preflight/mowe_feature_store_ddp8_reviewed_audit.json \
  --equivalence-report outputs/preflight/mowe_feature_store_equivalence_100.json \
  --soak-report outputs/preflight/mowe_feature_store_soak_ddp8.json \
  --ddp-runtime-audit outputs/preflight/mowe_ddp8_runtime_audit.json \
  --checkpoint outputs/train/stage1_extended_1000_ddp8_feature_store/checkpoint_latest.pt \
  --checkpoint-mode resume \
  --output outputs/preflight/mowe_stage1_step200_long_run_readiness.json

torchrun --standalone --nproc-per-node=8 scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml \
  --feature-store /hy-tmp/mowe_features_v1 \
  --resume outputs/train/stage1_extended_1000_ddp8_feature_store/checkpoint_latest.pt \
  --output-dir outputs/train/stage1_extended_1000_ddp8_feature_store \
  --max-steps 1000 --stop-step 1000 --precision bf16 \
  --long-run-readiness-report outputs/preflight/mowe_stage1_step200_long_run_readiness.json
```

attestation 不允许跨 checkpoint 误用：若选择在 step 500 主动停机，恢复到 1000 前必须针对 step-500 checkpoint 重新生成报告。连续任务仍在 step 500 记录 checkpoint/validation，但无需仅为门禁主动重启。每个观测边界检查 8/8 GPU、NaN/Inf、rank episode 互斥、GPU peak <85%、cgroup working set <80%、`oom/oom_kill` 零增量和参数/checkpoint 连续性。25/100-step 只用于早期故障发现，500/1000-step continuous evidence 才能批准完整长期运行。

在真实转换、100-window 等价性和 continuous soak 完成前，本节状态为“核心代码已实现、真实长训练门禁未通过”；不得把目录、合成测试或 config 的存在视为 feature-store 长训练已经通过。

## 11. 第一版 Definition of Done

第一版代码完成需要：

- nominal-flow、oracle expert warm-start、predicted-route joint 三个主入口均可在真实 LIBERO batch 上完成至少一个 optimizer step。
- 主 config 不依赖 predicate/event/simulator labels；训练期 verb/skill label 可审计且不进入部署输入。
- 推理 wrapper 不实例化 teacher、不生成视频。
- Router 使用简单 query `ActionMLP(A0[j]) + WorldProjection(h_(j+1)) + PositionEmbedding(j)` 输出 `[B,8,7]` schedule；不实现 delta-h query。
- Residual 用共享 6D motion solver 和 per-token motor heads；gripper 使用独立 binary head；null route 仅令 motion residual 为零并保留 nominal action。
- 推理仅交付首个预测 skill 边界前的 1～3 步，禁止开环执行完整 mixed-skill 8-step chunk。
- Checkpoint resume 不重置 action/router/flow solver schedules。
- Future-shuffle、history-only、behavior-prior、dense residual-flow、oracle/predicted/shuffled-skill、copy-current configs 可加载。
- 至少完成 LIBERO one-task smoke 后，才进入完整训练或 benchmark eval。
- CALVIN 作为第二主评测单独交付：正式结果前必须完成 ABC-only full-store audit、CALVIN 专用三阶段 checkpoint、单条五子任务 rollout 和官方 D 环境 1,000-sequence；这些外部门禁不以 adapter/合成测试替代。
- 所有真实命令和观察结果追加到 `DEV_LOG.md`。

## 12. 暂不实施

- 六 motor expert counterfactual rollout；
- 像素级 future-video decoder/diffusion；
- 显式 failure/recovery classifier；
- Simulator predicate/transition 主监督；
- 跨 episode retrieval memory；
- OpenVLA 7B 全量微调；
- 跨节点 DDP、FSDP/DeepSpeed（单机 DDP 已实现）；
- CALVIN 官方 simulator/runtime、真实 ABC→D 训练和正式 1,000-sequence 结果（reader/converter/adapter 代码已实现，但这些外部门禁仍未通过）；

## 13. Handoff Prompt

```text
先阅读 CODEX_PROJECT_RULES.md、PROJECT_PLAN.md、ARCHITECTURE_RISKS.md、IMPLEMENTATION_PLAN.md 和 DEV_LOG.md 最新条目。当前主线把 LIBERO 7D action 拆成 6D motion flow 与独立 1D gripper binary head；六个 motor residual experts 只修正 motion，null_finish 只做 residual bypass。现有 deterministic `latent_wam_residual_moe` 仅是 regression baseline。第一版按配置默认 sidecar timestep aligned，逐 timestep 直接生成 `[B,8]` labels/masks，不增加边界 ±1 mask、soft target 或 debounce。WAM 额外输出 `[B,8,128] h_1...h_8`；router query 严格使用 `ActionMLP(A0[j]) + WorldProjection(h_(j+1)) + PositionEmbedding(j)`，不加入 delta-h/pre-post-state 分支。随后实现 nominal motion-flow + gripper pretrain、oracle per-token warm-start、ST Gumbel joint routing和 hard schedule inference。部署只执行当前 skill 前缀，最多 3 步，然后重规划。长期 8 卡训练的数据主线按 10.2 节迁移为 LeRobot 风格 canonical archive + `mowe_feature_store_v1`；RLDS 只做离线转换/审计，训练热路径不得 import TensorFlow、解码视频或实例化冻结 7B，且必须先通过 100-window 等价性和 8-rank continuous memory gates。第二主 benchmark 固定为 CALVIN LH-MTLC ABC→D；严格 ABC reader/audit、feature converter、独立 action/policy adapter 和三阶段 DDP8 配置已实现，但真实数据转换、训练、simulator 和官方 1,000-sequence 仍是外部门禁。标准 CALVIN 未跑通前不扩展 L-CALVIN，且不得假设 CALVIN/LIBERO action contract 相同。不要修改 external/openvla-oft，不声称未运行的训练或 benchmark 结果。
```
