# MoWE-WAM 实现计划（旧中文快照，已被替代）

> 本文件保留早期 event/predicate 路线历史，不是当前实现合同。当前权威执行入口为仓库根目录 `IMPLEMENTATION_PLAN.md`，新会话不得默认读取本文件。

本文档是构建 MoWE-WAM 的工程参考。它将 `PROJECT_PLAN.md` 转化为分阶段、可验证的实现工作。

当前重要状态：

- 本工作区现在已有一个本地的 MoWE-WAM 包、mock 数据冒烟路径，以及一个单节点的 OpenVLA-OFT + LIBERO RLDS 训练入口。
- `external/openvla-oft/` 已在本地克隆用于检查上游代码，但它被父 git 仓库忽略，不得直接编辑。
- 用户报告的恒源云冒烟检查已通过：谓词 schema、mock 标签生成、mock 数据集检查、合成前向、谓词头 dry run，以及 router/expert dry run。
- 当前的真实数据路径将一个 OpenVLA-OFT adapter 绑定到上游 RLDS 预处理，但本工作区尚未观察到任何真实的 LIBERO batch、checkpoint 加载或 benchmark 结果。
- **基于最小事件-谓词记忆（Event-Predicate Memory）的未来预测式路由（future-predictive routing）** 代码路径已实现。它既不替换冻结的 backbone，也不替换五专家 MoE 基线。
- 轨迹窗口访问、模拟器派生的未来标签、稳定的 episode 标识，以及真实的 Torch 前向/反向证据仍为 `TBD`，必须在首次真实训练运行之前完成预检（preflight）。
- 所有数据集根目录、checkpoint 路径、云端机器详情、环境名称，以及长时运行命令均为 `TBD`。
- 除非某个实验实际运行并被观察到，否则不应记录任何实验结果。

## 0. 实现策略

### 0.1 基线与下一个最小可行实现

现有基线为：

```text
Frozen OpenVLA-OFT-style VLA backbone
+ world-predicate prediction head
+ Top-2 sparse MoE action experts
+ predicate-conditioned expert router
+ single-node LIBERO RLDS training entrypoint
```

下一个最小可行实现刻意保持小规模：

```text
Frozen OpenVLA-OFT-style VLA backbone
+ recent action/predicate trajectory window
+ fixed-size Event-Predicate Memory
+ temporal WorldTransitionHead predicting p(t+H), progress delta, risk, and recovery
+ predictive Top-2 expert router
```

在本里程碑中，不要添加视频 DiT、通用检索记忆库（generic retrieval memory bank）、自然语言记忆摘要、动作条件候选验证器（action-conditioned candidate verifier）或任何新的上游依赖。选择性验证、以对象为中心的谓词（object-centric predicates）、更强的恢复挖掘（recovery mining）、几何感知输入，以及真实机器人验证仍属于后续里程碑。

### 0.2 默认 Backbone

默认 backbone：`OpenVLA-OFT`。

理由：

- 它是一个强大且被广泛使用的、面向 LIBERO 的 VLA 基线。
- 其公开仓库将用户指向 `SETUP.md` 进行安装，以及 `LIBERO.md` 进行 LIBERO 微调/评估。
- 其方法已经使用了动作分块（action chunking）和高效动作解码，这是添加轻量级 MoE router 的实用基础。

备用 backbone：

- `TBD`，如果 OpenVLA-OFT 在目标云服务器上安装失败。
- 候选备用可能包括基础版 OpenVLA、SmolVLA，或某个评估框架的 model server，但在未更新 `PROJECT_PLAN.md`、本文件和 `DEV_LOG.md` 之前，不要切换 backbone。

### 0.3 Benchmark 顺序

使用以下顺序以避免浪费云端时间：

1. 上游 OpenVLA-OFT 导入/安装检查。
2. 依据其当前 README 进行上游 OpenVLA-OFT LIBERO 冒烟评估。
3. 使用假特征进行 MoWE-WAM 合成前向传播。
4. MoWE-WAM mock 数据两步训练 dry run。
5. MoWE-WAM 单任务 LIBERO 冒烟评估。
6. 小规模 LIBERO-Plus 或 LIBERO-X 扰动子集。
7. LIBERO 路径稳定后再进行 CALVIN/L-CALVIN。

### 0.4 建议的仓库布局

以下布局是建议方案。仅在上游 backbone 被克隆/挂载之后，或用户确认本项目应成为独立仓库之后，再创建它。

```text
.
├── CODEX_PROJECT_RULES.md
├── PROJECT_PLAN.md
├── IMPLEMENTATION_PLAN.md
├── DEV_LOG.md
├── VLA_WM_2026Q2_hotspots_survey.md
├── external/
│   └── openvla-oft/                  # proposed upstream clone; do not edit directly
├── mowe_wam/                         # proposed local package
│   ├── __init__.py
│   ├── predicates/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   └── labeler.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── libero_predicate_dataset.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── event_memory.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── world_head.py
│   │   ├── router.py
│   │   ├── experts.py
│   │   ├── world_transition.py
│   │   ├── predictive_router.py
│   │   └── policy_wrapper.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   └── train_utils.py
│   └── analysis/
│       ├── __init__.py
│       ├── expert_usage.py
│       └── predicate_timeline.py
├── configs/
│   └── mowe_wam/
│       ├── backbone.yaml
│       ├── predicates.yaml
│       ├── train_predicate_head.yaml
│       ├── train_mowe_router.yaml
│       ├── train_predictive_memory_router.yaml
│       └── ablations/
│           ├── dense_baseline.yaml
│           ├── task_id_moe.yaml
│           ├── observation_only_moe.yaml
│           ├── current_predicate_moe.yaml
│           ├── world_predicted_moe.yaml
│           ├── temporal_history_moe.yaml
│           ├── predictive_no_memory_moe.yaml
│           ├── predictive_event_memory_moe.yaml
│           ├── oracle_future_memory_moe.yaml
│           └── oracle_predicate_moe.yaml
├── scripts/
│   ├── check_backbone.py
│   ├── make_predicate_labels.py
│   ├── build_transition_labels.py
│   ├── inspect_predicate_dataset.py
│   ├── inspect_transition_dataset.py
│   ├── preflight_predictive_training.py
│   ├── train_predicate_head.py
│   ├── train_world_transition.py
│   ├── train_mowe_router.py
│   ├── check_predictive_router.py
│   ├── eval_mowe_libero.py
│   └── analyze_mowe_logs.py
└── outputs/                          # proposed local outputs; do not commit large artifacts
```

## 1. 数据与接口契约

### 1.1 核心 Batch 契约

所有 MoWE-WAM 模块都应就此逻辑 batch 达成一致。实际字段名可在检查上游 backbone 后进行调整。

```python
batch = {
    "images": Tensor,          # shape TBD, likely [B, T, C, H, W] or upstream-specific
    "language": Any,           # raw strings or tokenized upstream format
    "proprio": Tensor,         # [B, T, proprio_dim], if available
    "actions": Tensor,         # [B, T, action_dim] or [B, chunk, action_dim]
    "predicates": Tensor,      # [B, T, predicate_dim]
    "progress": Tensor,        # [B, T, 1]
    "risk": Tensor,            # [B, T, 1]
    "task_meta": dict,         # task name, object ids, goal ids, suite info, all optional/TBD
}
```

基线代码目前使用该契约的一个子集。预测式记忆（predictive-memory）里程碑用轨迹窗口字段对其进行扩展：

```python
predictive_batch = {
    **batch,
    "episode_id": Any,                 # stable episode identity; source must be verified
    "step_id": Tensor,                 # [B]
    "history_actions": Tensor,         # [B, K, action_dim]
    "history_predicates": Tensor,      # [B, K, predicate_dim]
    "current_predicates": Tensor,      # [B, predicate_dim]
    "future_predicates": Tensor,       # [B, predicate_dim], labels at t + H
    "progress_delta": Tensor,          # [B, 1], progress(t + H) - progress(t)
    "future_risk": Tensor,             # [B, 1]
    "future_recovery": Tensor,         # [B, 1]
    "memory_state": Tensor,            # [B, memory_dim], reconstructed up to t - 1
    "event_target": Tensor,            # [B], integer event class written at t
    "phase_target": Tensor,            # [B], weak expert-phase label; offline only
}
```

首个预测式实现使用 `K=4` 的历史步和 `H=8` 的未来步。它不需要历史图像张量或缓存的 VLA 特征：当前 VLA 特征、历史动作、历史谓词，以及一个结构化的事件状态足以支持首次实现。

### 1.2 谓词 Schema

首个版本保持小而固定：

```python
PREDICATE_NAMES = [
    "near_target_object",
    "contact_likely",
    "object_grasped",
    "object_lifted",
    "object_moving_with_gripper",
    "near_goal_region",
    "alignment_required",
    "progress_score",
    "failure_risk",
    "needs_recovery",
]
```

预期 API：

```python
def predicate_dim() -> int:
    ...

def predicate_index(name: str) -> int:
    ...

def validate_predicate_tensor(x: torch.Tensor) -> None:
    ...
```

实现规则：

- 值应归一化到 `[0, 1]`。
- 当与任务相关的距离或子目标完成度改善时，`progress_score` 应增大。
- 对于物体掉落、失去接触、碰撞或进度停滞，`failure_risk` 和 `needs_recovery` 应较高。
- 特权模拟器状态（privileged simulator state）仅用于离线标签，绝不作为推理输入。

### 1.3 专家集合

首个版本恰好使用五个专家：

```text
E1 Approach
E2 Contact/Engage
E3 Transport/Manipulate
E4 Align/Place/Insert
E5 Recovery
```

在专家使用分析表明五专家版本过于粗糙之前，不要增加更多专家。

### 1.4 模型输出契约

被包装的模型应返回：

```python
{
    "actions": Tensor,              # final action chunk
    "predicate_logits": Tensor,     # raw world-head outputs
    "predicates": Tensor,           # sigmoid/normalized predicates
    "router_logits": Tensor,        # [B, num_experts]
    "router_probs": Tensor,         # [B, num_experts]
    "topk_experts": Tensor,         # [B, top_k]
    "expert_actions": Tensor,       # [B, num_experts, chunk, action_dim] if materialized
}
```

出于内存控制考虑，首个实现可以只计算 Top-2 专家动作，而非全部专家动作，但分析代码仍应记录 router 概率。

预测式记忆包装器额外返回：

```python
{
    "future_predicate_logits": Tensor, # [B, predicate_dim]
    "future_predicates": Tensor,       # [B, predicate_dim]
    "progress_delta": Tensor,          # [B, 1]
    "future_risk_logits": Tensor,      # [B, 1]
    "future_recovery_logits": Tensor,  # [B, 1]
    "memory_context": Tensor,          # [B, memory_context_dim]
    "event_memory_state": Tensor,      # [B, memory_dim]
    "router_switch_penalty": Tensor,   # scalar or [B]
}
```

该契约不包含任何未来 RGB/视频输出。

## 2. 任务列表

## 任务 0：挂载或克隆 Backbone

目标：
在编写 MoWE-WAM 代码之前建立上游基线。首次真实编码会话应验证上游项目布局和 README 指引。

文件：

- `external/openvla-oft/` （建议）
- `DEV_LOG.md`

预期 API：

- 尚无 MoWE API。
- 预期结果是一份经过验证的上游入口清单：安装文档、训练文档、LIBERO 评估文档、模型/checkpoint 文档。

命令：

```bash
mkdir -p external
git clone https://github.com/moojink/openvla-oft.git external/openvla-oft
cd external/openvla-oft
find . -maxdepth 2 -type f | sort
sed -n '1,220p' README.md
```

然后遵循当前上游的 `SETUP.md` 和 `LIBERO.md`。如果这些文件与预期不同，不要凭空编造命令。

云端命令模板：

```bash
# machine: TBD
# gpu: TBD
# env_name: TBD
# dataset_root: TBD
# checkpoint_path: TBD
cd /path/to/project/TBD/external/openvla-oft
# Follow current SETUP.md exactly.
# Follow current LIBERO.md for the smallest available eval or training smoke.
```

完成标准：

- 目标机器上存在 `external/openvla-oft/`。
- `README.md`、`SETUP.md` 和 `LIBERO.md`（或其当前等价文件）已被识别。
- 最小的上游导入或评估冒烟命令已被复制到 `DEV_LOG.md`。
- 在检查上游指引之前不编写任何 MoWE-WAM 代码。
- 不启动任何长时训练。

后续：
完成本任务后，创建本地 `mowe_wam/` 包骨架，并用 mock 测试实现谓词 schema。

## 任务 1：定义谓词 Schema

目标：
创建稳定的低维世界谓词接口，供标签生成、世界预测、路由、损失函数和分析使用。

文件：

- `mowe_wam/predicates/schema.py` （建议）
- `configs/mowe_wam/predicates.yaml` （建议）
- `scripts/check_predicate_schema.py` （如有用则建议）

预期 API：

```python
PREDICATE_NAMES: list[str]

def predicate_dim() -> int:
    """Return len(PREDICATE_NAMES)."""

def predicate_index(name: str) -> int:
    """Return stable integer index for a predicate name."""

def validate_predicate_tensor(x: torch.Tensor) -> None:
    """Raise if the final dimension does not match predicate_dim()."""
```

配置内容：

```yaml
predicate_names:
  - near_target_object
  - contact_likely
  - object_grasped
  - object_lifted
  - object_moving_with_gripper
  - near_goal_region
  - alignment_required
  - progress_score
  - failure_risk
  - needs_recovery
normalization: zero_one
privileged_state_policy: offline_labels_only
```

命令：

```bash
python -m compileall mowe_wam
python scripts/check_predicate_schema.py
```

完成标准：

- Schema 在不导入上游 VLA 的情况下即可 import。
- `predicate_dim()` 返回 `10`。
- 最终维度为 `10` 的 mock 张量通过验证。
- 最终维度错误的 mock 张量会清晰地未通过验证。
- 本任务不需要任何数据集或模拟器。

后续：
完成本任务后，先用 mock 轨迹实现伪标签生成器，再绑定到 LIBERO 状态对象。

## 任务 2：构建伪标签生成器

目标：
从模拟器状态或轨迹元数据派生谓词、进度和风险标签。首个版本必须支持 mock 状态，以便在未安装 LIBERO 的情况下测试整条流水线。

文件：

- `mowe_wam/predicates/labeler.py` （建议）
- `scripts/make_predicate_labels.py` （建议）
- `outputs/mock_predicates/` （建议的本地输出）

预期 API：

```python
def compute_predicates(
    step: dict,
    next_steps: list[dict],
    task_meta: dict | None = None,
    cfg: dict | None = None,
) -> dict[str, float]:
    """Return normalized predicate labels for one timestep."""

def label_trajectory(
    trajectory: list[dict],
    task_meta: dict | None = None,
    cfg: dict | None = None,
) -> list[dict[str, float]]:
    """Return one predicate dict per step."""
```

首个版本的启发式标签：

- `near_target_object`：当目标位姿可用时，取归一化后夹爪-目标距离的逆值。
- `contact_likely`：当夹爪-物体距离低于阈值，或模拟器接触标志可用时较高。
- `object_grasped`：当夹爪闭合且物体运动跟随夹爪运动时较高。
- `object_lifted`：当目标物体高度超过桌面/物体初始高度阈值时较高。
- `object_moving_with_gripper`：当物体与夹爪的位移方向一致时较高。
- `near_goal_region`：当目标位姿可用时，取归一化后物体-目标距离的逆值。
- `alignment_required`：任务/元数据启发式；对于插入、放置、抽屉、把手以及需要精确目标的任务较高。
- `progress_score`：朝任务目标或子目标方向的归一化改善程度。
- `failure_risk`：对于物体掉落、目标距离增大、碰撞标志或长时间无进展时较高。
- `needs_recovery`：当风险较高且进度停滞或倒退时较高。

未知的 LIBERO 字段：

- 在检查 LIBERO/OpenVLA-OFT 数据结构之前，确切的模拟器状态键为 `TBD`。
- 如果目标物体或目标位姿不可用，labeler 应发出 `TBD` 警告并回退到可用的元数据，而不是悄悄伪造标签。

命令：

```bash
python scripts/make_predicate_labels.py --mock --output outputs/mock_predicates/mock_labels.jsonl
python -m compileall mowe_wam scripts
```

完成标准：

- 一条 mock 轨迹为每个时间步生成一个标签字典。
- 每个标签字典都包含全部 `PREDICATE_NAMES`。
- 所有值均为数值且被裁剪到 `[0, 1]`。
- 缺失的模拟器字段以警告或 `TBD` 形式呈现，而非被忽略。
- 不需要任何长时训练或云端运行。

后续：
mock 标注工作正常后，检查真实的 LIBERO/OpenVLA-OFT 轨迹结构，并实现数据集 adapter。

## 任务 3：数据集 Adapter

目标：
包装 LIBERO/OpenVLA 风格的轨迹，使每个训练样本都包含 images/language/proprio/actions，外加谓词/进度/风险标签。

文件：

- `mowe_wam/data/libero_predicate_dataset.py` （建议）
- `scripts/inspect_predicate_dataset.py` （建议）
- `configs/mowe_wam/dataset_libero.yaml` （建议）

预期 API：

```python
class LiberoPredicateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        predicate_label_path: str | None = None,
        limit: int | None = None,
        cfg: dict | None = None,
    ) -> None:
        ...

    def __getitem__(self, idx: int) -> dict:
        ...
```

预期输出键：

```python
{
    "images": ...,
    "language": ...,
    "proprio": ...,
    "actions": ...,
    "predicates": ...,
    "progress": ...,
    "risk": ...,
    "task_meta": ...,
}
```

实现说明：

- 先从读取任务 2 mock 标签的 mock 数据集模式开始。
- 仅在检查上游文件格式之后再添加真实 LIBERO 支持。
- 在验证之前，不要假定为 LeRobot、HDF5、RLDS 或原始 pickle 格式。
- 尽可能通过包装上游数据集来保留上游预处理。

命令：

```bash
python scripts/inspect_predicate_dataset.py --mock --limit 2
python scripts/inspect_predicate_dataset.py --dataset-root TBD --split train --limit 2
```

完成标准：

- mock 数据集模式在 `--limit 2` 下打印 batch 形状。
- 真实数据集模式要么打印形状，要么以清晰的 `TBD` 消息退出，说明缺失哪个路径/格式。
- 数据集 adapter 在 mock 模式下不导入重型模拟模块。
- 不需要任何长时训练。

后续：
数据集形状已知后，实现合成模型模块，其维度应从配置派生，而非硬编码上游假设。

## 任务 4：模型模块

目标：
在不修改上游源文件的前提下，围绕上游 backbone 实现本地 MoWE-WAM 模块。

文件：

- `mowe_wam/models/world_head.py` （建议）
- `mowe_wam/models/router.py` （建议）
- `mowe_wam/models/experts.py` （建议）
- `mowe_wam/models/policy_wrapper.py` （建议）
- `scripts/check_mowe_forward.py` （建议）
- `configs/mowe_wam/model.yaml` （建议）

预期 API：

```python
class WorldPredicateHead(nn.Module):
    def __init__(self, hidden_dim: int, predicate_dim: int, hidden_layers: list[int]) -> None:
        ...
    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        ...

class ExpertRouter(nn.Module):
    def __init__(self, hidden_dim: int, predicate_dim: int, num_experts: int, top_k: int = 2) -> None:
        ...
    def forward(self, features: torch.Tensor, predicates: torch.Tensor) -> dict[str, torch.Tensor]:
        ...

class MoEActionExperts(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int, num_experts: int) -> None:
        ...
    def forward(self, features: torch.Tensor, router_probs: torch.Tensor, topk_experts: torch.Tensor) -> dict[str, torch.Tensor]:
        ...

class MoWEPolicyWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, world_head: nn.Module, router: nn.Module, experts: nn.Module) -> None:
        ...
    def forward(self, batch: dict, use_oracle_predicates: bool = False) -> dict[str, torch.Tensor]:
        ...
```

特征提取契约：

- 首个合成版本：`features` 是形状为 `[B, hidden_dim]` 的随机张量。
- 绑定上游的版本：添加一个 adapter 方法，从 OpenVLA-OFT 提取正确的隐藏表示。
- 在检查 backbone 内部结构之前，上游提取方法为 `TBD`。

命令：

```bash
python scripts/check_mowe_forward.py --synthetic --batch-size 2 --hidden-dim 1024 --action-dim 7 --chunk-size 8
python -m compileall mowe_wam scripts
```

完成标准：

- 合成前向传播返回全部模型输出契约键。
- `router_probs` 形状为 `[B, 5]`。
- `topk_experts` 形状为 `[B, 2]`。
- `actions` 形状为 `[B, chunk_size, action_dim]`。
- 前向传播在不克隆 OpenVLA-OFT 的情况下即可工作。

后续：
合成前向通过后，实现损失函数并进行 dry-run 训练，然后再绑定到真实 backbone。

## 任务 5：损失函数与训练阶段

目标：
实现分阶段训练损失以及所需脚本，用于谓词头训练、router/expert 训练，以及可选的联合微调。

文件：

- `mowe_wam/training/losses.py` （建议）
- `mowe_wam/training/train_utils.py` （建议）
- `scripts/train_predicate_head.py` （建议）
- `scripts/train_mowe_router.py` （建议）
- `configs/mowe_wam/train_predicate_head.yaml` （建议）
- `configs/mowe_wam/train_mowe_router.yaml` （建议）

预期 API：

```python
def predicate_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    ...

def progress_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ...

def risk_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ...

def action_loss(pred_actions: torch.Tensor, target_actions: torch.Tensor, kind: str = "l1") -> torch.Tensor:
    ...

def load_balance_loss(router_probs: torch.Tensor) -> torch.Tensor:
    ...

def temporal_smoothness_loss(router_probs: torch.Tensor, event_mask: torch.Tensor | None = None) -> torch.Tensor:
    ...
```

训练阶段：

1. 谓词头阶段：
   - 冻结上游 backbone
   - 训练 `WorldPredicateHead`
   - 目标：谓词/进度/风险损失
2. Router/expert 阶段：
   - 冻结上游 backbone
   - 训练 `ExpertRouter` 和 `MoEActionExperts`
   - 目标：动作模仿 + 谓词监督 + 负载均衡（load balance） + 时间平滑（temporal smoothness）
3. 联合微调阶段：
   - 低学习率
   - 训练 world head/router/experts，并可选地训练小型 adapter
   - 除非明确批准并记录，否则不解冻整个 backbone

命令：

```bash
python scripts/train_predicate_head.py --config configs/mowe_wam/train_predicate_head.yaml --mock --dry-run --max-steps 2
python scripts/train_mowe_router.py --config configs/mowe_wam/train_mowe_router.yaml --mock --dry-run --max-steps 2
```

完成标准：

- 两个脚本都支持 `--mock`、`--dry-run` 和 `--max-steps`。
- 两步 dry run 在 CPU 或单 GPU 上无需真实数据集即可完成。
- 损失字典打印所有活跃项。
- 除非在真实数据上训练，否则不声称任何 checkpoint 有用。
- 不启动任何长时训练。

后续：
dry-run 训练工作正常后，创建消融（ablation）配置，使每一项论文对比都能被一致地实例化。

## 任务 6：基线与消融实验

目标：
通过实现配置驱动的变体，使论文主张可被验证。

文件：

- `configs/mowe_wam/ablations/dense_baseline.yaml` （建议）
- `configs/mowe_wam/ablations/task_id_moe.yaml` （建议）
- `configs/mowe_wam/ablations/observation_only_moe.yaml` （建议）
- `configs/mowe_wam/ablations/current_predicate_moe.yaml` （建议）
- `configs/mowe_wam/ablations/world_predicted_moe.yaml` （建议）
- `configs/mowe_wam/ablations/oracle_predicate_moe.yaml` （建议）
- `scripts/print_mowe_config.py` （建议）

变体定义：

- `dense_baseline`：无 MoE；使用上游动作头或单个稠密的本地动作头。
- `task_id_moe`：router 输入仅包含 task id 或 suite id（如可用）。
- `observation_only_moe`：router 使用当前 backbone 特征，不使用谓词。
- `current_predicate_moe`：router 使用预测/当前谓词，但不含未来时域。
- `world_predicted_moe`：主方法；router 使用世界预测的谓词/进度/风险。
- `oracle_predicate_moe`：上界；仅在评估期间使用离线标签用于分析，绝不用于可部署的推理主张。

命令：

```bash
python scripts/print_mowe_config.py --config configs/mowe_wam/ablations/world_predicted_moe.yaml
python scripts/print_mowe_config.py --config configs/mowe_wam/ablations/oracle_predicate_moe.yaml
```

完成标准：

- 每个消融配置都能加载。
- 配置打印结果说明活跃的 router 输入，以及是否使用 oracle 标签。
- Oracle-predicate 模式被清晰标记为不可部署且仅为上界。
- 本阶段不报告任何 benchmark 结果。

后续：
消融可配置后，实现评估包装器和日志 schema。

## 任务 7：评估工作流

目标：
为上游基线和 MoWE-WAM 创建冒烟优先（smoke-first）的评估模板。

文件：

- `scripts/eval_mowe_libero.py` （建议）
- `configs/mowe_wam/eval_libero_smoke.yaml` （建议）
- `configs/mowe_wam/eval_libero_plus_subset.yaml` （建议）
- `outputs/eval/` （建议）

预期 API：

```bash
python scripts/eval_mowe_libero.py \
  --config configs/mowe_wam/eval_libero_smoke.yaml \
  --checkpoint TBD \
  --dataset-root TBD \
  --suite TBD \
  --num-episodes 1 \
  --max-tasks 1 \
  --dry-run
```

执行顺序：

1. 完全按照上游 README 指引运行上游 OpenVLA-OFT LIBERO 冒烟。
2. 以 `--dry-run` 模式运行 MoWE 包装器。
3. 运行一个任务、一个 episode、一个 suite。
4. 运行小规模 LIBERO-Plus/LIBERO-X 扰动子集。
5. 仅在小子集工作正常且云端预算获批后，才运行完整 benchmark。
6. 仅在 LIBERO 路径稳定后再加入 CALVIN/L-CALVIN。

需记录的指标：

- `success`
- `episode_return`（如可用）
- `completed_subgoals`（如可用）
- `expert_usage_counts`
- `expert_usage_entropy`
- `router_top1_timeline`
- `predicate_timeline`
- `failure_or_recovery_events`
- `latency_ms_per_action_chunk`（如已测量）

命令：

```bash
python scripts/eval_mowe_libero.py --config configs/mowe_wam/eval_libero_smoke.yaml --dry-run
python scripts/eval_mowe_libero.py --config configs/mowe_wam/eval_libero_smoke.yaml --num-episodes 1 --max-tasks 1
```

云端命令模板：

```bash
# machine: TBD
# gpu: TBD
# env_name: TBD
# code_snapshot: TBD
# dataset_root: TBD
# checkpoint_path: TBD
# output_dir: TBD
cd /path/to/project/TBD
python scripts/eval_mowe_libero.py \
  --config configs/mowe_wam/eval_libero_smoke.yaml \
  --checkpoint TBD \
  --dataset-root TBD \
  --num-episodes 1 \
  --max-tasks 1 \
  --output-dir outputs/eval/TBD
```

完成标准：

- dry-run 命令打印解析后的配置并干净退出。
- 单 episode 冒烟命令可用，且在实际执行之前标记为 `not yet run`。
- 评估日志包含专家路由信息。
- 在单任务冒烟通过之前，不运行完整 benchmark。

后续：
评估日志存在后，实现用于专家使用和谓词时间线的分析脚本。

## 任务 8：日志与分析

目标：
从训练/评估日志生成面向审稿人的诊断信息。

文件：

- `mowe_wam/analysis/expert_usage.py` （建议）
- `mowe_wam/analysis/predicate_timeline.py` （建议）
- `scripts/analyze_mowe_logs.py` （建议）
- `outputs/analysis/` （建议）

预期 API：

```python
def summarize_expert_usage(log_path: str) -> dict:
    ...

def build_predicate_timeline(log_path: str) -> dict:
    ...

def compute_phase_expert_alignment(log_path: str) -> dict:
    ...
```

命令：

```bash
python scripts/analyze_mowe_logs.py --mock --output-dir outputs/analysis/mock
python scripts/analyze_mowe_logs.py --log-path outputs/eval/TBD/eval_log.jsonl --output-dir outputs/analysis/TBD
```

完成标准：

- mock 日志生成专家使用计数和熵。
- mock 日志生成谓词时间线摘要。
- 真实日志分析为 `TBD`，直到评估日志存在。
- 分析绝不伪造成功率数字。

后续：
分析工作正常后，下一个 Codex 会话应仅使用实际生成的产物或清晰标注的计划图表来更新 `PROJECT_PLAN.md` 的图/表章节。

## 任务 9：可选增强里程碑

目标：
在不阻塞基础方法的前提下，编码未来路线图。

可选里程碑：

- 选择性慢路径验证（Selective slow-path verification）：
  - 仅当 router 熵、风险、接触可能性或进度停滞越过阈值时，才添加更重的验证器。
  - 所需指标：慢路径激活率、延迟、成功率，以及成功率-延迟权衡。
- 更强的恢复专家（Stronger recovery expert）：
  - 挖掘失败/停滞的 rollout，并在失败窗口上训练 `E5 Recovery`。
  - 所需指标：恢复成功率和失败类型分解。
- 以对象为中心的谓词（Object-centric predicates）：
  - 添加目标物体、干扰物体、夹爪-物体，以及目标区域关系谓词。
  - 所需指标：对象替换（object-swap）和语言接地（language-grounding）鲁棒性。
- 推测式世界专家路由（Speculative world-expert routing）：
  - 让一个快速 router 起草一小段专家/动作块，由世界验证器接受、截断或重新路由。
  - 必须被表述为风险受限的推测式控制（risk-bounded speculative control），而非 LLM 式的无损推测解码。
- 几何感知谓词（Geometry-aware predicates）：
  - 如果仅 RGB 的谓词不足，则添加深度、点云、场景流或 3D 关系。

完成标准：

- 在基础方法于小规模 benchmark 冒烟中胜过仅观测路由（observation-only routing）之前，任何可选里程碑都不启动。
- 每个可选里程碑都有自己的配置和消融。
- 在实验存在之前，任何可选里程碑都不被描述为已完成的贡献。

## 阶段 2：基于最小事件记忆的未来预测式路由

本阶段是当前计划中的研究里程碑。它刻意比通用记忆架构更窄，且必须保留现有的 router 优先（router-first）MoE 设计。

当前实现状态（2026-07-13）：任务 11-14 的本地代码脚手架已存在，包括记忆状态、transition/router 模块、配置、标签缓存工具，以及 mock/静态检查。任务 10 仍是外部门槛：本工作区尚未观察到任何真实的 RLDS 轨迹窗口、稳定的 episode 标识符、模拟器派生的未来标签缓存，或 Torch 前向/反向证据。

### 任务 10：轨迹窗口与未来标签预检

目标：

确定所选的 LIBERO RLDS 数据源是否暴露 episode 边界、有序 transition，以及生成未来物理标签所需的模拟器/轨迹信息。不要从无序样本构造伪轨迹窗口。

文件：

- `mowe_wam/data/libero_predicate_dataset.py` （预检后修改）
- `mowe_wam/predicates/labeler.py` （预检后修改）
- `scripts/build_transition_labels.py` （新增）
- `scripts/inspect_transition_dataset.py` （新增）
- `configs/mowe_wam/dataset_libero.yaml` （修改）

每个 `(episode_id, step_id=t)` 所需的离线标签记录：

```python
{
    "predicates_t": p_t,
    "predicates_future": p_t_plus_H,
    "progress_delta": progress_t_plus_H - progress_t,
    "future_risk": risk_t_plus_H,
    "future_recovery": recovery_t_plus_H,
    "event_target": event_at_t,
    "phase_target": weak_phase_at_t,
}
```

首个版本的事件类型固定为 `none`、`grasp_acquired`、`contact_lost`、`progress_stall`、`subgoal_complete` 和 `recovery_started`。从现有的固定谓词 schema 和时间差分推导它们；不要向模型输入添加图像关键帧、原始模拟器状态或新的谓词维度。

预检检查：

```bash
python scripts/inspect_transition_dataset.py \
  --data-root /PATH/TO/modified_libero_rlds \
  --dataset-name libero_spatial_no_noops \
  --history-steps 4 \
  --prediction-horizon 8
```

完成标准：

- 证明数据源保留了有序 episode 窗口，或能被确定性地转换为有序 episode 窗口。
- 离线标签不含 NaN 值，且满足现有谓词范围契约。
- 报告事件率、未来风险率和有效窗口数，且不声称任何策略性能。
- 如果模拟器/轨迹状态不可用，记录该阻碍。不要将 `fallback_predicates_from_action()` 用作主要的未来 transition 监督。

### 任务 11：最小事件-谓词记忆

目标：

实现一个固定大小、可解释的记忆状态，在整条轨迹上提供路由证据。这不是检索库，且首个版本没有学习式的写入器（learned writer）。

文件：

- `mowe_wam/memory/__init__.py` （新增）
- `mowe_wam/memory/event_memory.py` （新增）
- `mowe_wam/data/libero_predicate_dataset.py` （修改以附加离线记忆快照）
- `scripts/check_predictive_router.py` （新增）

API：

```python
class EventMemoryState:
    def reset(self) -> None: ...
    def update(self, predicates, progress, risk, selected_expert) -> None: ...
    def as_tensor(self, device=None): ...

class EventMemoryEncoder(nn.Module):
    def forward(self, memory_state): ...  # [B, memory_dim] -> [B, memory_context_dim]
```

存储的状态仅限于：上一个专家、上一个事件类型、停滞时长、重试次数、上一次进度、上一次风险，以及上一次抓取状态。写入器使用阈值加持续性（threshold-and-persistence）规则，使单步噪声预测无法制造记忆事件。

完成标准：

- `reset()` 使 episode 记忆在各轨迹之间相互独立。
- 位于 `t` 处的离线记忆快照仅使用不晚于 `t - 1` 的事件。
- 合成检查覆盖抓取获得、失去接触、停滞、子目标完成，以及恢复开始。
- 该模块不添加任何新的运行时依赖。

### 任务 12：WorldTransitionHead 与预测式 Router

目标：

仅在新的预测式变体中替换基础方法的单帧 world head。为所有现有基线配置保留 `WorldPredicateHead` 和 `ExpertRouter`。

文件：

- `mowe_wam/models/world_transition.py` （新增）
- `mowe_wam/models/predictive_router.py` （新增）
- `mowe_wam/models/policy_wrapper.py` （修改）
- `mowe_wam/models/__init__.py` （修改）
- `configs/mowe_wam/model.yaml` （修改）
- `scripts/check_predictive_router.py` （新增）

目标 API：

```python
class WorldTransitionHead(nn.Module):
    def forward(
        self,
        current_features,       # [B, backbone_dim]
        history_actions,        # [B, K, action_dim]
        history_predicates,     # [B, K, predicate_dim]
        memory_context,         # [B, memory_context_dim]
    ) -> dict[str, Tensor]: ...

class PredictiveExpertRouter(nn.Module):
    def forward(
        self,
        current_features,
        future_predicates,
        progress_delta,
        future_risk,
        future_recovery,
        memory_context,
        previous_expert=None,
    ) -> dict[str, Tensor]: ...
```

实现约束：

- 初始架构：将 4096 维当前特征投影到 512 维，然后对动作/谓词历史使用一个两层因果 Transformer（causal Transformer）。预期可训练规模约为 6M-8M（不包括现有专家）。
- Router 输入必须使用独立投影或 FiLM/gating。不要直接把 4096 维 VLA 特征与 10 维未来谓词向量拼接后就称之为预测式路由。
- 基础 router 选择 Top-2 专家。首个预测式里程碑保留现有的专家动作实现，且不添加全局候选动作重排序（candidate-action reranking）。
- 添加一个上一专家或切换代价（switch-cost）输入，以抑制在事件边界之外的振荡。

完成标准：

- 合成前向输出 1.4 节中的每个预测式输出，其值有限且形状正确。
- 现有 `WorldPredicateHead` 基线前向保持不变。
- 一个配置开关即可选择基线或预测式记忆包装器，而无需改动上游文件。

### 任务 13：分阶段训练与配置

目标：

先训练 transition 模块，再让预测的 transition 信号去控制 router，然后逐步对齐训练与部署输入。

文件：

- `mowe_wam/training/losses.py` （修改）
- `scripts/train_world_transition.py` （新增）
- `scripts/train_mowe_wam.py` （修改）
- `configs/mowe_wam/train_predictive_memory_router.yaml` （新增）
- `configs/mowe_wam/train_mowe_wam_libero.yaml` （保持与基线兼容）

损失：

```text
action imitation
+ future predicate BCE
+ progress-delta regression
+ future risk BCE
+ future recovery BCE
+ weak phase-router CE
+ load balance
+ expert-switch regularization
```

阶段：

1. 冻结 backbone，在离线 oracle 轨迹窗口上训练 `WorldTransitionHead` 加记忆编码器。
2. 使用 oracle 未来标签/记忆快照进行弱阶段监督，训练预测式 router 和专家。
3. 通过在预测的未来信号上路由进行联合训练。oracle-future 预热（warmup）仍是一个可选的调试/消融控制项，但其默认值为 `0`，以避免训练/部署路由不匹配。首个训练实现使用无泄漏（leak-free）的离线记忆快照；`EventMemoryState.update()` 提供部署时的确定性写入器。学习式的预测记忆 rollout 是后续增强，不得声称已完成。

完成标准：

- 各阶段特定的 checkpoint 内容和配置字段是明确的。
- 优化器仅包含 `requires_grad=True` 的参数。
- 在任何长时运行之前，都能进行一次合成优化器步和一次真实数据 batch 入口检查。

### 任务 14：预测式记忆消融与分析

目标：

使新增的历史、预测和记忆角色可被证伪，而不是只报告一个汇总的成功率数字。

文件：

- `configs/mowe_wam/ablations/temporal_history_moe.yaml` （新增）
- `configs/mowe_wam/ablations/predictive_no_memory_moe.yaml` （新增）
- `configs/mowe_wam/ablations/predictive_event_memory_moe.yaml` （新增；主方法）
- `configs/mowe_wam/ablations/oracle_future_memory_moe.yaml` （新增；不可部署的上界）
- `mowe_wam/analysis/memory_usage.py` （新增）
- `mowe_wam/analysis/predicate_timeline.py` （修改）
- `scripts/analyze_mowe_logs.py` （修改）

所需分析：

```text
future-state prediction quality
event write precision/recall
expert switch count and switch timing
stall duration and repeated-failure count
recovery-expert usage after contact loss or stalled progress
```

完成标准：

- 在可能的情况下，每个消融恰好只在一个 历史/预测/记忆 因素上有差异。
- Oracle 变体被明显标记为不可部署。
- 分析代码可在合成日志上运行，且不伪造 benchmark 结果。

### 阶段 2 冒烟优先顺序

```bash
python -m compileall mowe_wam scripts
python scripts/check_predictive_router.py --synthetic --batch-size 2
python scripts/inspect_transition_dataset.py --mock --history-steps 4 --prediction-horizon 8
python scripts/preflight_predictive_training.py --config configs/mowe_wam/train_predictive_memory_router.yaml --data-root /PATH/TO/modified_libero_rlds --dataset-name libero_spatial_no_noops --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --transition-label-path /PATH/TO/transition_labels.jsonl --backward
python scripts/train_world_transition.py --mock --dry-run --max-steps 2
python scripts/train_mowe_wam.py \
  --config configs/mowe_wam/train_predictive_memory_router.yaml \
  --max-steps 1 --limit-batches 1
```

最后一条命令只是在合适的环境和数据集可用后进行的训练循环入口检查。它不是 benchmark 结果。

## 3. 云端执行清单

在任何云端运行之前，在 `DEV_LOG.md` 中记录：

- machine provider: `TBD`
- machine type: `TBD`
- GPU type/count: `TBD`
- environment name: `TBD`
- code snapshot or commit: `TBD`
- upstream repo commit: `TBD`
- dataset root: `TBD`
- checkpoint path: `TBD`
- exact command: `TBD`
- output directory: `TBD`

最小云端序列：

```bash
# 1. Verify environment.
nvidia-smi
python --version

# 2. Verify upstream backbone.
cd external/openvla-oft
# Follow current upstream setup/import smoke.

# 3. Verify local MoWE package.
cd /path/to/project/TBD
python -m compileall mowe_wam scripts
python scripts/check_mowe_forward.py --synthetic --batch-size 2

# 4. Verify dry-run training.
python scripts/train_predicate_head.py --mock --dry-run --max-steps 2
python scripts/train_mowe_router.py --mock --dry-run --max-steps 2

# 4b. Verify predictive-memory modules before any real run.
python scripts/check_predictive_router.py --synthetic --batch-size 2
python scripts/train_world_transition.py --mock --dry-run --max-steps 2

# 5. Only then run one-task eval.
python scripts/eval_mowe_libero.py --config configs/mowe_wam/eval_libero_smoke.yaml --num-episodes 1 --max-tasks 1
```

停止规则：

- 如果上游 OpenVLA-OFT 导入/评估冒烟失败，停止。
- 如果谓词标签包含 NaN 或越界值，停止。
- 如果合成前向传播失败，停止。
- 如果两步 dry-run 训练失败，停止。
- 如果单任务评估无法产生路由日志，停止。
- 在所有停止规则均被清除之前，不启动全 suite 评估。

## 4. 首个可实现版本的完成定义

首个实现里程碑仅在满足以下条件时才算完成：

- 上游 backbone 已被克隆/挂载，且其最小冒烟路径已记录。
- 谓词 schema 和 mock labeler 工作正常。
- 数据集 adapter 具备 mock 模式，且有清晰的真实 LIBERO 路径或 `TBD` 阻碍。
- world head、router、experts 和 wrapper 通过合成前向检查。
- 损失函数和训练脚本通过两步 dry run。
- 消融配置能实例化主要变体。
- 评估脚本具备 dry-run 模式和单任务命令模板。
- 分析脚本能汇总 mock 路由日志。
- `DEV_LOG.md` 记录所有实际运行的命令。
- 不声称任何虚假的 benchmark 或云端结果。

## 4.1 预测式记忆里程碑的完成定义

预测式记忆扩展仅在满足以下条件时才可进行真实训练：

- 轨迹源和有效的 `(episode_id, step_id)` 窗口已被验证。
- 离线未来谓词、进度差分、风险/恢复，以及事件标签是从合法的轨迹或模拟器信息生成的。
- 事件记忆具有确定性的 reset/write/read 契约，且不会从未来时间步泄漏事件。
- `WorldTransitionHead` 在合成数据和一个真实 batch 上预测所有未来状态输出，无形状或 NaN 失败。
- 预测式 router、基线 router 和 oracle 变体可通过配置选择，且清晰区分。
- 训练日志包含未来状态损失、事件记忆摘要、router logits、所选专家，以及切换信息。
- 在被观察到之前，不记录任何真实训练、benchmark、恢复或记忆改进的主张。

## 5. 移交给另一个 Codex 会话的提示词

在开始一个全新的编码会话时使用此提示词：

```text
Read CODEX_PROJECT_RULES.md, PROJECT_PLAN.md, IMPLEMENTATION_PLAN.md, and the latest DEV_LOG.md entry. Phase 2 code is present but not yet validated on a PyTorch-enabled real-data environment. First run the documented synthetic forward/backward smoke checks, then verify RLDS episode IDs and build a legitimate transition-label cache before launching stage-1 WorldTransitionHead training. Do not invent experiment results. Use TBD for unknown paths, datasets, checkpoints, cloud details, and upstream APIs. Prefer the smallest smoke run before any cloud-scale run. Do not modify upstream OpenVLA-OFT directly; add wrappers under mowe_wam/ unless IMPLEMENTATION_PLAN.md is updated with a specific reason.
```
