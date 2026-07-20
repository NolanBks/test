# MoWE-WAM 项目计划

更新时间：2026-07-20（CALVIN Stage 1 多尺度 loss 与 mechanism checkpoint v2 修订）

本文档是项目研究方向、论文故事和实验边界的最高优先级来源。具体代码接口与任务顺序由 `IMPLEMENTATION_PLAN.md` 规定；架构风险和验收门槛由 `ARCHITECTURE_RISKS.md` 规定；实际完成状态只记录在 `DEV_LOG.md`。

## 1. 当前主线

当前主方法为：

> **Nominal Flow Policy + Verb-Seeded Residual Flow Skill Experts + MAW Routing**

中文描述：

> 使用视觉—动作历史和任务指令形成多时间尺度记忆。Nominal policy 以 flow matching 生成 16 步 6D motion chunk，并用独立 binary head 预测 16 步 gripper；轻量动作条件 Latent WAM 预测多 horizon 未来视觉 latent，并额外输出低维逐步 world tokens `h_1...h_16`；MAW router 以 `ActionMLP(A0[j]) + WorldProjection(h_{j+1}) + PositionEmbedding(j)` 为简单逐位置 query，预测 per-token skill schedule。六个 motor residual-flow experts 只修正 6D motion；`null_finish` 表示 residual bypass，不表示停止。部署采用同步 receding-horizon：稳定时提交 8 步，边界信号可疑时缩为 4 步，只有高风险边界才在边界前停止；提交段结束后重新观测和同步推理，旧 chunk 未执行后缀直接丢弃。

旧的 `WorldPredicateHead / WorldTransitionHead + predicate/event memory` 代码保留为 legacy baseline 和历史实现，不再是论文主方法，也不再作为真实训练的默认路径。

第一版唯一冻结 context backbone 固定为 Hugging Face [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) 的原始权重。LIBERO 与 CALVIN 可以共享同一个不可变 base snapshot，但必须分别生成 feature store、action statistics 和三阶段 checkpoint。任何 LIBERO-finetuned OpenVLA-OFT 权重都不进入当前主线的缓存、训练、恢复或评测合同。

当前工程阶段先完成原始-backbone 数据链、单机 8 卡长训练和两个 benchmark 主流程；消融配置与论文对照暂缓，不作为本轮实现或启动长训练的阻塞项。机制正确性仍保留 `future > copy_current`、无泄漏、残差有界等最低门禁。

## 2. 论文标题候选

- MoWE-WAM: Predictive Routing for Verb-Seeded Flow Skill Experts
- MoWE-WAM: Latent World Models for Residual Flow Expert Selection
- Predict the Consequence, Select the Skill: World-Grounded Flow Policies for Manipulation

## 3. 一句话主张

长程机器人操作中的 expert routing 不应只依赖当前观测或无监督 MoE 分工；它应从训练期可审计的子任务谓语 supervision 获得稳定的技能分工，再依据历史交互和 nominal action plan 的预期未来视觉后果，在部署时选择相应的 residual flow skill expert。

## 4. 研究问题

同一张当前图像可能对应不同控制需求：机器人可能刚刚抓稳物体、正在滑落、重复接近却未产生进展，或即将从搬运切换到精细放置。Reactive VLA 或 observation-only MoE router 缺少两个关键信号：

- 当前状态是如何由之前的视觉和动作演化而来的；
- 当前准备执行的短期动作计划可能把世界带向什么状态。

本文研究：

> 一个不生成像素视频、仅预测未来视觉 latent 的轻量世界模型，能否预测 16 步计划内的 skill 时序，并在六类显式初始化的 motor experts 与一个 null route 之间做出更可靠的逐位置选择，从而改善长程、易积累错误的机器人操作？

## 5. 核心假设

### H1：显式 verb/skill warm-start 比完全无监督 MoE 更稳定

仅依赖最终动作误差让 router 与 experts 自行分工，存在 credit assignment 不清和 expert collapse 风险。训练期从每个 timestep 的当前子任务标注最后一句提取首谓语，映射到六类 coarse motor skill 或 `null_finish`，并只更新对应位置的 residual expert，可让每个 expert 先获得可验证的样本覆盖、梯度和控制角色。该标签只用于训练；部署不读取标签。第一版不声称 expert 对应 approach/contact 等逐帧物理阶段。

### H2：未来监督比单纯增加历史编码器更有价值

只增加 temporal Transformer 可能提升容量，但不保证其学习控制相关的物理演化。使用独立视觉 teacher 的未来 latent 监督，应使 memory/WAM 表示包含物体运动、接触变化和任务进展信息。

### H3：Future change 比当前状态更适合作为 router 依据

当前特征主要描述“现在是什么”；`Δz(t→t+H)` 描述“nominal action 预计将改变什么”。Expert specialization 应更自然地围绕不同修正模式形成。

### H4：共享 nominal flow proposal 是动作条件世界预测与 residual flow 修正之间的合理桥梁

完全不输入未来动作时，模型只预测 demonstration policy 下的行为先验；为六个 motor experts 分别 rollout 又过于昂贵且缺少反事实监督。共享 nominal action proposal 只运行一次，使 WAM 的未来预测具有明确控制条件，同时保留第一版的工程可行性。

### H5：Residual flow experts 可以在不重写基础动作的前提下修正控制模式

LIBERO 7D action 拆分为 6D motion 与 1D absolute gripper。最终动作逐位置定义为：

```text
motion_final[j] = clip(A0_motion[j] + R_motion[j])
gripper_final[j] = binarize(sigmoid(g0_logit[j]))
R_motion[j] = FlowExpert_{y_j}[j] when y_j ∈ {0,...,5}, else 0
A_final[j] = concat(motion_final[j], gripper_final[j])
```

`A0_motion` 与 motor residual `R_motion` 由 flow matching generator 产生；gripper 不进入 flow/noise/residual addition，而由共享 binary head 输出 logits。Router 对 16 个位置分别输出 7 路 Top-1：6 个 motor expert + 1 个 `null_finish`。Residual 使用一次共享 6D flow solve，并在每个 token 位置按 skill 选择 adapter/velocity head；禁止分别采样六个完整 residual chunks 后再拼接。`null_finish` 只令 `R_motion=0`，最终仍执行 nominal motion/gripper，不承担 episode termination。系统预测 16 步但最多提交 8 步；低风险 skill 边界允许跨越，可疑边界缩短到 4 步，高风险边界在切换前停止。每轮只使用最新观测同步生成，既不异步生成，也不拼接或复用旧 tail。

## 6. 方法总览

```text
最近 K=8 组 primary/wrist 双视角 + 最近执行动作
                    │
instruction + 稀疏 episode-prefix summaries
                    │
                    ▼
冻结原始 OpenVLA-7B 双视角 visual/language context encoder
                    │
语言条件轻量 View Fusion（初始 0.5/0.5）
                    │
         multi-scale context / memory c_t
                    │
                    ├───────────────┐
                    ▼               │
 Nominal 6D Motion Flow + Gripper   │
  A0_motion_(t:t+15), g0_logits    │
                    │               │
                    ▼               │
      Action-Conditioned Latent WAM │
                    │               │
        world belief w_t            │
       low-dim h_1 ... h_16         │
      z_hat(t+1,t+4,t+8,t+16)       │
        delta-z_hat(t→t+H)          │
        optional uncertainty        │
                    │               │
                    ▼               │
 MAW Router: E_A(A0[j])+W_h(h_(j+1)) │
              + position_j          │
       y_(t:t+15), 7 routes         │
                    │               │
                    ▼               │
 shared 6D residual flow solve with│
 per-token skill adapters/heads ◀───┘
                    │
                    ▼
      final 16-step candidate chunk
                    │
                    ▼
 sync execute risk-gated prefix: 8 / 4 / boundary-stop
```

训练时额外存在：

```text
未来真实图像 I(t+H)
        ↓
冻结视觉 teacher（默认 DINOv2 ViT-S/14）
        ↓
16 个 future spatial latent tokens z*(t+H)
```

Teacher 仅定义训练目标，不进入部署路径。

训练期另有一条不进入部署的 expert-supervision lane：

```text
16 个位置各自的当前子任务 annotation / 可审计离线谓语标注
        ↓
expert_skill_labels y_(t:t+15) ∈ {-1, 0, ..., 6} + per-step mask
        ↓
per-token oracle route + router CE warm-start
```

标签来源必须在数据审计中明确记录。`libero_cot_rlds` 默认只读取当前帧标注最后一句的首谓语，不把整段 `<think>` 文本输入 backbone/router，不使用未来图像、simulator state 或未知语义的 7D action 猜测技能。

## 7. 组件定义

### 7.1 非泄漏 OpenVLA Context Encoder

默认且唯一的第一版 backbone 为原始 `openvla/openvla-7b`。模型按固定 Hugging Face revision 下载为不可变本地 snapshot，冻结全部 7B 参数，只暴露：

- 当前/历史的 `primary` 与 `wrist` 图像分别编码后的视觉特征；
- instruction embedding；

LIBERO 两个视角不在像素空间拼成大图，也不在输入前直接相加。每个视角独立使用原始 OpenVLA 的同一 image processor；随后通过仓库现有的 OpenVLA-OFT-compatible multi-image loader，按 upstream multi-image channel contract 一次进入共享且冻结的原始 OpenVLA vision backbone，输出 token 再按固定顺序 `[primary, wrist]` 分开 pooling。这里复用的是多图加载接口，不是 LIBERO-finetuned OFT 权重。冻结 backbone 之外新增共享的语言条件轻量融合器：

```text
u_v = tanh(W_f LN(f_v) + W_l LN(f_language) + e_view(v))
s_v = w_score^T u_v
alpha = softmax([s_primary, s_wrist])
f_fused = alpha_primary * f_primary + alpha_wrist * f_wrist
```

同一模块用于 current、short history 和 sparse long history。`w_score` 零初始化，使训练起点严格为 `0.5/0.5`，避免随机屏蔽一个视角。融合权重可诊断但不作为因果解释；论文若要声称视角贡献，仍需 primary-only、wrist-only 或 view-mask 评估。

Context feature 必须在 action target token 之前提取。旧代码中的 `last_action_hidden` 可能包含 teacher-forced action token 信息，不得继续作为 WAM/router 的默认输入。

### 7.2 多时间尺度 Memory

第一版 memory 保持最小，但必须名实相符：

- Short memory：最近 8 个连续视觉 tokens 与最近 7/8 步真实执行动作。
- Long context：episode instruction token + 4 个来自更早 episode prefix 的稀疏 summary tokens。
- Optional proprio：只允许通用 robot state，例如 EEF/gripper state；不使用 object/goal simulator state。
- 每个 episode reset；不做跨 episode retrieval。

第一版不在 memory 中输入自然语言 subtask summary。训练期允许逐位置 `expert_skill_labels` 监督 router/expert，但该标签不作为部署期输入，也不进入 memory。默认路由空间为六个 coarse motor skills 加一个无参数 null route：

1. `pick_grasp`：pick、grasp、grab、lift；
2. `place_release`：place、put、release、set、stack；
3. `move_transport`：move、carry、bring、position、align、approach、reach；
4. `open_close`：open、close；
5. `turn_rotate`：turn、rotate；
6. `push_pull`：push、pull；
7. `null_finish`：finish、stop、done、check、hold；不对应可学习 expert，residual 恒为零。

未命中白名单的词映射为 `unknown=-1`，只屏蔽该 timestep 的 route/expert loss，不再因为 chunk 内存在 skill 切换而屏蔽整个 16 步样本。

对象和目标参数不进入 expert ID，但仍由 instruction、视觉 context、memory 与 MAW belief 提供。例如 `pick cup` 与 `pick bowl` 共享 `pick_grasp` expert，动作仍由对象位置与状态条件化。

### 7.3 Nominal Motion Flow 与 Gripper Head

共享 nominal motion-flow head 根据当前 context、memory、连续 flow time 与 action-noise sample 预测 velocity field，并以固定步数 ODE solver 生成 6D motion；独立 gripper head 从相同 context/memory 生成 8 个 binary logits：

```text
A0_motion ∈ R^(8 × 6)
g0_logits ∈ R^(8 × 1)
A0 = concat(A0_motion, sigmoid(g0_logits)) ∈ R^(8 × 7)
```

只有前 6 维 motion 在联合归一化 action space 上以 conditional flow matching 训练；第 7 维 gripper 使用数据集与环境 adapter 明确定义的 canonical absolute binary 语义，以 `BCEWithLogits` 训练，并在环境 adapter 中完成 binarize/sign conversion。MoWE 不使用原始 OpenVLA 的 action head 或其 per-robot unnormalization key。推理使用固定 random seed/solver schedule 以便 motion residual target 与诊断可复现。WAM/router 的 nominal action token 同时编码 `A0_motion[j]` 与 `sigmoid(g0_logits[j])`。

Residual expert 的训练 target、采样 endpoint 与诊断输出都按每个 timestep 的 6D L2 范数投影到 `max_residual_l2=0.5`；随后与 nominal motion 相加并裁剪到 `[-1,1]`。这是第一版在 normalized motion space 中的保守安全默认值，防止尚未充分训练的 expert 覆盖 nominal proposal；只有真实 action statistics、clip fraction 和 simulator 结果共同支持时才调整。

### 7.4 Visual Target Encoder

默认 teacher：冻结 DINOv2 ViT-S/14。

第一版不使用单个 pooled vector，而将原生 patch grid 通过固定 `4×4` adaptive average pooling 得到：

```text
z* ∈ R^(16 × 384)
```

DINOv2 ViT-S/14 的原生 token dim 为 384，因此默认 target 不增加可学习投影；student WAM 使用独立 predictor 从 hidden 512 映射到 384。第一版只编码 `primary` 当前帧和 `H=[1,4,8,16]` 的未来帧，以构造 future target 和 delta target；`wrist` 参与 policy context，但不复制一套 teacher target。这样保持 world target 语义与缓存成本稳定。接口必须允许后续切换到 SigLIP2、V-JEPA 或 VAE encoder，但第一版不同时维护多个 teacher。

### 7.5 Latent World Action Model

默认是 20M～30M 级 causal Transformer：

- hidden dim：512；
- layers：6；
- heads：8；
- MLP ratio：4；
- 输入：OpenVLA context、short/long memory、nominal action tokens、horizon tokens；
- 输出：`w_t`、低维逐步 world tokens `h_1...h_16`、四个 horizon 的 future spatial tokens与 delta tokens；
- uncertainty head 默认关闭，仅作为可选异方差回归扩展。

`h_k ∈ R^128` 表示 nominal action prefix 到第 `k` 个位置的轻量未来 world state。高维视觉监督放在 `H=[1,4,8,16]`，并由 `h_1/h_4/h_8/h_16` 解码/投影得到对应 spatial future/delta targets；不新增 16 组高维视觉 latent。

第一版不包含像素 decoder、视频 diffusion、6B DiT 或 Fast-WAM MoT 复刻。

### 7.6 Future-Grounded Predictive Router

Router 主要读取：

```text
w_t 的窄投影
multi-horizon future delta tokens
pooled future tokens
multi-scale memory
optional uncertainty
h_1 ... h_8
```

完整当前 OpenVLA feature 不得直接无损送入 router，以减少其绕过 future prediction 的捷径。第一版固定使用简单 query；对 `j∈{0,...,7}`：

```text
q_j = ActionMLP(A0[j])
    + WorldProjection(h_(j+1))
    + PositionEmbedding(j)
router_logits[j] = Router(q_j, memory, pooled_future_context)
```

三项先投影到相同 hidden dimension 后再相加，不是把原始 7D action 与 world vector 直接数值相加。第一版不加入 `h_(j+1)-h_j`、前后状态 pair 或其他 transition-difference query。Router 输出 `R^(8×7)`，位置 0 是当前 skill，位置 1～7 是未来 schedule；warm-start 用逐位置 masked CE，联合训练用 straight-through Gumbel-Softmax，推理逐位置 hard Top-1。

### 7.7 Residual Action Experts

第一版定义 6 个显式初始化的 6D motor residual-flow experts。它们共享 motion-flow trunk、flow time/noise/solver，仅保留独立 skill adapter 与 velocity head。`null_finish` 不创建第七个 expert 参数。对每个 flow step `s` 和 action 位置 `j`：

```text
v_i[j] = FlowExpert_i(shared_features[j], context, memory, A0_motion, r_s, s)
v[j] = sum_i gate[j,i] * v_i[j]       # i 仅遍历 6 个 motor experts
v[j] = 0 when route[j] = null_finish
```

最终动作：

```text
motion_final[j] = clip(A0_motion[j] + R_motion[j])
gripper_final[j] = binarize(sigmoid(g0_logit[j]))
A_final[j] = concat(motion_final[j], gripper_final[j])
```

六类 motor skill 在 warm-start 前由首谓语映射定义。warm-start 中每个有效 timestep 只更新其标签对应的 6D motor expert；gripper head 始终是共享独立头，不由 expert residual 修正。`null_finish` 只监督 router 且强制 `R_motion=0`，其语义固定为 residual bypass，不表示 stop/done。联合微调后仍必须报告逐位置 label/route confusion、每个 motor expert 使用率与 motion endpoint error、gripper accuracy、null bypass 正确率及 oracle-route 上界。

## 8. 训练路线

### Stage 0：数据与特征准备

- 混合四个 LIBERO suites：spatial、object、goal、LIBERO-10。
- 构造同 episode 的多时间尺度窗口和 `H=[1,4,8,16]` future frames。
- 计算统一 LIBERO 6D motion normalization statistics；gripper 保持 canonical binary/absolute 语义，不进入 motion statistics。
- 固定原始 `openvla/openvla-7b` 的 repo id、Hugging Face revision、本地 snapshot identity 和权重 fingerprint；在此 identity 上完成 primary/wrist 双视角真实 forward smoke 后，才允许生成正式 OpenVLA feature store。
- 冻结 teacher future tokens 使用按 episode/timestep 去重的 float16 分片离线 cache；冻结 OpenVLA pooled visual context 使用进程内有界 LRU，复用重叠窗口与重规划中的相同预处理帧。
- 持久化 teacher cache 必须保存 checkpoint、transform、resolution、dataset/sidecar fingerprint 与 episode/step 索引；进程内视觉 LRU 以精确预处理像素 hash 键控，不跨 checkpoint 持久化。
- 旧 LIBERO-OFT backbone 生成的 feature store 与 Stage 1/2 checkpoint 只保留为历史产物，不得改 manifest 后复用；主线必须从原始 backbone 重建 feature store，并从 Stage 1 step 0 新建 checkpoint lineage。
- 第一版按用户设定默认 `libero_cot_rlds` sidecar 与 action timestep 已对齐，直接生成 `expert_skill_labels[16]`、`expert_skill_mask[16]` 与逐位置 `label_source`；该项是设计假设，不在第一版设置额外对齐审计 gate，也不表述为已实证验证。
- 逐 timestep 直接使用标签，不增加边界 `±1` mask、soft target 或去抖规则。Pick→Move、Move→Place 的跨界 chunk 保留训练，只屏蔽 unknown/padding 位置。
- 输出每路样本数、episode 覆盖、连续段长度、逐位置覆盖、类别转换矩阵和人工抽检清单；任何 motor expert 长期为空时不得启动 expert 训练。

### Stage 1：Nominal Flow Policy 与 Latent WAM 预训练

冻结原始 OpenVLA-7B 与视觉 teacher，训练：

- multi-scale memory encoder；
- nominal 6D motion-flow head；
- independent gripper binary head；
- Latent WAM。

nominal motion 使用 conditional flow matching，gripper 使用逐位置 BCE。训练早期用真实 future action chunk 条件化 WAM，随后逐渐切换到 nominal sample；使用 nominal action 时，以 6D motion distance 为主的 gate 降低错误因果配对的 world loss。WAM 同时输出 `h_1...h_16`，并由 `h_1/h_4/h_8/h_16` 预测高维 future/delta targets。expert warm-start 开始前冻结 nominal motion-flow 与 gripper heads，并以固定 solver/noise schedule 生成可复现 `A0`。

CALVIN 的高帧率 static-camera 数据使 H=1 visual delta 接近零，不能让其不稳定方向与 H=4/8/16 等权主导 world gradient。CALVIN v2 保留 H=1 与逐步 `h_1`，但高维 horizon loss 权重固定为 `0.25/1/1/1`；delta Smooth-L1 使用带下限的 batch/horizon RMS normalization，delta cosine 按 target-delta magnitude 连续降权。该规则只改变训练目标，不向部署输入加入 teacher target。

### Stage 2：Label-Seeded Residual Flow Expert Warm-start

加载 Stage 1 checkpoint；保持 nominal flow head 冻结，加入：

- future-grounded router 的 label-supervised head；
- 6 个 motor residual-flow experts + 无参数 `null_finish` route。

对每个有效 timestep，只训练 `y_j` 对应 motor expert 去生成 `R_motion*[j]=A_motion*[j]-stopgrad(A0_motion[j])`，并以逐位置 masked CE 训练 router。`null_finish` timestep 不产生 expert flow loss，motion residual 严格置零；gripper 不进入 residual expert。此阶段以 oracle per-token one-hot routing warm-start，同时分别报告每类 motion-flow endpoint error、gripper accuracy 与各 chunk 位置覆盖。WAM 使用更小学习率，原始 OpenVLA-7B 始终冻结。

### Stage 3：MAW-Routed Flow Experts 联合微调

逐步将 oracle per-token routing 替换为 router 的 straight-through Gumbel-Softmax routing：forward 为 hard one-hot，backward 经 soft probabilities 传递梯度；温度按配置退火。允许对所有训练模块使用小学习率联合微调，但保留逐位置 `L_route`、类别覆盖监控、null-zero 断言和 residual-norm 约束。部署始终使用 hard per-token Top-1。

### Stage 4：Benchmark 微调

在目标 benchmark 上：

- 保留预训练 WAM/memory；
- 微调 nominal head、router、experts；
- WAM 使用较小学习率；
- action dimension/单位不同则新增 benchmark action adapter；
- 第一版不对原始 OpenVLA-7B 开启 LoRA；如未来改变，必须作为新的实验合同和 checkpoint lineage 单独审批。

## 9. 损失函数

Flow matching 的 endpoint 与 velocity target 明确定义为：

```text
A_s = (1-s) * eps_A + s * A_motion*   u_A* = A_motion* - eps_A
R*  = A_motion* - stopgrad(A0_motion)
R_s = (1-s) * eps_R + s * R*          u_R* = R* - eps_R

L_FM_nominal = E ||v_0(A_s, s, context) - u_A*||^2
L_FM_expert  = E sum_j motor_mask[j]
                 ||sum_i gate[j,i] v_i(R_s, s, context, A0_motion)[j] - u_R*[j]||^2
L_gripper    = BCEWithLogits(g0_logits, gripper_target)
```

其中 `i` 只遍历六个 motor experts；`null_finish` 和 unknown/padding 位置不进入 `L_FM_expert`。

```text
L =
    lambda_flow_nominal * L_FM_nominal
  + lambda_flow_expert  * L_FM_expert
  + lambda_gripper      * L_gripper
  + lambda_route        * sum_j label_mask[j] * CE(router_logits[j], expert_skill_labels[j])
  + lambda_world    * sum_H [L_cos(z_hat_H, z*_H)
                              + 0.5 * SmoothL1(z_hat_H, z*_H)]
  + lambda_delta    * sum_H [L_cos(delta-z_hat_H, delta-z*_H)
                              + 0.5 * SmoothL1(delta-z_hat_H, delta-z*_H)]
  + lambda_balance  * L_router_load_balance
  + lambda_residual * ||R||^2
  + lambda_endpoint * L1(motion_final, A_motion*)  # joint 微调期可选弱项
```

默认建议：

```text
lambda_flow_nominal = 1.0
lambda_flow_expert  = 1.0（warm-start 与 joint；若 predicted route 初期不稳再降至 0.5）
lambda_gripper      = 1.0
lambda_route        = 1.0（仅有效 timestep；unknown/padding 屏蔽）
lambda_world    = 1.0
lambda_delta    = 0.5
lambda_balance  = 0.01
lambda_residual = 0.001
lambda_endpoint = 0.0（warm-start）；0.05（joint 起点）
```

上述是通用/LIBERO 默认。CALVIN v2 仍保留 `lambda_delta=0.5`，但内部先应用 `H=[1,4,8,16]` 的 `0.25/1/1/1` 权重、per-horizon RMS normalization 与 magnitude-aware cosine；避免通过简单降低整体 delta 权重掩盖短时近零 target 的数值问题。CALVIN Stage 1 同时保留 total-loss best 与 mechanism best，后者按 H=4/8/16 absolute future 相对 copy-current 的门禁语义选择并作为 Stage 2 predecessor。

不得继续将 predicate/progress/risk/event 或 simulator-state label loss 放入主方法默认总损失。Flow losses 和 endpoint L1 只覆盖 6D motion；gripper 只由 binary BCE 监督。`expert_skill_labels` 是训练期逐 timestep verb/skill supervision，不是部署输入。`null_finish` 没有可学习 residual loss，代码必须断言其 motion residual/velocity 为精确零；它不将 nominal action 置零，也不触发 termination。

## 10. 推理流程

```text
current image + cached visual/action memory + instruction + optional proprio
        → frozen OpenVLA context
        → nominal flow sample A0
        → small Latent WAM(A0)
       → future-delta temporal router → 16-step hard skill schedule
        → one shared residual flow solve with per-token skill heads
        → final 16-step candidate chunk
        → synchronously execute risk-gated prefix (8 / 4 / boundary-stop)
```

推理时：

- 不读取未来真实图像；
- 不运行 DINO/VAE teacher；
- 不生成未来 RGB/video；
- 不读取 simulator state、predicate、expert label 或 JSONL labels；
- 只检查前 8 个候选动作中的预测 skill change。若没有 high-risk 边界则默认执行 8；任一边界达到 caution 阈值则执行 4；首个 high-risk 边界位于位置 `b` 时执行 `max(1,b)`，在新 skill token 前停止。低 entropy、高 margin、motion/residual 连续的 Pick→Move→Place 可在同一提交段中执行；
- 预测 horizon 固定为 16，单次执行不超过 8；提交段结束后用最新观测同步生成下一 chunk，旧 suffix 直接丢弃。禁止异步生成、旧 tail 复用、temporal ensemble 和 motion-only stitching；
- 每个新帧只编码一次并写入 memory cache。

## 11. 目标贡献

1. **Verb-seeded residual flow skill experts**：通过训练期可审计的逐 timestep 子任务首谓语 supervision 解决无监督 MoE 的初始 credit assignment 与 collapse，同时保留部署期无标签路由。
2. **Future-grounded temporal skill routing**：将多 horizon 未来视觉变化用于预测 16 步 skill schedule，而不是只预测整个 chunk 的单一 expert ID。
3. **Risk-gated synchronous receding-horizon execution**：以 16 步预测提供前瞻信息，稳定时提交 8 步，可疑时缩为 4 步，高风险时在边界前停止；不依赖异步生成或跨 chunk 拼接。
4. **Nominal-action-conditioned latent consequence model**：用单个共享 nominal flow proposal 连接可控未来预测与 residual flow 修正，避免无动作行为先验和六路反事实 rollout 两个极端。
5. **Minimal multi-scale memory**：用短期连续历史和稀疏长期 summary 支持长程决策，不依赖 simulator 标签、检索数据库或部署期 subtask annotation。

## 12. 与相关工作的边界

本次修订不引入外部论文的 Think、语言 CoT decoder、π backbone 或特定 flow 实现；只采用与具体论文无关的训练原则：先用本项目数据中可审计的当前子任务谓语标签稳定 skill expert credit assignment，再在部署期由 MAW router 进行无标签选择。

- Fast-WAM 保留训练期视频建模、推理期不显式生成未来；本项目接受这一效率结论，但使用小型 latent predictor，并将 predicted future change 显式用于 sparse routing，而不是复刻大视频 DiT/MoT。参考：<https://arxiv.org/abs/2603.16666>
- FLARE 将未来 embedding alignment 注入 dense action policy；本项目的区分点是 nominal-action-conditioned future prediction 与 residual expert routing。参考：<https://arxiv.org/abs/2505.15659>
- V-JEPA 2-AC 证明 action-conditioned latent dynamics 可用于规划；本项目不运行 MPC，而使用一次 nominal latent rollout 选择稀疏修正专家。参考：<https://arxiv.org/abs/2506.09985>
- 普通 MoE-VLA 使用当前 token、任务或 embodiment 路由；本项目要求 router 的核心证据是 predicted future change。

## 13. 数据与 Benchmark 策略

第一版论文采用明确的双基准结构，而不是在多个长程 benchmark 之间继续摇摆：

| 层级 | Benchmark | 主要回答的问题 | 正式协议与主指标 | 当前边界 |
|---|---|---|---|---|
| 基础主基准 | LIBERO 四 suites | 基础操作成功率、future/WAM 机制、router/expert 消融与跨 suite transfer 是否成立 | 各 suite task success rate；matched-capacity baseline 与机制消融 | 先完成 Stage 1→3、one-task smoke，再运行可恢复 full-suite evaluation |
| 第二主基准 | CALVIN | memory、skill boundary 与闭环重规划能否改善跨子任务长程状态延续 | 官方 LH-MTLC ABC→D；average sequence length 与连续完成 1/2/3/4/5 个子任务的成功率 | 使用独立 CALVIN 数据/action/checkpoint contract；真实数据、训练和 simulator 仍待验证 |

交付顺序固定为 **LIBERO 训练与机制门禁 → LIBERO 正式评测 → CALVIN ABC 数据审计/转换与三阶段训练 → CALVIN D 官方 1,000-sequence 评测**。CALVIN 是后续正式结果的一部分，但不得在 LIBERO Stage 1 future predictor 尚未明显优于 copy-current、8 卡长期训练门禁未通过时并行启动大规模训练，以免同时放大模型与数据系统两类不确定性。

两个 benchmark 的 frozen-backbone identity 均固定为同一原始 `openvla/openvla-7b` revision；共享 base 权重不等于共享 benchmark 产物。LIBERO 与 CALVIN 的 processor contract、feature-store manifest、action statistics、训练 checkpoint 和 evaluator evidence 必须独立绑定。

### 13.1 预训练数据

第一版使用现有 `modified_libero_rlds` 的四 suite 混合数据，并将 `yinchenghust/libero_cot_rlds` 的 `cot_file.json` 作为训练期 skill annotation sidecar。当前实现按“去除标签、取最后一句、读取 leading verb、应用本文 skill 映射表”重新审计 273,465 条 annotation，得到：`place_release` 92,320（33.76%）、`pick_grasp` 81,137（29.67%）、`null_finish` 49,276（18.02%）、`move_transport` 30,112（11.01%）、`open_close` 8,974（3.28%）、`turn_rotate` 5,395（1.97%）、`push_pull` 4,262（1.56%）、unknown 1,989（0.73%）。该数字取代早期临时解析器的统计快照。代码在 upstream standardization 前读取 `traj_metadata.episode_metadata.file_path`，用 `file_path + deterministic global trajectory index + timestep` 精确 lookup；为保持全局 trajectory index 可复现，原始 TFDS 读取固定为 `shuffle_files=false`、`num_parallel_reads=16`，数据混洗移到 overlay 后的 window 层。历史四-suite audit 已验证 1,693/1,693 个 episode key 与 273,465/273,465 个 timestep key 存在，并在旧 H=8 合同下得到 259,921 个有效 window；unknown 仍为 1,989。该结果只证明结构键可精确关联，新的 H=16/action-chunk-16 formal store 必须重新转换和审计；它也不等于 CoT 语义边界已经实证校准，因此 `alignment_verified=false` 保留。License 与 train/test 隔离仍必须遵守。

同一离线标注中，skill 连续保持至少 2/4/8 步的比例约为 67.58%/44.07%/26.51%。因此“整个 chunk 必须同 skill”的 Prefix Step Mask 会丢弃大量边界样本；当前主线改为逐位置监督 + 风险门控执行。

必须区分两种实验：

- In-domain joint training：在目标 benchmark 的 train split 联合训练，证明方法本身有效。
- Cross-suite transfer：三个 suites 预训练，在留出的第四 suite 上少量微调，证明 WAM/memory 可迁移。

### 13.2 基础 Benchmark

- LIBERO 四 suites：验证基本成功率、future representation、router/expert 消融和跨 suite transfer。
- 仓库提供 Flow-WAM variable-prefix 的 one-task smoke 与可恢复 full-suite evaluator；正式 full-suite 只接受 Stage 3 joint、LIBERO-bound action statistics/checkpoint，每个 task/trial 即时写 JSONL 并绑定 seed、flow seed 与 upstream commit。当前仍无由该入口生成的真实 simulator 数字。

### 13.3 第二主 Benchmark：CALVIN

后续第二个正式评测固定为 **CALVIN**，不再保留 BOSS/CALVIN 二选一。CALVIN 的固定基座、单臂、语言条件闭环连续控制和五子任务连续组合，正好补充 LIBERO 的基础成功率与 suite transfer：LIBERO 负责验证基础操作和机制消融，CALVIN 负责验证跨子任务状态延续、skill boundary、memory 和闭环重规划是否真的改善长程执行。

第一阶段采用 CALVIN 官方 Long-Horizon Multi-Task Language Control（LH-MTLC）协议，并以 **ABC→D** 作为目标泛化设置：只使用 A/B/C 的训练数据和允许的语言标注，最终在 D 环境按官方生成的五子任务序列评测。主指标固定为平均连续完成子任务数，以及至少连续完成 1/2/3/4/5 个子任务的成功率；同时报告每类子任务成功率、失败位置分布、重规划次数和实际执行前缀长度。正式数字必须来自官方评测器，不得用训练窗口 accuracy 代替。

CALVIN 接入必须复用官方 custom-policy 的 `step(obs, goal)`，并显式处理当前官方实现的生命周期差异：`evaluate_policy.py` 在每个 subtask 前调用 `model.reset()`，而 simulator environment 只在五子任务 sequence 开始时 reset。MoWE bridge 因此在环境 reset 处调用 `reset_sequence()` 清空 episode memory，把官方 subtask `reset()` 解释为清空旧 goal/action suffix，并必须同时报告“每 subtask 清空 memory”的公平性消融。静态相机和 gripper 相机分别映射为 MoWE 的 primary/wrist view；action adapter 需要显式审计相对/绝对坐标系、位置与旋转缩放、Euler/axis-angle 表示、gripper 开合符号、控制频率和 action chunk 队列语义，不能假设 CALVIN 7D action 与 LIBERO 7D action 可直接互换。官方仓库与论文：<https://github.com/mees/calvin>、<https://arxiv.org/abs/2112.03227>。

L-CALVIN 只作为标准 CALVIN 跑通后的可选扩展；BOSS、VLABench 和 RoboTwin 不进入第一版主结果表。当前仓库已实现 dependency-light 的 CALVIN action/policy adapter、官方 evaluator bridge、严格 `task_ABC_D/training` language-segment reader/audit、ABC-only action statistics、单次扫描 canonical+feature 双层 converter、benchmark-specific raw/cache equivalence audit、三阶段 DDP8 配置和 synthetic contract smoke。canonical archive 保留 static `200×200` 与 gripper `84×84` 的独立原始 camera shape；reader 会拒绝 validation/D，有限 segment 转换产物会标记为 smoke-only 并被训练 runtime 拒绝。但仓库仍无真实 CALVIN 数据全量审计/转换、环境、ABC→D 训练 checkpoint 或 simulator 结果；因此只能称为“训练与评测接口代码已实现、真实 benchmark 待验证”，不得写成 CALVIN 已经可运行或已有结果。

## 14. 主要指标

- 任务成功率、长程连续完成长度或 completed subgoals；CALVIN 固定报告 average sequence length 与连续完成 1/2/3/4/5 个子任务的成功率。
- 扰动条件下成功率和闭环纠错次数。
- 6D nominal/residual motion-flow loss、motion solver endpoint L1、gripper BCE/accuracy 与 final motion endpoint error。
- Future latent cosine、Smooth L1、delta error；按 horizon 和 motion magnitude 分桶。
- Router 的逐位置 accuracy、macro-F1、每位置 entropy、skill-boundary precision/recall、schedule edit distance、当前 skill accuracy 与 label/route confusion matrix。
- Oracle temporal schedule 与 predicted temporal schedule 的 success / endpoint gap。
- 实际执行长度分布、越过真实 skill boundary 的比例、重规划频率与 full-8-open-loop safety diagnostic。
- `||motion_final-A0_motion||`、每个 expert motion-residual magnitude、`h_1...h_16` norm/variance。
- Future-shuffle、current-copy、history-only 的性能变化。
- 每步延迟、WAM/teacher/backbone 参数量与峰值显存。

在没有失败/扰动数据前，不报告 failure probability 或 recovery success 学习结论。

## 15. 必须完成的 Baselines 与 Ablations

### Policy / Router Baselines

- Frozen original OpenVLA-7B / dense nominal policy。
- Nominal flow policy（无 expert）。
- Dense residual flow policy，参数量与 MoE 匹配。
- Oracle per-token skill schedule residual flow routing（expert 分工上界）。
- Predicted per-token skill schedule residual flow routing + synchronous risk-gated execution（主方法）。
- Observation-only MoE。
- History-only MoE，无 future teacher loss。
- Behavior-prior future router，不输入 nominal action。
- Nominal-action-conditioned future router，主方法。

### World Modeling Ablations

- No future loss。
- Current-copy predictor。
- Pooled teacher target vs 16 spatial tokens。
- Horizons `[1]`、`[1,4]`、`[1,4,8]`、`[1,4,8,16]`。
- No delta loss。
- Future latent shuffle/mask。

### Memory / MoE Ablations

- No memory。
- Short memory only。
- Short + sparse long memory。
- 合并 taxonomy vs 六 motor experts（仅在主方法稳定后的后续实验；当前不作为实现 blocker）。
- Oracle label vs predicted label vs shuffled label。
- 无 `L_route` 的无监督 router。
- 单一 chunk route vs per-token temporal schedule。
- 固定执行 4/8 步 vs 默认 8、caution 4、high-risk boundary-stop。
- Full action flow experts vs residual flow experts。

## 16. 论文主张边界

允许主张：

- training-time future visual latent supervision；
- nominal-action-conditioned latent world prediction；
- future-grounded residual expert routing；
- training-time verb/skill-supervised expert warm-start；
- 6D nominal/residual motion conditional flow matching + independent binary gripper head；
- inference-time no video decoding；
- multi-scale episodic context；
- 经实验支持后的 long-horizon improvement。

第一版不得主张：

- 六 expert 精确反事实规划；
- 像素级未来视频想象；
- 通用世界模型；
- 无失败数据时的 failure-aware/recovery-aware policy；
- 没有标签审计与 oracle-route 证据时的可泛化 semantic expert 结论；
- 未运行 benchmark 的性能结论。

## 17. Figure 与 Table 规划

- Figure 1：nominal flow proposal → latent consequence prediction → 16-step skill schedule → shared per-token residual flow → risk-gated synchronous execution，并以虚线标出训练期 verb-seeded warm-start。
- Figure 2：short/long memory、multi-horizon future tokens、skill boundary 与 8/4/high-risk 闭环更新时序。
- Figure 3：一条长程轨迹中的 predicted delta、router 切换、nominal/final action 差异。
- Table 1：LIBERO 基础结果与 matched-capacity baselines。
- Table 2：CALVIN LH-MTLC ABC→D 长程主结果（average sequence length、连续完成 1/2/3/4/5 个子任务的成功率及 per-task diagnostics）。
- Table 3：skill label、future/action-conditioning、memory、flow expert 消融。
- Table 4：预测误差、oracle/predicted route gap、expert usage、延迟、显存和参数量。

## 18. 风险与停止条件

完整风险见 `ARCHITECTURE_RISKS.md`。以下情况出现时停止长训练：

- Context feature 包含 action target 泄漏。
- `expert_skill_labels` 由未来帧、simulator state 或 test episodes 生成，或把整段 CoT 作为部署输入。
- 六类任一 motor expert 无有效训练样本、其 flow loss 无梯度，或 oracle temporal routing 不优于 nominal flow baseline。
- Dataset window 跨 episode 或未来索引错误。
- WAM 只达到 current-copy 水平。
- Router 在 shuffle future 后几乎无变化。
- 6 个 motor experts 中有长期零使用或零梯度 expert，或 `null_finish` 产生任何非零 motion residual。
- 推理会执行预测 skill 边界之后的动作，或会开环执行完整 8-step mixed-skill chunk。
- Nominal action 与 demonstration 差异很大，却继续用 demonstration future 作为强 world target。
- 真实一个 optimizer step 出现 NaN/OOM，而日志不足以定位。
- feature store、训练 checkpoint 或评测 CLI 使用的 backbone 不是已签发的原始 `openvla/openvla-7b` revision，或尝试从旧 LIBERO-OFT backbone lineage 恢复。
- CALVIN D/validation 数据进入 feature cache、action statistics、early stopping 或 checkpoint selection，或直接复用 LIBERO action normalization/checkpoint 进行正式评测。

## 19. Fallback Plan

若 action-conditioned WAM 不稳定：

1. 保留 nominal flow head，先退回 behavior-prior future prediction，作为明确的弱版本。
2. 将 spatial target 从 16 tokens 降为 4 tokens，保留 delta loss。
3. 将 WAM 从 6 层降到 4 层，先证明 future-grounded routing 机制。
4. 若 predicted router 崩溃，保留 oracle-skill residual flow experts 作为诊断上界；若 oracle 也无增益，则退回 dense residual flow policy，而不是继续调节 router。
5. 若 long memory 数据管线过重，保留 short memory + instruction，并在论文中准确降级为 temporal context。

像素视频生成、六路反事实 rollout 和 7B 全量微调不是第一版 fallback；它们只会扩大问题。

## 20. 后续开发规则

- 不修改 `external/openvla-oft/`；所有适配放在 `mowe_wam/` 和 `scripts/`。
- `IMPLEMENTATION_PLAN.md` 是下一步编码的唯一主参考。
- 旧 predicate/event 模块保留用于 legacy baseline，不再扩展为主数据依赖。
- 每次实现或文档变更追加 `DEV_LOG.md`。
- 本地只记录真实运行的静态/小型检查；云端训练和 benchmark 结果必须附准确命令与输出路径。
