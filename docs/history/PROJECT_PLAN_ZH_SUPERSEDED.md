# MoWE-WAM 项目计划（旧中文快照，已被替代）

> 本文件保留早期 event/predicate 路线历史，不是当前研究方向。当前权威计划为仓库根目录 `PROJECT_PLAN.md`，新会话不得默认读取本文件。

## 论文标题候选

- MoWE-WAM: Predictive World-Transition Routing for Robust Robotic Manipulation
- Event-Memory Guided Mixture of World Experts for Long-Horizon Manipulation
- Routing Robotic Experts by Predicted Physical Regimes

## 一句话主张

机器人操作策略应当利用语言、轨迹历史，以及预测的下一个物理状态（physical regime）来路由动作专家，而不是仅依赖任务身份（task identity）或单一的当前观测。

## 研究问题

长时域、富接触的操作并非单一的平稳控制问题。同一张当前图像可能需要不同的动作，这取决于机器人是刚刚建立接触、已经完成某个子目标、反复抓取失败，还是即将进入放置阶段。单体式（monolithic）VLA 动作头，或仅以当前观测为条件的 MoE router，都没有显式的接口来做出这种依赖于轨迹的控制模式决策。

本论文提出的问题是：

> 一个紧凑的、对下一个与控制相关的物理状态进行建模的模块，在配合对任务事件的最小记忆的支持下，能否在长时域且易失败的操作中，更可靠地路由专门化的动作专家？

## 范围与边界

第一篇论文不是一篇视频生成式（video-generative）WAM 论文。它不训练或解码未来 RGB 视频，不引入大型视频 DiT，也不主张动作条件的反事实规划（action-conditioned counterfactual planning）。

相反，MoWE-WAM 使用一个紧凑的 **WorldTransitionHead**，从当前 VLA 特征、语言条件的轨迹上下文、近期动作，以及一个小型事件记忆，来预测未来的物理谓词、进度变化、失败风险和恢复可能性。这是一个预测式路由（predictive-routing）接口：它在观测到的轨迹下预测可能的下一个状态，然后让 router 选择下一个控制专家。

选择性的动作条件结果验证（selective action-conditioned outcome verification）仍是一个后续可选扩展。在它被实现和评估之前，不得将其描述为基础方法的一部分。

## 动机

现有的 WAM 工作表明，具备未来感知（future-aware）的表示能够改善控制，但许多方法将其用于视频生成、动作打分，或单个动作解码器。MoWE-WAM 聚焦于一个不同的决策点：专家路由。相关的问题不在于模型能否合成视觉上逼真的未来，而在于它能否足够早地预判一个具有物理意义的状态转移，以便在失败或阶段变化在单帧中变得明显之前切换控制模式。

记忆仅作为一个最小的路由辅助被引入。短期历史捕捉局部运动、接触和停滞；一个小型的事件-谓词记忆（event-predicate memory）记录稀疏的事实，例如成功抓取、失去接触、完成子目标、重试次数，以及恢复开始。它不是一个通用检索库、不是自然语言摘要系统，也不是额外的主要贡献。

## 现有方法的弱点

- 常规的 MoE-VLA 设计可以按任务身份、token 特征、具身（embodiment）或当前观测进行路由，但这些信号并不显式预测下一个物理控制状态。
- 当正确的动作取决于早先的失败尝试、已完成的子目标或停滞的进展时，单张图像是模糊的。
- 整体式（holistic）WAM 能够预测或对未来结果打分，但完整的未来视频生成成本高昂，且可能把容量分配给对控制模式路由并不必要的视觉细节。
- 仅进度（progress-only）的方法无法区分接触、抓取、运输、对齐、风险和恢复等状态转移。
- 通用记忆库可能成本高、难以解释，且容易发生无关检索；长时域路由需要的是稀疏证据，而非无限制的历史存储。

## 提出的方法

MoWE-WAM 由一个冻结的 VLA backbone、一个时序 WorldTransitionHead、一个最小事件-谓词记忆、一个预测式 router，以及五个操作专家组成。

```text
current RGB / language / optional proprio
        -> frozen VLA feature h_t
recent actions + predicate history + Event-Predicate Memory
        -> WorldTransitionHead
        -> future predicates, progress delta, risk, recovery
current feature + predicted transition + memory context
        -> predictive router -> Top-2 action experts -> action chunk
```

### WorldTransitionHead

在时间步 `t`，transition head 消费当前语言条件的 VLA 特征、最近的 `K` 个动作和谓词，以及一个紧凑的记忆状态。它预测一个与动作块（action chunk）对齐的短未来时域 `H`：

```text
p(t + H)
progress(t + H) - progress(t)
risk(t + H)
needs_recovery(t + H)
```

固定的谓词空间保持不变：

- `near_target_object`
- `contact_likely`
- `object_grasped`
- `object_lifted`
- `object_moving_with_gripper`
- `near_goal_region`
- `alignment_required`
- `progress_score`
- `failure_risk`
- `needs_recovery`

### 最小事件-谓词记忆

首个版本只存储一个固定大小的结构化状态，而非过去的图像或检索数据库：

```text
previous expert
last event type
stall duration
retry count
last progress and risk
last grasped state
```

记忆写入器只记录从谓词和轨迹标签派生的事件边界：`grasp_acquired`、`contact_lost`、`progress_stall`、`subgoal_complete` 和 `recovery_started`。在训练时，事件状态从离线标签重建；在部署时，同一个确定性写入器根据预测的谓词和已执行的动作进行更新。

### 预测式 Router 与专家

Router 使用当前 VLA 特征、预测的未来谓词、进度变化、风险/恢复输出、上一个专家，以及编码后的事件记忆。它从五个动作专家中选出 Top-2：

- `E1 Approach`：自由空间接近、粗定位、末端执行器姿态。
- `E2 Contact/Engage`：抓取、触碰、推、拉、按压、把手接合。
- `E3 Transport/Manipulate`：在保持稳定交互的同时移动受控物体。
- `E4 Align/Place/Insert`：目标区域对齐、放置、插入、释放。
- `E5 Recovery`：抓取失败、物体掉落、进度停滞、失去接触、重试行为。

基础方法将专家路由保持为核心决策。它不用全局反事实动作重排序器（counterfactual action reranker）替换 router。使用一个 router 持续性或切换惩罚项，以防止专家在事件边界之外快速振荡。

### 训练与部署

模拟器状态或轨迹元数据仅在离线阶段使用，用于派生当前/未来谓词、进度差分、风险/恢复目标、事件标签，以及弱阶段标签（weak phase labels）。部署时仅使用 RGB、语言、可选本体感受（proprio）、动作历史、预测的谓词，以及紧凑的事件状态。

分阶段训练目标结合了：动作模仿、未来谓词预测、进度差分预测、未来风险/恢复预测、弱阶段-router 监督、负载均衡（load balance），以及专家切换正则化（expert-switch regularization）。

## 目标贡献

- 一种时序世界转移路由（world-transition routing）架构，它使用预测的下一个物理状态，而非仅依赖当前观测或任务身份，来选择 MoE 动作专家。
- 一个最小事件-谓词记忆，为长时域路由记录稀疏的成功、失败、停滞和恢复证据，且不使用通用记忆库或视频记忆模型。
- 一条轨迹级伪标签流水线，用于生成未来物理谓词、进度变化、失败风险、恢复，以及事件边界。
- 一项 benchmark 研究，展示未来预测式路由和事件记忆在何处优于仅观测（observation-only）、仅历史（history-only）和当前谓词（current-predicate）路由。
- 可解释性分析，将预测的状态转移、记忆事件、专家切换和恢复行为联系起来。

## 未来扩展与增强点

以下项目不属于基础方法，只应在预测式 router 基线展现出可测量的优势之后再添加。

优先级 1 扩展：

- 选择性动作条件结果验证：当 router 熵、预测风险或事件记忆表明存在关键决策时，让验证器接受、截断或重新路由 router 提出的动作块。这必须保持为一个选择性的慢路径，而非专家路由的替代。
- 更强的恢复专家：从 LIBERO-Plus/CALVIN 中挖掘失败或停滞的 rollout，并在重试和回溯窗口上训练 `E5 Recovery`。
- 以对象为中心的谓词：为杂乱（clutter）和对象替换（object-swap）场景添加目标物体、干扰物体、夹爪-物体，以及目标区域关系。

优先级 2 扩展：

- 在结构化事件状态之外，增加压缩的视觉/关键帧记忆。
- 使用深度、点云、场景流或 3D 关系的几何感知谓词。
- 将较慢验证器的结果离线蒸馏到快速预测式 router 中。
- 对预测的谓词状态转移进行世界价值（world-value）重排序。

优先级 3 扩展：

- 具有共享谓词和事件接口的跨具身（cross-embodiment）专家 adapter。
- 针对状态转移表示的人类视频或无动作（action-free）预训练。
- 聚焦于事件记忆、恢复和专家切换的低成本真实机器人验证。

扩展门槛：

- 在基础的未来预测式 router 于一个小型真实 benchmark 设置中胜过仅观测和当前谓词路由之前，不启动任何扩展。
- 不为机器人控制主张推测解码式（speculative-decoding-style）的保证。
- 每个扩展保持在各自的配置和消融中；在实验存在之前，不将其描述为已完成。

## 与相关工作的区别

- 与常规 MoE-VLA 不同，路由是以对下一个物理状态的预测和稀疏事件历史为条件，而不仅是任务 token 或当前状态特征。
- 与视频生成式 WAM 不同，世界模块预测紧凑的未来谓词，而非解码未来 RGB 视频。
- 与仅进度方法不同，预测的状态包含接触、抓取、运输、对齐、风险、恢复，以及进度变化。
- 与通用记忆增强的 VLA 不同，这里的记忆是一个固定的结构化事件状态，其唯一作用是提供路由证据。
- 与验证式规划方法不同，基础系统使用预测的状态转移直接选择动作专家；任何动作块验证器都是可选的。

论文写作期间需重新查阅的参考来源：

- WAM Survey: <https://world-action-models.github.io/>
- Fast-WAM: local paper `2603.16666v2.pdf`
- VLA Memory Methods Implementation Survey: local project survey
- OA-WAM: <https://arxiv.org/abs/2605.06481>
- EV-WM: <https://arxiv.org/html/2606.13053v2>
- ProgressVLA: <https://arxiv.org/html/2603.27670v1>
- OpenVLA-OFT: <https://github.com/moojink/openvla-oft>

## 预期 Benchmark

主要：

- `LIBERO-Plus` 或 `LIBERO-X`，用于扰动和失败恢复。
- `CALVIN` / `L-CALVIN`，用于长时域操作。

次要：

- `LIBERO-CF` / `LIBERO-Para`，用于语言/对象接地（grounding）。
- `VLABench` 的长时域/富接触子集，如工程时间允许。

所有数据集路径、checkpoint、云端机器详情，以及可运行的 benchmark 命令，在实际验证之前均保持为 `TBD`。

## 主要指标

- 任务成功率和平均完成子目标数。
- 在相机、布局、初始状态和噪声扰动下的成功率。
- 失去接触或进度停滞后的恢复成功率。
- 恢复前的重复失败次数，以及停滞所耗时间。
- 未来谓词/进度/风险预测质量。
- 专家使用熵、阶段-专家对齐、专家切换率，以及切换相对于事件的时机。
- 事件写入器相对于离线轨迹事件的精确率/召回率（precision/recall）。
- 延迟；仅当添加了可选验证器时，才统计慢路径激活率。

## 基线与消融

- 稠密 VLA/WAM 基线。
- Task-ID MoE。
- 仅观测 MoE。
- 当前谓词 MoE。
- 无未来预测的短历史 router。
- 无事件记忆的未来预测式 router。
- 带事件-谓词记忆的未来预测式 router，即本方法（ours）。
- Oracle 未来状态和 oracle 事件记忆路由，作为不可部署的上界。
- 消融：无历史、无进度差分、无风险/恢复、无记忆、无恢复专家、无负载均衡、无切换正则化。

## 图与表规划

- 图 1：backbone、时序 WorldTransitionHead、事件-谓词记忆、预测式 router，以及五个专家。
- 图 2：轨迹时间线，展示未来状态预测、记忆写入、专家激活，以及一次恢复转移。
- 图 3：可选的选择性动作条件验证，仅在实现并验证后展示。
- 表 1：主要的长时域和恢复 benchmark 结果。
- 表 2：路由和记忆消融。
- 表 3：按扰动和失败类型划分的鲁棒性。
- 表 4：未来状态预测、事件记忆、专家切换，以及延迟分析。

## 审稿人风险分析

- 风险：MoE 的新颖性可能受到质疑。缓解：证明是预测的未来物理状态，而非仅 MoE 容量，改善了相对于容量匹配的仅观测和仅历史 router 的路由。
- 风险：未来状态预测可能不准确。缓解：将预测路由与 oracle 未来状态路由进行比较，并报告预测校准（calibration）。
- 风险：记忆带来的增益可能被归因于仅仅增加了更多上下文。缓解：在容量匹配的条件下，比较无记忆、稠密历史、短历史和结构化事件记忆的变体。
- 风险：事件记忆可能有噪声或泄漏模拟器特权信息。缓解：模拟器状态仅用于离线标签；部署时从预测信号重建记忆，并报告写入器质量。
- 风险：原始 LIBERO 可能已饱和。缓解：优先考虑扰动、长时域、失去接触、停滞和恢复子集。
- 风险：专家坍缩或振荡。缓解：负载均衡、router 持续性、切换正则化，以及事件对齐的路由可视化。
- 风险：可选验证器可能被夸大。缓解：在评估之前将其保持在基础方法之外。

## 回退计划

如果时序未来预测较弱，先将目标缩减为未来进度、接触、抓取、接近目标、风险和恢复。如果事件写入器有噪声，保留短动作/谓词历史并禁用长期事件状态。如果 MoE 训练不稳定，保持 backbone 冻结，先训练一个轻量级预测式 router/adaptor，再恢复全部五个专家。如果 benchmark 集成较慢，先从一个小型轨迹级 LIBERO 子集开始，并在实际运行之前不报告任何云端结果。

## 后续规划的实现说明

- 基线 OpenVLA-OFT + world-predicate MoE 代码已在本地存在。下一个编码里程碑是一个带最小事件记忆的、轨迹感知（trajectory-aware）的预测式 router。
- `IMPLEMENTATION_PLAN.md` 是本里程碑文件级的编码参考；`DEV_LOG.md` 只能记录实际的更改和命令。
- 不要修改 `external/openvla-oft/`；将所有 wrapper 和模块添加到 `mowe_wam/` 下。
- 在首个实现中，不要引入大型视频 DiT、通用检索库、自然语言记忆摘要，或新的外部依赖。
- 繁重实验稍后在云服务器上运行；本地工作应聚焦于文档、标签契约、代码结构、语法检查，以及小型冒烟测试。
