# MoWE-WAM 架构风险登记表

更新时间：2026-07-17（单机 DDP/feature store 与 LIBERO + CALVIN 双基准修订）

本文档记录当前主方案“Nominal Flow Policy + Verb-Seeded Residual Flow Skill Experts + MAW Routing”中必须持续检查的概念、训练和评测风险。它不是实验结果，也不是可选的审稿意见清单；`PROJECT_PLAN.md` 负责研究方向，`IMPLEMENTATION_PLAN.md` 负责落地方式，本文档负责定义不得被实现悄悄破坏的边界和验收门槛。

## 1. 当前方法定义

当前主方法固定为：

```text
历史视觉 + 历史真实动作 + 当前图像 + instruction + optional proprio
                              ↓
                 非泄漏 OpenVLA context
                              ↓
       nominal 6D motion flow + gripper binary head
                              ↓
             action-conditioned Latent WAM(A0)
                              ↓
future visual latent / delta + h_1...h_16
                              ↓
      per-token temporal skill router [16 × 7]
                              ↓
       shared residual solve + token skill heads
                              ↓
   motion_final=A0_motion+R_motion; gripper=nominal binary
                              ↓
 sync execute 8 / 4 / high-risk-boundary prefix
```

训练时使用冻结、独立参数的视觉 teacher 提供未来图像 latent 目标，并使用逐位置 `expert_skill_labels[16]` warm-start router 和六个 motor experts；`null_finish` 是无参数 motion-residual bypass。推理时不运行 teacher、不生成视频、不读取 simulator predicate、trajectory JSONL、CoT 文本或 expert-skill 标签。

## 2. 不可违反的架构规则

1. WAM 预测的是 nominal action 的预期后果；第一版不声称比较六个 motor expert 的反事实未来。
2. Router 的主要输入必须是 predicted future change、world belief 和 memory；不能依靠完整当前 VLA feature 绕过 WAM。训练期 label 仅为 supervision，不能成为部署输入。
3. 只有前 6 维 relative motion 使用 conditional flow matching；第 7 维 absolute gripper 必须使用独立 binary head/BCE，不进入 motion normalization、flow noise 或 residual addition。六个 experts 只输出 6D motion residual；Residual trunk/solver 只运行一次，禁止独立采样六个完整 chunks 后拼接。
4. OpenVLA context 必须在 action target token 之前提取；不得继续以 teacher-forced `last_action_hidden` 作为 world/router 唯一输入。
5. Future teacher 只提供训练目标；不得成为推理期必需组件。
6. 不使用 object pose、contact、collision、goal predicate、simulator state 或未来帧作为 skill 标签来源。`expert_skill_labels[j]` 只能来自 timestep `t+j` 的 annotation 最后一句 leading verb 或可审计等价标注，且必须有逐位置 mask/source/fingerprint；整段 CoT 不进入模型，chunk 起点标签不得复制到未来位置。
7. Memory 必须来自过去和当前时刻；不得读取未来帧、未来 action 或未来 teacher feature。
8. 论文不得把 latent prediction 写成像素视频生成，也不得把 nominal consequence 写成精确 counterfactual planning；也不得在 oracle-route 结果之外宣称部署期 router 能读到 expert 标签。
9. `null_finish` 不得实例化可学习 residual head，其 motion velocity/state/residual 必须精确为零，但 nominal motion/gripper 仍保留；不得把它解释为 termination。
10. 预测 horizon 为 16，默认同步执行 horizon 为 8。低风险 skill change 可跨越；caution 边界缩为 4 步；high-risk 边界才在切换前停止。推理不得执行超过 8 步的长开环，也不得复用旧 suffix、做 motion stitching 或异步生成。
11. WAM 输出 `[B,16,128] h_1...h_16`；router query 固定为 `ActionMLP(A0[j]) + WorldProjection(h_(j+1)) + PositionEmbedding(j)`。第一版不得悄悄加入 delta-h 或 pre/post-state query。
12. sidecar lookup 键固定为 `traj_metadata.episode_metadata.file_path + deterministic global trajectory index + timestep`；原始 TFDS 读取固定为 `shuffle_files=false`、`num_parallel_reads=16`，overlay 后才允许 window-level shuffle。1,693/1,693 episode 与 273,465/273,465 timestep 的 exact match 只证明结构关联；`libero_cot_rlds` 与 action 的语义时序对齐仍是第一版设计假设，不增加边界 `±1` mask、soft target、debounce 或额外 alignment gate。

## 3. 风险总表

| ID | 风险 | 严重度 | 第一版处理 | 通过门槛 |
|---|---|---:|---|---|
| R1 | nominal future 与最终 residual action 不一致 | 高 | residual policy、短步执行、闭环重规划 | final action 与示范动作优于 nominal，residual 不失控 |
| R2 | Router 标签与部署期选择脱节 | 高 | per-token verb/skill CE warm-start + oracle-to-ST-to-hard schedule | predicted schedule 接近 oracle schedule，且 shuffle-label 明显更差 |
| R3 | Future latent 退化为复制当前帧 | 高 | spatial tokens、delta loss、多 horizon | copy-current baseline 明显更差 |
| R4 | Router 绕过 WAM | 高 | 限制 current-context skip，router 以 delta 为主 | shuffle/mask future 后 routing 或性能明显下降 |
| R5 | 所谓长期 memory 实际只看 8 帧 | 高 | 多时间尺度窗口；后续再做 recurrent segment | 文档命名与真实时间范围一致 |
| R6 | expert label 覆盖不足、expert collapse 或路由振荡 | 高 | 六 motor + null coverage gate、oracle per-token warm-start、ST Gumbel joint | 每个 motor expert 有样本/梯度，oracle 分工优于 nominal，predicted usage 不退化 |
| R7 | teacher forcing 到 nominal action 的分布偏移 | 高 | 分阶段 scheduled conditioning + action-distance gate | nominal-conditioned world loss 稳定且无 NaN |
| R8 | Teacher target 过度 pooled 或与 policy 表示耦合 | 中高 | 16 个 spatial tokens；teacher 独立冻结 | pooled/spatial teacher 消融 |
| R9 | 6D motion normalization 与 absolute gripper 语义混用 | 高 | motion-only stats + canonical binary gripper adapter | gripper 不进入 flow/residual；motion 在进环境前按 checkpoint q01/q99 反归一化，gripper sign conversion 可测试 |
| R10 | LIBERO-finetuned checkpoint 偏置或混入多-suite/CALVIN 主线 | 高 | 唯一冻结原始 `openvla/openvla-7b`，各 benchmark 独立训练 head/store | 正式 manifest/checkpoint/eval 不含 OFT-finetuned backbone |
| R11 | 历史帧重复运行 7B backbone 导致不可训练 | 高 | instruction cache + 精确像素键控 pooled-visual LRU + teacher 离线分片缓存 | cache hit/eviction 测试通过，且 K=8 batch 能在目标 GPU 完成一步 |
| R12 | 随机窗口无法训练持续 memory | 中高 | 稀疏 episode-prefix summary；在线端复用相同索引 | sample 不跨 episode，在线/训练 prefix indices 一致，memory 无未来泄漏 |
| R13 | 成功示范无法监督 failure/recovery | 高 | 第一版只预测 latent/uncertainty，不声称 failure classifier | 加扰动/失败数据前不报告 recovery 学习结论 |
| R14 | 只在 LIBERO 上无法证明长程主张 | 高 | LIBERO 基础验证 + CALVIN LH-MTLC ABC→D | CALVIN 官方五子任务序列指标单独报告，不以离线 proxy 替代 |
| R15 | 预训练和测试任务泄漏 | 高 | 明确 in-domain 与 leave-one-suite-out/few-shot 协议 | 不读取 test episode，公开数据划分 |
| R16 | Future prediction 多模态导致均值 latent | 中 | `[1,4,8]`、cosine + Smooth L1、可选异方差 | 长 horizon 不比 current-copy 更差 |
| R17 | flow solver、共享 trunk、随机噪声或过大 residual 导致 endpoint 不稳 | 高 | 固定 solver/seed contract；逐 timestep residual L2 投影到 0.5；分阶段冻结 | endpoint 有限、可复现、有界且延迟可接受 |
| R18 | 参数量提升而非方法机制带来收益 | 高 | matched-capacity dense/history baselines | 主增益在匹配参数量对比中保持 |
| R19 | expert label 未来泄漏、错键或伪语义 | 高 | leading-verb-only、source/mask/fingerprint；确定性 global trajectory key；语义 alignment 按第一版假设接受 | exact key 全覆盖，标签只读允许字段，unknown 不训练 expert |
| R20 | 16-step chunk 内存在 skill 边界 | 高 | 逐位置直接标签/损失；部署按边界风险选择 8/4/边界前停止 | 稳定边界可跨越，高风险边界及时截断 |
| R21 | `null_finish` 被错误实现为可学习 expert，或零 velocity 仍保留随机 residual 初值 | 高 | 无参数 bypass；用 ST motor-gate 将 solver state 从零初始化并逐步归零 | null forward 精确为零且 joint route gradient 非零 |
| R22 | 分别采样多个 full expert chunks 后拼接造成 flow 不一致与高延迟 | 高 | 单 shared residual solve + per-token adapters/heads | 只调用一次 residual solver，endpoint 连续且延迟达标 |
| R23 | temporal router 提前/延迟预测边界，导致不必要短执行或危险跨越 | 高 | entropy/margin/motion-jump/residual 联合风险门控；每轮重规划 | 8/4/high-risk 分布、boundary F1 和跨边界率可审计 |
| R24 | evaluator action queue 继续执行旧 16-step 后缀 | 高 | variable-prefix policy adapter；只入队已提交前缀，耗尽即以最新观测同步重查 | query-id 测试证明 8/4/短前缀后重查且无 stale action |
| R25 | `[1,4,8,16]` 高维 targets 无法给 16 个 route positions 提供逐步状态 | 高 | 额外输出低维 `h_1...h_16`；简单 action+world+position query | `[B,16,128]` 有限且 route 不复制 global logits |
| R26 | 双视角使冻结 OpenVLA 视觉计算与缓存压力增加 | 中高 | paired-frame LRU、一次 ordered multi-image forward、primary-only teacher target | 真实 7B K=8 optimizer step 无 OOM，cache/吞吐可接受 |
| R27 | View Fusion 坍缩到单视角或权重成为无效装饰 | 高 | score-head 零初始化 0.5/0.5、语言/当前视觉条件、逐 skill/位置日志 | 权重有限且和为 1；primary/wrist mask 评估与 success-rate 共同验收 |
| R28 | 单机 DDP 重复数据、状态漂移、rank 写冲突或 NCCL 死锁 | 高 | post-sidecar episode shard、rank-0 I/O/validation、effective-batch migration gate、per-rank RNG；资源保护交由云平台 | 8 卡 shard union/互斥、base-lineage 2→4 resume、参数一致、完整三阶段 torchrun 可恢复 |
| R29 | 八套 TensorFlow/RLDS pipeline 或 MP4 随机解码进入长训练热路径 | 高 | RLDS 仅做离线转换；LeRobot 风格 canonical archive + memory-mapped MoWE feature store + map-style Dataset | 新旧路径等价；训练进程不 import TF、不解码视频、不加载冻结 7B |
| R30 | CALVIN 与 LIBERO 的相机、动作、控制频率和 episode 生命周期不一致，导致 adapter 静默错配或评测泄漏 | 高 | 独立 CALVIN adapter；显式坐标/缩放/旋转/gripper/queue contract；ABC→D 隔离 | round-trip 与 rollout smoke 通过，D 不进入训练统计，官方 evaluator 可复现 |
| R31 | 原始 OpenVLA alias、feature store 与旧 OFT checkpoint identity 静默错配 | 高 | 固定 HF revision/权重 fingerprint；store/checkpoint/eval 全链绑定；禁止 backbone migration | 双视角 base smoke 通过，所有正式产物 identity 一致，旧 lineage 启动即失败 |

## 4. 核心风险与缓解方案

### R1：Nominal future 与最终动作不一致

WAM 的条件是 nominal action `A0=(A0_motion, gripper)`，但机器人候选执行 motion 为：

```text
motion_final[j] = A0_motion[j] + R_motion[j]
gripper_final[j] = binarize(sigmoid(g0_logit[j]))
```

因此第一版必须解释为：WAM 预测 nominal policy 继续执行时的趋势，router 根据这一趋势选择修正 expert。它不预测 expert 修正后的精确未来。

缓解措施：

- Experts 只输出 6D motion residual flow sample，且逐位置 target 是 `A_motion*[j] - stopgrad(A0_motion[j])`；gripper 只由 binary BCE 训练。
- 加入弱 `L_residual = ||ΔA||²`，默认权重 `1e-3`。
- 每轮预测 16 步但默认只提交 8 步；边界风险可疑时提交 4 步，高风险时在边界前停止，随后重新观察和路由。
- 记录 `||motion_final-A0_motion||` 与 `||motion_final-A_motion*||`；若 motion residual 长期大于 nominal motion 本身，说明架构已失去 nominal-consistency。Gripper 另报 BCE/accuracy。
- residual flow 的 solver、seed 与 6D normalized motion space 必须记录；gripper canonical/sign conversion 另记。六路反事实 rollout 只保留为后续扩展。

### R2：Router 标签与部署期选择脱节

纯 imitation learning 只靠 endpoint action loss 时，router 不知道每个 action 位置应分配哪个 expert。新路线用训练期逐位置 `expert_skill_labels` 以 masked CE 监督 router，并以 oracle per-token gates 训练对应 residual flow head；联合阶段再用 ST Gumbel 连接 route/action 梯度。但标签不是部署输入，且高层谓语可能比 MAW future 更容易从任务语言直接猜出。因此论文用语必须是“training-time verb-seeded, future-grounded temporal skill routing”，不能写成显式 world-value optimization 或 oracle subtask access。

必须完成的机制验证：

- `future_shuffle`：把 future latent 在 batch 内打乱。
- `current_only`：使用当前 latent 替代未来 latent。
- `history_only`：保留同参数量 temporal encoder，但删除 future teacher loss。
- `stop_future_gradient`：检查 action loss 是否仅把 WAM 变成另一个 policy head。
- `oracle_route`：用真实训练期逐位置 label 形成 schedule，作为 expert 分工上界。
- `predicted_route`：部署输入下预测 hard schedule，必须单独报告 current skill、future positions 与 boundary 指标。
- `shuffled_skill_label`：打乱 warm-start 标签；若与真标签无差异，不能宣称标签训练有效。
- `instruction_only_router` 暂不作为第一版实现/长训练门槛；待主方法跑通后再决定是否加入论文消融。

### R3：Future predictor 复制当前状态

机器人视频背景大多静止，单纯回归 `z(t+H)` 很容易通过复制 `z(t)` 取巧。

第一版 target 固定为：

```text
z*(t+H)       : 未来 spatial visual tokens
Δz*(t→t+H)    : z*(t+H) - z*(t)
H             : [1, 4, 8, 16]
```

损失必须同时包含 cosine、Smooth L1 和 delta 项。分析必须加入 `copy_current` 基线，并按 motion magnitude 分桶报告误差。Stage 1 晋级使用跨 H=4/8/16 的平均改善而不是要求每个聚合点绝对 Pareto 占优：平均改善至少 10%，且任一 horizon 相对回退不得超过 5%，避免短时域强 copy baseline 的微小波动误阻塞明显改善的中长时域预测。

### R4：Router 绕过 future signal

如果 router 同时得到完整 4096 维当前 VLA feature 和 256 维 future delta，它很可能只使用前者。

默认接口：

```text
router_input = projected_world_belief(128)
             + pooled_future_delta(256)
             + multiscale_memory(128)
            + route_world_tokens h_1...h_16
             + optional_uncertainty
```

每个位置的 query 固定为 `ActionMLP(A0[j]) + WorldProjection(h_(j+1)) + PositionEmbedding(j)`；第一版不加入 delta-h。当前 context 只能经过窄投影进入 router；完整 context 仍可供 nominal head 和 experts 使用。必须记录各输入分支的梯度范数或门控强度。

### R5/R12：Memory 名称大于真实能力

第一版不实现跨任意 episode 的检索记忆。默认多时间尺度 memory：

- 短期：最近 8 个连续视觉/action tokens。
- 长期：instruction token + 从当前 episode 更早前缀中提取的 4 个稀疏 summary tokens。
- 推理：保存当前 episode 已观测序列；每次 query 使用与训练端相同的短期连续和均匀稀疏 prefix indices；episode reset 清空状态。

若 dataset 只返回随机 8 帧窗口，则只能称为 temporal context，不能称为 long-term memory。真正 recurrent memory 需要 episode-ordered segment 和 truncated BPTT，作为后续独立里程碑。

### R6/R17/R19/R20/R21/R22/R23/R24/R25：标签质量、时序状态、expert collapse 与 flow route

默认训练调度：

- Stage 0：按确定性 global trajectory key 完成结构 join audit，再按默认 semantic-aligned 假设读取每 timestep 最后一句 leading verb，报告未知率、六 motor + null coverage 与各 chunk 位置覆盖；不设置额外语义 alignment 或 boundary gate。
- Stage 2：nominal flow 冻结，oracle per-token labels 选择对应 motor heads；按位置更新 residual-flow loss，null 位置只训练 CE 且 residual 强制为零。
- Stage 2 的 frozen nominal-flow/gripper loss 只作为诊断指标记录，权重固定为 0，不得混入用于 validation/早停的 `total_loss`；oracle routing 下 load-balance 权重也为 0。
- Stage 3：从 oracle gates 退火到 ST Gumbel per-token gates，最终用 hard predicted schedule；按 source/position 分开记录 endpoint error。
- 使用 class weighting 或 sampler 平衡处理长尾；load-balance 只能在 predicted/soft 分支作为辅项，不能覆盖真实 label 失衡。
- Top-2 flow-field mixture 不是第一版；每个 timestep 始终只有一个 hard route。
- 记录每个 motor expert 的标签样本数、梯度范数、flow loss、endpoint error，以及逐位置 router confusion、entropy、boundary F1 与切换率。
- action chunk 内每个 timestep 独立监督；跨 Pick→Move 或 Move→Place 边界的 chunk 保留训练，只 mask unknown/padding 位置；不做边界 `±1` mask、soft target 或 debounce。
- `null_finish` 单独记录 route accuracy、末端 padding 比例与 zero assertion；其 residual solver state 必须通过 `motor_gate=sum(gates[:6])` 从零而非 Gaussian noise 初始化并逐步归零。不能使用 detached boolean clamp 切断 ST route gradient；任何非零 null forward state/velocity/residual 都是实现错误。
- Residual trunk 和 solver 每个 chunk 只运行一次；六个轻量 heads 可并行算 velocity 后按 per-token gate 组合，不得采样六个完整 endpoint 再拼接。
- 部署执行长度由风险门控决定：无高风险时默认 8，caution 为 4，high-risk 在首个命中边界前停止；每轮重规划必须丢弃旧 chunk 未执行后缀。
- 新 adapter 必须接受 1～8 步变长 prefix，只把本轮已提交前缀放入 queue，并在队列耗尽后使用最新观测同步重查；禁止旧 queue、temporal ensemble 或 stitching 继续执行未选后缀。

如果任何 motor expert 无样本、其标签被单一 suite 垄断、其 flow loss 无梯度，oracle temporal route 不优于 nominal flow，或 predicted boundary 大量越界，停止长训练并修订 taxonomy/labeler/router；不能用温度或 load-balance 掩盖数据问题。当前六类只是 coarse reusable skills，不得称为逐帧 atomic physical phases。

### R7：真实未来动作与 nominal action 的分布偏移

WAM 初期用真实 action condition 更容易学习，但推理只能使用 `A0`。默认调度：

```text
0%～30% steps   : 100% ground-truth action condition
30%～70% steps  : 逐步增加 nominal condition
70%～100% steps : 80% nominal + 20% ground truth
```

使用 nominal condition 时，按其与示范动作的距离降低 world loss：

```text
q = exp(-beta * distance(A0, A*))
L_world_nominal = q * L_world
```

不要把差异很大的预测动作与示范轨迹未来帧强行配对成因果监督。

### R8：Teacher 目标选择

默认 teacher 为冻结 DINOv2 ViT-S/14，但目标不是单个 pooled vector，而是将原生 patch grid 通过固定 `4×4` spatial average pooling 得到 16 个 384 维 tokens。Target path 不放可学习 projector，避免 teacher target 随训练漂移；student predictor 负责从 WAM hidden 映射到 teacher 维度。Teacher 仅训练时加载。

“独立”指参数、优化器和推理路径独立，不声称它与 OpenVLA 的视觉预训练家族完全无关。论文至少需要 pooled 与 spatial target 的消融；若资源允许，再比较 SigLIP2 或 V-JEPA teacher。

### R9/R10/R31：原始 backbone、多 suite 与 checkpoint 一致性

四个 LIBERO suites 使用相同 7D interface，但只应为前 6 维 relative motion 计算联合 normalization statistics；第 7 维 gripper 保持 canonical binary/absolute 语义，并只在 evaluator 做环境侧 sign conversion。第一版唯一 frozen context backbone 是固定 revision 的原始 `openvla/openvla-7b`。现有 `moojink/*-oft-finetuned-*` 或 `/hy-tmp/openvla-7b-oft-libero-all` 产物只代表历史 pilot，不得进入正式多-suite/CALVIN 主线。

多-suite 默认路线：

- 冻结原始 `openvla/openvla-7b` context encoder；`external/openvla-oft` 只提供 compatible multi-image loader。
- 在四 suite 混合数据上训练 shared nominal action head。
- 第一版不启用 multi-suite LoRA，不加载 OpenVLA/OFT action head。

正式 base snapshot 必须记录 repo id、不可变 Hugging Face revision、processor identity 和权重 fingerprint。所有 feature store、resolved config、MoWE checkpoint 和 evaluator JSONL 必须能追溯到同一 identity。不能把旧 store 的 manifest 字符串改成新路径，也不能使用 `--allow-world-size-change` 绕过 backbone 变化；任何 backbone 切换都必须重建 store 并从 Stage 1 step 0 新建 lineage。

### R11：冻结 backbone 仍可能过慢

训练时不能在每个 step 对同一窗口的 8 帧重复运行完整 OpenVLA language-action 解码器。

默认：

- language embedding 每个 instruction 缓存一次；
- 历史帧只经过视觉分支；相同预处理帧的 pooled feature 由有界 LRU 复用，augmentation 后像素不同则不会错误命中；
- 推理时每个新帧加入当前 episode memory；后续 query 按训练一致的 indices 选择 short/prefix frames，并由同一视觉 LRU 避免重复编码；
- teacher future targets 按 episode/timestep 去重后离线缓存为 float16 shards，训练端按 manifest index 延迟读取；
- cache 必须带 encoder checkpoint、transform、resolution 和 dataset fingerprint，防止静默错配。

### R26/R27：双视角成本与融合有效性

默认 context 固定使用 `[primary, wrist]`：分别预处理、按 upstream OpenVLA-OFT multi-image contract 拼 channel 进入同一次冻结 vision forward，再按 view token 顺序分别 pooling。不得将两张图做空间 mosaic，也不得先把特征相加后再学习权重。

融合器位于冻结 backbone 之外，由 current visual、instruction 和 view identity 共同计算两个 scalar scores；初始严格为 0.5/0.5。训练日志记录 current/history 权重、entropy 和 current-skill 分组。注意 attention weight 不是因果贡献证明：如果最终 success rate 没提升，或 mask 任一视角几乎不改变结果，应优先回退/简化，而不是凭权重图声称机制有效。DINO future target 第一版只看 primary，避免 target/caching 成本翻倍；若 wrist 对精细操作仍明显不足，再把 dual-view teacher 作为后续消融，而不是默认复杂度。

### R28：单机 DDP 正确性与节点内存

单卡训练通过不代表八个独立 Python/TensorFlow pipeline 能安全长期运行。DDP 必须在 sidecar exact join 后且 frame transform 前按 episode shard，保持 `_traj_index` 不变；任意两个 rank 的 episode ID 重叠都视为数据契约失败。feature-store writer 预计算每 episode target-skill histogram，sampler 采用 deterministic suite/window/skill minimax 完整-episode 分配；audit 还必须证明各 rank target-skill counts 合并后与完整 train partition 一致，并报告 window/suite/skill imbalance，避免“episode 不重叠但罕见 expert 几乎只落在一个 rank”的隐性失衡。只有 rank 0 可写日志/checkpoint，并且 rank-0-only validation 必须绕过 DDP wrapper，其他 ranks 在边界等待。

从单卡 checkpoint 迁移时，只有 backbone/store identity 完全相同、显式授权且 effective global batch 不变才允许恢复 optimizer/scheduler；该通用能力不能用于当前旧 OFT→原始 OpenVLA 的切换。当前主线必须直接在 world-size 8 上从 step 0 建立新 lineage。rank 0 保存 checkpoint 时必须先完整写临时文件再原子替换，避免抢占/故障中断损坏唯一的 `checkpoint_latest.pt`；sidecar 不可与 checkpoint generation 错配。按当前平台约束，`start_mtp.py` 不采集 RSS、cgroup、OOM event 或 GPU 显存 telemetry，资源配额与告警由云平台承担。

same-stage resume 必须额外锁定完整 scheduler/optimization contract：`max_steps`、LR groups、AdamW、warmup/min-LR、loss weights、router/action-conditioning schedule、window contract、route mode 和 ablation 均不可改变；只允许修改 `stop_step` 与日志/保存频率。否则即使 optimizer state 能加载，也会在恢复后沿用新的 LambdaLR 或目标权重，形成不可审计的混合训练。

### R29：长期数据读取 backend 与可回收 page cache

当前 RLDS backend 即使限制 `window_shuffle_buffer_size`、frame parallel calls 和 PyTorch workers，仍会在单机 8 卡下创建至少八套 TensorFlow pipeline；若 rank 0 同时保留 validation pipeline，则实际为九套。已有单卡 cgroup OOM 证据表明，单纯增加 RAM 或分段重启不能视为长期解决方案。

正式长训练必须迁移到 `IMPLEMENTATION_PLAN.md` 10.2 节定义的两层数据结构：LeRobot 风格 canonical archive 负责保留原始多模态数据，MoWE feature store 负责训练热路径。由于第一版冻结 OpenVLA 且关闭 image augmentation，paired OpenVLA view features、instruction feature 和 DINO targets 可以离线缓存；训练时从 memory-mapped fixed-shape arrays 构造与现有 window contract 等价的 sample。

截至 2026-07-19，feature-store writer、mmap Dataset、precomputed backbone、episode-aware sampler、sampler resume 与结构/等价性审计均已实现；feature pending episode 有原子索引/checksum，rank 内采用 shard-aware block shuffle，策略与 block size 属于 checkpoint 恢复契约。通用 runtime 的资源诊断工具仍保留给独立运维使用，但一键入口固定 `resource_monitoring=false`，不调用 soak、resource runtime audit 或 resource readiness。R29 因此由 formal store/checksum、100-window 等价性、无 TensorFlow/视频热路径、8-rank assignment 和真实长训练稳定性来约束；平台资源风险不由训练脚本闭环。

关键边界：

- feature 必须从原始 RLDS frame 生成，不得从有损 MP4 反向生成正式 cache。
- canonical 与 feature store 必须在同一确定性 RLDS/sidecar pass 中独立提交；任一输出已完成而另一输出缺失时，只补缺失输出，不重复覆盖已提交 episode。
- primary/wrist 必须按当前 fused OpenVLA 双视角路径成对编码；不得分别编码后假定等价。canonical 视频必须分别保存每路原始 shape，特别是 CALVIN `200×200` static 与 `84×84` gripper，不允许为方便封装而提前统一 resize。
- feature manifest 必须绑定 OpenVLA/DINO checkpoint、processor/transform、view order、RLDS/sidecar fingerprint 和 conversion code version。
- image augmentation、backbone 解冻或输入契约变化后，旧 feature store 自动失效。
- `memory.current` 包含可回收 file page cache；内存门禁必须同时读取 `memory.stat` 的 `anon/file/inactive_file` 和 `memory.events`，以 `working_set ~= current - inactive_file` 与 anon 增长率区分泄漏和正常缓存。
- 512 GiB 是目标可运行节点；940 GiB 只提供额外 page-cache/系统余量，不能作为容忍单调内存增长的理由。

### R13/R14/R30：长程证据与 CALVIN 接入边界

LIBERO 多 suite 适合预训练、基础 benchmark 和消融，但成功 demonstrations 不能直接监督 failure/recovery。论文实验应至少包含：

- 基础 benchmark：LIBERO。
- 第二主 benchmark：标准 CALVIN LH-MTLC，目标设置为 ABC→D；主指标为 average sequence length 和连续完成 1/2/3/4/5 个子任务的成功率。
- 可选扰动：动作噪声、初始位姿变化、抓取后小位移或视觉扰动。

在加入失败/扰动轨迹之前，uncertainty 只能称为预测不确定性，不能命名为 failure probability。

CALVIN 被选定不代表 benchmark 已经跑通。代码已固定官方仓库 commit `fa03f01f19c65920e18cf37398a9ce859274af76`，并实现官方 `task_ABC_D/training` NPZ reader 与 512-shard `calvin_abc` train RLDS reader、ABC-only q01/q99 审计、feature-store converter、benchmark-specific raw/cache equivalence、独立 action/policy adapter、三阶段 DDP8 配置及 `start_mtp_calvin.py`。本地真实 RLDS 全量审计已通过：512/512 shard checksum、17,870 records、1,071,807 frames、785,887 H=16 windows、六 motor 类与 unknown=0；70 个复用的 source episode id 已改用 `(shard, record_index, source_episode_id, timestep)` 消歧。统一 readiness gate 会按 manifest 强制 `calvin_abc_d` equivalence identity，不能复用 LIBERO 报告。原始-backbone formal conversion、100-window equivalence、训练曲线和官方 D simulator 仍未验证，R30 保持开放。正式接入必须继续审计以下契约：

- `rgb_static` / `rgb_gripper` 到 primary/wrist 的预处理和 view order；
- relative/absolute action mode、坐标系、position/rotation scaling、Euler/axis-angle 转换和 gripper sign；
- 当前官方实现由 simulator 在五子任务 sequence 开始时 reset environment，但在每个 subtask 前调用 `model.reset()`，并在每个环境步调用 `step(obs, goal)`；必须用可审计 bridge 在 environment sequence reset 时调用 `reset_sequence()`，把官方 subtask reset 映射为 goal/queue reset，同时使 goal 变化立即丢弃旧 action 后缀。主结果同时报告 per-subtask memory reset 消融，避免把生命周期改动藏成方法增益；
- A/B/C 训练与 D 评测隔离，D 不进入 feature cache、normalization、模型选择或 early stopping；
- language leading-verb taxonomy 在真实 ABC annotations 上的 unknown ratio、六 motor 类覆盖、每类 window 数和 null route 缺失情况；任一 motor 类无覆盖时停止 Stage 2，不得用 class weight 或阈值修改掩盖；
- 所有正式长程数字来自官方 evaluator，并绑定 checkpoint、backbone、seed、repo commit 和完整 resolved config；bridge 同步记录 per-task success、失败位置与 prefix/query 诊断，summary 必须原子写入。

标准 CALVIN 跑通前不扩展 L-CALVIN；BOSS 和 VLABench 不进入第一版主结果表。任何 adapter 单测或单序列 smoke 都不能写成完整 CALVIN benchmark 已通过。

### R15：预训练与测试泄漏

必须区分：

- `in_domain_joint_training`：同一 benchmark train split 上联合训练；expert-skill label 同样仅来自该 split 的允许字段。
- `leave_one_suite_out`：三个 LIBERO suites 预训练，第四个 suite 微调/评估。
- `few_shot_transfer`：目标任务只使用规定数量训练轨迹。

任何 future teacher cache 都只能来自训练 episode；不得预提取或读取 test episode latent。

## 5. 第一版默认关闭的功能

- 像素级视频 decoder 或 diffusion。
- 六 motor expert counterfactual future rollout。
- Top-2 residual-flow vector-field mixture。
- 显式 failure/recovery classifier。
- Simulator predicate/event labels。
- OpenVLA 7B 全量微调。
- 跨 episode 检索数据库。
- 不受约束的自然语言 subtask summary。
- 推理期 DINO/VAE teacher。

## 6. 进入长训练前的风险门槛

只有以下条件全部满足，才允许启动真实长训练：

- Sequence window 不跨 episode，未来 horizon mask 正确。
- 原始 `openvla/openvla-7b` immutable revision、processor 与权重 fingerprint 已签发；primary/wrist ordered multi-image BF16 smoke 通过，且正式链路未加载任何 LIBERO-finetuned OFT 权重。
- OpenVLA context 不含 action target token。
- Nominal motion-flow head 以固定 solver/seed 输出有限的 `[B,16,6]` motion，gripper head 输出 `[B,16,1]` logits，拼接 action 为 `[B,16,7]`。
- WAM 在真实一个 batch 上输出 `[B,16,128] h_1...h_16`、`[B,4,16,D]` future tokens 和 delta tokens。
- RLDS、online memory 与 evaluator 均提供同 timestep 的 primary/wrist；融合权重为 `[B,2]`、有限且和为 1，DINO cache metadata 明确 `teacher_target_views=[primary]`。
- `copy_current` baseline 已实现。
- 六 motor + null 的逐位置 skill labels 已完成来源/未来泄漏、leading-verb 与 position coverage 记录；sidecar timestep alignment 按第一版配置假设接受，不作为额外 gate；每个 motor class 至少有一个 real-batch 有限 motion-expert gradient。
- Oracle per-token、ST Gumbel 和 hard-predicted schedules 均可运行，且 label 不进入 deployment batch。
- `null_finish` 的 solver-step velocity 与 endpoint residual zero assertion 通过。
- 所有 motor residual target、sample 和诊断的逐 timestep 6D L2 范数不超过 `0.5`，clip fraction 被记录；final normalized motion 保持在 `[-1,1]`。
- 简单 query、`[B,16,7]` router shape、单次 shared 6D residual solve、独立 gripper head、8/4/high-risk-boundary 执行与旧后缀丢弃均有测试。
- LIBERO evaluator 的 variable-prefix queue 在 8/4/短前缀后都会重新 query，query id 与执行 action 来源可追踪。
- feature-store checkpoint 在 simulator 中显式重建 online frozen OpenVLA；评测入口在模型加载前强制 CLI backbone 与 checkpoint 内 `backbone_identifier` 一致。full-suite evaluator 只接受 Stage 3 joint 与 LIBERO source contract，并用绑定 checkpoint/suite/seeds 的逐 episode JSONL 恢复，禁止更换形状兼容但语义不同的 backbone，或在中断后混合不同实验记录。
- Teacher cache metadata 与 checkpoint/transform 完全匹配。
- 真实一个 optimizer step loss/gradient 有限。
- 日志包含 nominal/final action distance、world/delta loss、expert usage、router entropy 和 cache fingerprint。
- 2-rank synthetic DDP 更新后参数 checksum 一致；2-rank 小型 Flow-WAM feature-store run 能保存 step N 并由新进程恢复到 N+1，且两 rank RNG/sampler cursor 与 rank-0 单写日志连续。8-rank episode shard 无交集且 union 完整，sidecar `_traj_index`/label 不因 sharding 改变。
- rank 0 是唯一 JSONL/checkpoint writer；原始-backbone Stage 1 从 world-size 8 的 step 0 新建 lineage，不恢复旧 OFT-backbone 单卡 checkpoint。
- 8 卡 step 0→2 smoke 和 2→4 resume 通过，八卡持续工作且 checkpoint step/scheduler/参数一致；随后通过 25-step 与 100-step 压力门禁。
- `mowe_feature_store_v1` 的 episode/frame/window 数、suite partition、sidecar 与 source manifest fingerprints 与 RLDS source 完全一致；converter 保存 TFDS statistics 推导的 expected counts，finalize 与 actual counts 不一致时必须 `formal_training_ready=false`，结构文件存在或未使用 `--limit` 都不能替代完整性证明。
- 至少 100 个真实窗口的新旧路径 OpenVLA views、language、DINO targets、完整 model outputs/losses 在声明 tolerance 内一致。
- 正式长训练进程不 import TensorFlow、不解码 MP4、不加载冻结 OpenVLA 7B；RLDS 只在离线 converter/审计中出现。
- feature-store manifest 必须绑定生成缓存使用的原始 OpenVLA repo id/revision/weight fingerprint、processor identity 和 DINO identifiers；训练配置未显式指定时从 manifest 写入 resolved config，显式指定不一致则启动失败。训练节点无需为了 identifier 校验保存 7B 权重副本。
- episode-aware sampler 的八 rank episode 集合互斥、union 完整，assignment fingerprint 和 sampler cursor 可 checkpoint/resume。
- `start_mtp.py --dry-run` 必须证明完整 Stage 1→2→3 命令链不包含 `/proc`、cgroup、RSS、OOM-event、GPU-memory、soak 或资源 readiness 操作；真实运行的资源保障由平台侧完成。

## 7. 论文用语边界

允许：

- nominal-action-conditioned latent world prediction
- verb-seeded, future-grounded temporal residual flow skill routing
- training-time per-timestep coarse-skill supervision with deployment-time label-free routing
- multi-horizon predictive visual representations
- training-time future visual supervision
- inference-time latent rollout without video decoding

暂不允许：

- exact counterfactual expert planning
- explicit future video imagination
- failure-aware routing（没有失败/扰动实验时）
- universally interpretable semantic experts（没有标签审计、混淆分析与 oracle-route 证据时）
- atomic physical-phase experts（当前标签只支持 coarse subtask/verb skill）
- universal world model（只在 LIBERO 上训练时）

## 8. 风险维护规则

- 每次架构或训练协议发生变化，都要检查本文件是否需要更新。
- 风险只有在真实证据满足“通过门槛”后才能标记为缓解，不得依据代码存在或 mock 检查直接关闭。
- 新风险应记录：触发条件、可能造成的错误结论、最小缓解措施、验收证据。
- `DEV_LOG.md` 记录实际执行情况；本文档不记录虚构或预期结果。
