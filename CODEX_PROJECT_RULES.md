# MoWE Codex/Agent 工作规则

更新时间：2026-07-17（精简活动规则）

本文件是所有新 Codex/agent 会话的项目入口。完整旧规则模板保存在 [`docs/history/CODEX_PROJECT_RULES_FULL_2026-07-17.md`](docs/history/CODEX_PROJECT_RULES_FULL_2026-07-17.md)，默认不读取。

## 1. 文档权威层级

1. `PROJECT_PLAN.md`：研究方向、论文主张、Benchmark 与实验边界。
2. `IMPLEMENTATION_PLAN.md`：当前代码合同、任务顺序、命令和 Definition of Ready。
3. `ARCHITECTURE_RISKS.md`：不可违反的架构边界、停止条件和证据门槛。
4. `DEV_LOG.md`：当前实际状态、真实执行命令、结果、问题和下一步。
5. `docs/history/`：完整历史与追溯资料，不是当前状态来源。

发生冲突时，先检查真实代码/config/log/checkpoint，再按上述职责修正文档。不得用计划文字覆盖实际失败，也不得用历史归档声称当前 tree 已验证。

## 2. 新会话启动协议

### 必须读取

1. 本文件。
2. `PROJECT_PLAN.md` 中与任务相关的章节；最低读取“当前主线”“训练路线”“数据与 Benchmark 策略”“风险与停止条件”。
3. 根 `IMPLEMENTATION_PLAN.md` 全文。它是已经压缩的当前执行合同。
4. `ARCHITECTURE_RISKS.md` 的风险总表、长训练门槛和当前任务对应风险；不要求无关章节全文加载。
5. `DEV_LOG.md` 的“当前状态快照”和最新 1～3 条详细记录。
6. `git status --short`、`rg --files` 和任务相关代码/config/test。

### 禁止默认读取

- 不全文读取 `docs/history/IMPLEMENTATION_PLAN_FULL_2026-07-17.md`。
- 不全文读取 `docs/history/DEV_LOG_2026-07-08_to_2026-07-17.md`。
- 不全文读取旧规则、notebook checkpoint、外部仓库 README 或无关基准资料。

需要历史信息时，先定位再局部读取：

```bash
rg -n '<keyword>' docs/history
sed -n '<start>,<end>p' docs/history/<matched-file>.md
```

### 启动检查

```bash
git status --short
rg --files -g '!docs/history/**' -g '!.ipynb_checkpoints/**' | sort
```

当前工作树可能包含用户或前序 agent 的未提交修改。除非用户明确要求，不得使用 `git reset --hard`、`git checkout --`、批量删除或覆盖这些修改。

## 3. 当前默认方向

- 主线是 `flow_wam_skill_moe`，不是早期 predicate/event 或 regression residual-MoE。
- 先完成 LIBERO formal feature store、等价性、8-rank/8-GPU readiness、Stage 1→2→3 和正式评测，再推进 CALVIN。
- 当前唯一详细执行顺序以 `IMPLEMENTATION_PLAN.md` 第 2 节为准。
- 不因代码、CLI 或 mock 测试存在而声称真实 GPU、数据、simulator 或 benchmark 已通过。

## 4. 编辑前

- 从真实 repo 定位文件、API、config inheritance 和 entrypoint；不根据旧计划猜路径。
- 确认任务属于研究方向、当前执行链或明确用户请求。
- 检查 `ARCHITECTURE_RISKS.md` 中相关 stop condition。
- 不确定的数据、checkpoint、服务器路径、阈值或实验结果必须标记 `TBD` 或明确未验证。
- 只修改任务所需文件，保留无关 dirty changes。

## 5. 编辑中

- 优先最小、可验证、可恢复的实现；不默认启动长训练。
- 不修改 `external/` 或官方上游代码，除非计划和用户明确允许。
- 训练/评测合同必须 config-driven，并把 resolved config、fingerprint、checkpoint lineage 和证据路径写入输出。
- 严格保持 LIBERO/CALVIN、motion/gripper、train/deployment labels、mock/real evidence 的边界。
- 计划发生实质变化时同步 `IMPLEMENTATION_PLAN.md`；研究主张变化时同步 `PROJECT_PLAN.md`；风险或门槛变化时同步 `ARCHITECTURE_RISKS.md`。
- 不把调试流水、整段日志或重复接口粘贴到活动计划；写入日志文件并在 `DEV_LOG.md` 记录摘要与路径。

## 6. 编辑后

1. 运行与风险成比例的最小检查。
2. 检查 `git diff --check`。
3. 更新 `DEV_LOG.md` 顶部快照（仅当当前状态/下一步改变）。
4. 在 `DEV_LOG.md` 末尾追加 `Goal / Changed / Commands Run / Result / Issues / Next`。
5. 如果执行合同或任务状态改变，更新 `IMPLEMENTATION_PLAN.md`；纯实现细节不重复扩写计划。
6. 明确区分：
   - 静态/语法检查通过；
   - mock/synthetic contract 通过；
   - 本地真实数据 preflight 通过；
   - 云端真实 GPU optimizer/恢复通过；
   - simulator/benchmark 正式结果通过。

不得越级表述证据。

## 7. 活动文档和归档维护

- 根活动文档必须短、当前、可直接执行。
- `DEV_LOG.md` 最多保留最近 10 条详细记录；更旧条目移入 `docs/history/DEV_LOG_<period>.md`。
- `IMPLEMENTATION_PLAN.md` 只保留当前合同、活动任务和下一证据。稳定但很长的规格移入 `docs/specs/`；已完成或被替代内容移入 `docs/history/`。
- 归档默认只追加或生成新快照；需要勘误时追加说明，不重写历史实验结果。
- 所有归档都必须能从根活动文档找到；移动内容后检查链接。

## 8. 最小交接提示

新会话可直接使用：

```text
先读取 AGENTS.md 和 CODEX_PROJECT_RULES.md，按其中的最小启动协议接管项目。以根 IMPLEMENTATION_PLAN.md 为当前执行合同，读取 DEV_LOG.md 当前快照和最新记录；不要全文读取 docs/history。先检查 git status 和真实代码/config，再从 IMPLEMENTATION_PLAN.md 第 2 节的第一个未完成门槛继续。不要覆盖未提交修改，不要虚构实验结果。
```
