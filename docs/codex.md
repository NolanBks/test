这个错误已经定位并修复，不是 NCCL、数据或显存问题。

失败原因：

52/53   = nominal_action_head.flow_trunk.token_condition_projection
168/169 = world_model.route_world_head

Stage 1 中这四个参数被设置成可训练，但没有参与 Stage 1 loss。DDP 第一步能够运行，第二次 forward 前
检查上一轮梯度同步时发现它们没有梯度，因此所有 rank 一起退出。

修复策略：

- 永久冻结 nominal head 不使用的 token_condition_projection。
- Stage 1 冻结没有监督路径的 route_world_head。
- Stage 2/3 自动重新启用 route_world_head。
- 没有粗暴开启 find_unused_parameters=True。

修复位于 MoWE/mowe_wam/training/flow_runtime.py。

## 1. 同步修改后的代码

至少需要把新版本的这个文件上传到 8 卡服务器：

MoWE/mowe_wam/training/flow_runtime.py

建议直接同步整个当前 MoWE 代码目录，避免之前 equivalence 修复遗漏。

同步后检查服务器代码：

cd /home/ma-user/work/algorithm/chaoxintao_2/MoWE/MoWE

grep -n "token_condition_projection.parameters" \
mowe_wam/training/flow_runtime.py

grep -n "world_model.route_world_head.parameters" \
mowe_wam/training/flow_runtime.py

两个命令都应该能找到内容。

运行语法检查：

conda activate mowe

python -m compileall -q \
mowe_wam/training/flow_runtime.py \
scripts/pretrain_nominal_flow_wam.py

## 2. 检查失败运行是否产生 checkpoint

STAGE1="/home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs/libero_original_openvla_h16_v1/ddp8/
stage1"

find "$STAGE1" -maxdepth 1 -type f -printf '%f\n' | sort

这次错误发生在第二步 forward 前，而 save_freq=2，正常情况下没有成功 checkpoint。

如果存在 checkpoint，检查 step：

python - "$STAGE1/checkpoint_latest.pt.metadata.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
  metadata = json.loads(path.read_text())
  print("stage:", metadata.get("stage"))
  print("step:", metadata.get("step"))
else:
  print("No checkpoint metadata")
PY

无论是否存在，都建议这次从 Stage 1 step 0 重新开始。因为修复后 optimizer 参数组发生了变化，不应恢复
错误版本生成的 checkpoint。

## 3. 备份失败的 Stage 1 目录

不要直接删除，先移动保存日志：

STAGE1="/home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs/libero_original_openvla_h16_v1/ddp8/
stage1"

if [ -d "$STAGE1" ]; then
mv "$STAGE1" "${STAGE1}_failed_unused_params_20260719"
fi

mkdir -p "$STAGE1"

只移动 Stage 1 目录，不要删除：

reports/
launcher_state.json
configs/
feature equivalence
soak/runtime audit

这些已经通过的前置结果可以继续复用。

## 4. 建议先单独验证 8 卡 step 0→2

使用日志里的命令重新测试：

cd /home/ma-user/work/algorithm/chaoxintao_2/MoWE/MoWE
conda activate mowe

export MOWE_DISABLE_SYSTEM_MONITORING=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
python -m torch.distributed.run \
--standalone \
--nnodes=1 \
--nproc-per-node=8 \
scripts/pretrain_nominal_flow_wam.py \
--config /home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs/libero_original_openvla_h16_v1/
configs/stage1.json \
--feature-store /home/ma-user/work/algorithm/chaoxintao_2/MoWE/mowe_store/libero_h16_formal_4090
\
--checkpoint /home/ma-user/work/algorithm/chaoxintao_2/MoWE/openvla-7b \
--backbone-revision 47a0ec7fc4ec123775a391911046cf33cf9ed83f \
--teacher-checkpoint /home/ma-user/work/algorithm/chaoxintao_2/MoWE/facebook-dinov2-small \
--skill-expert-config /home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs/
libero_original_openvla_h16_v1/reports/skill_experts_h16.json \
--output-dir "$STAGE1" \
--max-steps 50000 \
--stop-step 2 \
--grad-accumulation-steps 1 \
--flow-solver-steps 4 \
--precision bf16 \
--save-freq 2 \
--log-freq 1

我把：

export PYTORCH_CUDA_ALLOC_CONF=

设为空，是为了去除这条无害警告：

expandable_segments not supported on this platform

这不是训练失败原因。

成功标准：

test -f "$STAGE1/checkpoint_latest.pt"
test -f "$STAGE1/checkpoint_latest.pt.metadata.json"

并检查：

python - "$STAGE1/checkpoint_latest.pt.metadata.json" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
print("stage:", metadata["stage"])
print("step:", metadata["step"])

assert metadata["stage"] == "nominal_flow_pretrain"
assert metadata["step"] == 2
print("Stage 1 DDP 0→2 passed")
PY

## 5. 通过后恢复一键脚本

step 2 验证成功后，直接使用原来的一键启动命令，保持：

相同 --run-root-dir
相同 --run-id
相同所有训练参数

并保留：

--disable-system-monitoring

启动器会识别 step 2 checkpoint，然后继续：

Stage 1：2→25→100→1000→50000
Stage 2：0→100→50000
Stage 3：0→100→30000

不需要再重新生成 feature store。

你日志中的第一步训练数据本身是正常的：

- 8 卡 NCCL 已正常工作；
- world_size=8；
- effective batch 为 8；
- loss 和 gradient 都是有限值；
- 每卡峰值显存约 1.18 GiB；
- 所有 rank 报告相同的 unused 参数。

因此当前只需要同步代码、重新跑 Stage 1 0→2。