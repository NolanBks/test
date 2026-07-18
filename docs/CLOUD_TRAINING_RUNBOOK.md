# MoWE-WAM 云服务器单卡与 8 卡训练全流程

更新时间：2026-07-18  
适用主线：`flow_wam_skill_moe`，H=16，原始 `openvla/openvla-7b` frozen backbone，LIBERO feature store

## 0. 先给结论：当前超参数是否“最优”

不能严谨地说已经最优。当前参数是我认为适合第一轮正式训练的强默认值，已经满足模型尺度、全局 batch、一致恢复和数值稳定性的要求，但 H=16 新合同还没有真实 LIBERO 长训练、simulator success rate 或多 seed 证据。应把它们称为 **frozen v1 defaults**，先完成一条可复现主线，再只调少数高价值参数。

当前建议保持不动：

| 项目 | Stage 1 | Stage 2 | Stage 3 | 判断 |
|---|---:|---:|---:|---|
| optimizer steps | 50,000 | 50,000 | 30,000 | 首轮正式预算，不代表最佳停止点 |
| effective global batch | 8 | 8 | 8 | 单卡和 8 卡保持一致 |
| precision | BF16 | BF16 | BF16 | A100/A800/H100/新卡优先 |
| Adam betas | 0.9/0.95 | 0.9/0.95 | 0.9/0.95 | 合理 |
| weight decay | 0.01 | 0.01 | 0.01 | 合理 |
| warmup ratio | 0.04 | 0.04 | 0.05 | 合理 |
| min LR ratio | 0.10 | 0.10 | 0.10 | 合理 |
| gradient clipping | 1.0 | 1.0 | 1.0 | 必须保留 |
| flow Euler steps | 4 | 4 | 4 | 训练/评测保持一致 |
| nominal/world LR | `1e-4` | `1e-5` | `1e-5` | Stage 2/3 保护已有表示 |
| router/expert LR | 不训练 | `1e-4` | `1e-4` | 合理的 10× 分组 LR |
| residual L2 cap | 0.5 | 0.5 | 0.5 | 安全合同，不作为普通调参项 |

首轮只允许在真实 pilot 后考虑以下三类调参：

1. Stage 1 的 `1e-4` 是否因验证损失震荡需要降到 `5e-5`。
2. Stage 2 是否在 20k 左右已经收敛，无需跑满 50k。
3. Stage 3 router/expert LR 是否由 `1e-4` 降到 `5e-5`，以及风险门控阈值是否需要用 rollout 校准。

不要一开始同时改网络宽度、loss weights、batch、LR、route schedule 和执行阈值；否则无法判断问题来自哪里。

## 1. 两条执行路径的边界

### 1.1 单卡路径

用途：环境、原始 OpenVLA、真实 RLDS、feature store、Stage 1/2/3 forward/backward、checkpoint、resume 和指标完整性调试。

- 每阶段最多累计 100 个未认证 optimizer steps。
- `batch_size=1`，`grad_accumulation_steps=8`，effective batch=8。
- 不作为正式长训练或论文结果。
- 不允许通过多个 100-step 输出目录规避 readiness。

### 1.2 8 卡路径

用途：正式 lineage、长期训练和论文 checkpoint。

- 单机 8×A100/A800/H100，NCCL。
- 每卡 batch=1，accumulation=1，effective batch=8。
- 必须通过 formal store、100-window equivalence、8-rank soak、8-GPU runtime 和 checkpoint-bound readiness。
- Stage 1/2/3 各自拥有独立 output directory 和 readiness report。

## 2. 推荐服务器规格

单卡调试：

- Ubuntu 22.04。
- Python 3.10。
- NVIDIA GPU 至少 24 GiB；原始 7B 双视角 preflight 推荐 48 GiB。
- RAM 至少 128 GiB，推荐 256 GiB。
- 可用磁盘至少 150 GiB，推荐 250 GiB。

8 卡正式训练：

- 8×A100/A800 80 GiB 或同等级 GPU。
- RAM 推荐 512 GiB；必须同时检查容器 cgroup 限额，而不是只看宿主机内存。
- 本地 NVMe，避免 feature shards 位于高延迟网络盘。
- cgroup v2、NCCL、每卡 CUDA 指标必须可读取。

## 3. 一次性路径变量

以下路径只需要按服务器修改一次。后续命令显式引用数据、模型、store、checkpoint 和输出目录。

```bash
export MOWE_ROOT=/hy-tmp/MoWE
export MOWE_DATA_ROOT=/hy-tmp/libero_cot_rlds
export MOWE_SIDECAR_ROOT=/hy-tmp/libero_cot_rlds
export MOWE_SIDECAR_JSON=/hy-tmp/libero_cot_rlds/cot_file.json
export MOWE_MODEL_ROOT=/hy-tmp/models
export MOWE_OPENVLA_SNAPSHOT=/hy-tmp/openvla-7b
export MOWE_DINO_SNAPSHOT=/hy-tmp/facebook-dinov2-small
export MOWE_STORE_SMOKE=/hy-tmp/mowe_store/libero_h16_smoke
export MOWE_STORE_FORMAL=/hy-tmp/mowe_store/libero_h16_formal
export MOWE_LIBERO_ROOT=/hy-tmp/LIBERO
export MOWE_REPORT_ROOT=/hy-tmp/MoWE/outputs/cloud_reports
export MOWE_SINGLE_ROOT=/hy-tmp/MoWE/outputs/cloud_single_gpu
export MOWE_DDP_ROOT=/hy-tmp/MoWE/outputs/cloud_ddp8
export MOWE_HF_CACHE=/hy-tmp/huggingface

export HF_HOME="$MOWE_HF_CACHE"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$MOWE_ROOT:$MOWE_ROOT/external/openvla-oft:$MOWE_LIBERO_ROOT:${PYTHONPATH:-}"

mkdir -p "$MOWE_MODEL_ROOT" "$MOWE_REPORT_ROOT" "$MOWE_SINGLE_ROOT" "$MOWE_DDP_ROOT"
cd "$MOWE_ROOT"
```

### 3.1 当前单卡服务器执行记录（2026-07-18）

本机为单张 NVIDIA A100 80 GiB，`torch=2.2.1+cu121`，CUDA/BF16 可用。当前已核对的实际路径就是上一节变量值：`libero_cot_rlds` 同时包含四个 LIBERO TFDS RLDS suite 与 `cot_file.json`，不可变原始 OpenVLA snapshot 的 revision 为 `47a0ec7fc4ec123775a391911046cf33cf9ed83f`，DINO snapshot 为 `/hy-tmp/facebook-dinov2-small`。

为使用原始 checkpoint 的 OFT-compatible ordered multi-image loader，额外安装并固定了本地 checkout：

```bash
python -m pip install -e /hy-tmp/transformers-openvla-oft --no-deps
python -m pip install -e /hy-tmp/dlimp_openvla
```

实际 fork commits：`transformers-openvla-oft=bc339d9ad707454c0c115970db43c260067c61ab`，`dlimp_openvla=040105d256bd28866cc6620621a3d5f7b6b91b46`。同时需要 `tensorflow==2.15.0`、`tensorflow-datasets==4.9.3`、`tensorflow-metadata==1.14.0`、`protobuf==3.20.3`、`tensorflow-graphics==2021.12.3`、`pyarrow>=16` 与 `wandb==0.16.6`。

已完成的真实证据（均非 benchmark 成功率）：

- H=16 RLDS 审计通过：1,693 episodes、273,465 transitions、246,377 valid windows，四 suite 的 episode/annotation exact match 均为 1.0，gripper non-binary=0。报告：`outputs/cloud_reports/libero_rlds_h16_audit.json`。
- 原始 7B 双视角 BF16 real-batch backward preflight 通过，`total_loss=4.308042526245117`，所有 risk gates 通过。报告：`outputs/cloud_reports/openvla_real_preflight.{stdout,stderr}.log`。
- 两 episode feature-store smoke 通过：2 episodes、253 frames、221 windows，checksum 全通过；样本 actions 为 `[16,7]`、future targets 为 `[4,16,384]`、OpenVLA views 为 `[2,4096]`。这是故意 non-formal 的 partial store，审计 `valid=false`/`formal_training_ready=false` 是正确结果，绝不能训练。报告：`outputs/cloud_reports/store_smoke_audit.json`。

为使原始 snapshot 使用本地 OFT-compatible 多图实现，adapter 必须保持 `trust_remote_code=false`；否则 snapshot 的 `auto_map` 会覆盖本地注册并加载缺少 ordered multi-image API 的单图实现。该行为已有回归测试覆盖。

### 3.2 A100 到 4090 的 formal conversion 交接（2026-07-18）

`/hy-tmp/mowe_store/libero_h16_formal` 已在 A100 上开始 partial conversion，用户报告进度至少为 `feature_episodes=100`。该目录不是完整 formal store，不可用于训练、equivalence 或 readiness；更不能在另一张 GPU 上继续写入。虽然不同 GPU 的 BF16 输出通常接近，但第一份正式缓存不能混合 A100/4090 生成的特征。

在 4090 48 GiB 上必须从一个新目录重新开始。真实双视角 conversion smoke 在 A100 上约使用 15.3 GiB，先保留 `--encode-batch-size 8`；若 OOM，依次降至 4、2。不要添加 `--limit-episodes`，转换可被中断并以完全相同命令恢复。

```bash
export MOWE_STORE_FORMAL=/hy-tmp/mowe_store/libero_h16_formal_4090
mkdir -p "$MOWE_STORE_FORMAL" "$MOWE_REPORT_ROOT"

CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false \
PYTHONPATH="$MOWE_ROOT:$MOWE_ROOT/external/openvla-oft:${PYTHONPATH:-}" \
python scripts/convert_rlds_to_mowe_store.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root "$MOWE_DATA_ROOT" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-sidecar "$MOWE_SIDECAR_JSON" \
  --output "$MOWE_STORE_FORMAL" \
  --encode-batch-size 8 \
  --episodes-per-shard 96 \
  --device cuda:0 \
  --precision bf16 \
  | tee "$MOWE_REPORT_ROOT/store_formal_4090_conversion.log"
```

完成条件为 `feature_episodes=1693`、`formal_training_ready=true`、`episode_count=1693`、`frame_count=273465`、`window_count=246377`、`counts_match=true`。`canonical_episodes=0` 是当前命令的正常结果，因为未请求 `--canonical-output`。

不要把旧的 LIBERO-finetuned OpenVLA-OFT checkpoint 填入 `MOWE_OPENVLA_SNAPSHOT`。主线只接受原始 `openvla/openvla-7b`。

## 4. 环境搭建

### 4.1 系统依赖

以下命令需要 root 或 sudo：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build git git-lfs ffmpeg jq \
  libegl1 libgl1 libglib2.0-0 libosmesa6-dev patchelf
git lfs install
```

无 sudo 时不要跳过 EGL/FFmpeg 检查；应让云平台安装对应系统包。

### 4.2 Conda 与 Python

```bash
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda create -n mowe python=3.10 -y
conda activate mowe
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

如果服务器的 Conda 不在 `/usr/local/miniconda3`，用 `conda info --base` 找到其 `etc/profile.d/conda.sh`。

### 4.3 PyTorch 与 OpenVLA-OFT 依赖

当前上游合同是 PyTorch 2.2.0、torchvision 0.17.0、torchaudio 2.2.0 和自定义 transformers fork：

```bash
python -m pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -e "$MOWE_ROOT/external/openvla-oft"
python -m pip install -r "$MOWE_ROOT/requirements-mowe.txt"
python -m pip install "flash-attn==2.5.5" --no-build-isolation
```

安装 LIBERO 与历史上已验证的 simulator 版本：

```bash
if [ ! -d "$MOWE_LIBERO_ROOT/.git" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$MOWE_LIBERO_ROOT"
fi
python -m pip install -e "$MOWE_LIBERO_ROOT"
python -m pip install -r "$MOWE_ROOT/external/openvla-oft/experiments/robot/libero/libero_requirements.txt"
python -m pip install \
  numpy==1.26.4 mujoco==2.3.7 opencv-python==4.10.0.84 \
  protobuf==3.20.3 wandb==0.16.6
python -m pip check
```

### 4.4 环境硬检查

```bash
nvidia-smi
python - <<'PY'
import json
import torch
import transformers
import tensorflow as tf
import tensorflow_datasets as tfds

print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "transformers": transformers.__version__,
    "tensorflow": tf.__version__,
    "tfds": tfds.__version__,
}, indent=2))
PY
```

达标条件：

- `cuda_available=true`。
- GPU 数与租用规格一致。
- `bf16_supported=true`。
- `pip check` 无依赖冲突。
- 8 卡机器还必须满足 `torch.cuda.device_count()==8`。

保存环境证据：

```bash
mkdir -p "$MOWE_REPORT_ROOT/environment"
python -m pip freeze > "$MOWE_REPORT_ROOT/environment/pip_freeze.txt"
nvidia-smi -q > "$MOWE_REPORT_ROOT/environment/nvidia_smi_q.txt"
uname -a > "$MOWE_REPORT_ROOT/environment/uname.txt"
```

## 5. 下载并冻结模型与数据

### 5.1 获取不可变 OpenVLA revision

```bash
export MOWE_OPENVLA_REVISION=$(python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("openvla/openvla-7b").sha)
PY
)

test "${#MOWE_OPENVLA_REVISION}" -eq 40
printf '%s\n' "$MOWE_OPENVLA_REVISION" | tee "$MOWE_REPORT_ROOT/environment/openvla_revision.txt"
```

下载固定 revision：

```bash
export MOWE_OPENVLA_SNAPSHOT MOWE_OPENVLA_REVISION
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openvla/openvla-7b",
    revision=os.environ["MOWE_OPENVLA_REVISION"],
    local_dir=os.environ["MOWE_OPENVLA_SNAPSHOT"],
)
PY
```

下载固定 DINO teacher：

```bash
export MOWE_DINO_REVISION=$(python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("facebook/dinov2-small").sha)
PY
)
export MOWE_DINO_SNAPSHOT MOWE_DINO_REVISION
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/dinov2-small",
    revision=os.environ["MOWE_DINO_REVISION"],
    local_dir=os.environ["MOWE_DINO_SNAPSHOT"],
)
PY
```

验证原始 OpenVLA identity：

```bash
python - <<'PY' | tee "$MOWE_REPORT_ROOT/environment/openvla_identity.json"
import json
import os
from mowe_wam.backbones import resolve_original_openvla_identity

identity = resolve_original_openvla_identity(
    os.environ["MOWE_OPENVLA_SNAPSHOT"],
    revision=os.environ["MOWE_OPENVLA_REVISION"],
    repo_id="openvla/openvla-7b",
)
print(json.dumps(identity, indent=2, sort_keys=True))
PY
```

达标条件：命令退出码为 0，repo 为 `openvla/openvla-7b`，revision 为保存的 40 位 commit，并生成 `identity_sha256`。

### 5.2 LIBERO RLDS 与 skill sidecar

目录不存在时执行：

```bash
if [ ! -d "$MOWE_DATA_ROOT/.git" ]; then
  git clone https://huggingface.co/datasets/openvla/modified_libero_rlds "$MOWE_DATA_ROOT"
fi
git -C "$MOWE_DATA_ROOT" lfs pull

if [ ! -d "$MOWE_SIDECAR_ROOT/.git" ]; then
  git clone https://huggingface.co/datasets/yinchenghust/libero_cot_rlds "$MOWE_SIDECAR_ROOT"
fi
git -C "$MOWE_SIDECAR_ROOT" lfs pull

test -f "$MOWE_SIDECAR_JSON"
```

必须看到四个数据集目录：

```bash
for name in \
  libero_spatial_no_noops \
  libero_object_no_noops \
  libero_goal_no_noops \
  libero_10_no_noops; do
  test -d "$MOWE_DATA_ROOT/$name"
done
```

## 6. 代码与真实数据前序检查

### 6.1 本地合同测试

```bash
cd "$MOWE_ROOT"
python -m compileall -q mowe_wam scripts tests
python -m unittest discover -s tests
python scripts/check_flow_wam_forward.py --synthetic --batch-size 2
python scripts/eval_libero_temporal_skill.py --queue-smoke
```

达标条件：

- 当前应为 63 项或更多测试全部通过。
- synthetic 输出 action/router positions 为 16，future horizons 为 4。
- queue smoke 的 query observation 为 0、8、12，证明 8/4/2 前缀耗尽后重查。

### 6.2 H=16 RLDS/sidecar 全量审计

```bash
python scripts/audit_flow_wam_rlds.py \
  --data-root "$MOWE_DATA_ROOT" \
  --skill-sidecar "$MOWE_SIDECAR_JSON" \
  --max-horizon 16 \
  --output "$MOWE_REPORT_ROOT/libero_rlds_h16_audit.json" \
  | tee "$MOWE_REPORT_ROOT/libero_rlds_h16_audit.stdout.log"
```

达标条件：

- 每个 suite 的 `trajectory_ids_contiguous=true`。
- `exact_episode_key_match_ratio=1.0`。
- 总 `annotation_step_match_ratio=1.0`。
- gripper `non_binary=0`。
- `valid_windows>0` 且 max horizon 为 16。
- `alignment_verified=false` 必须保留；它表示语义时序对齐仍是显式假设。

### 6.3 生成与服务器数据指纹匹配的 skill config

```bash
export MOWE_SKILL_CONFIG="$MOWE_REPORT_ROOT/skill_experts_h16.json"
export MOWE_ROOT MOWE_SIDECAR_JSON MOWE_SKILL_CONFIG MOWE_REPORT_ROOT
python - <<'PY'
import json
import math
import os
from collections import Counter
from pathlib import Path

root = Path(os.environ["MOWE_ROOT"])
report_path = Path(os.environ["MOWE_REPORT_ROOT"]) / "libero_rlds_h16_audit.json"
report = json.loads(report_path.read_text())
cfg = json.loads((root / "configs/mowe_wam/skill_experts.yaml").read_text())

counts = Counter()
for suite in report["suites"]:
    counts.update(suite["parsed_skill_counts"])

skill_names = [
    "pick_grasp", "place_release", "move_transport", "open_close",
    "turn_rotate", "push_pull", "null_finish",
]
inverse = [1.0 / math.sqrt(max(int(counts[name]), 1)) for name in skill_names]
scale = len(inverse) / sum(inverse)

cfg["source_path"] = os.environ["MOWE_SIDECAR_JSON"]
cfg["audit"].update({
    "report": str(report_path),
    "dataset_manifest_fingerprint_sha256": report["dataset_manifest_fingerprint_sha256"],
    "sidecar_fingerprint_sha256": report["sidecar_fingerprint_sha256"],
    "episodes": report["totals"]["episodes"],
    "transitions": report["totals"]["transitions"],
    "valid_windows_h16": report["totals"]["valid_windows"],
    "exact_episode_key_matches": report["totals"]["exact_episode_key_matches"],
    "annotation_step_match_ratio": report["totals"]["annotation_step_match_ratio"],
    "alignment_verified": False,
    "label_counts": {name: int(counts[name]) for name in [*skill_names, "unknown"]},
})
cfg["class_weights_inverse_sqrt"] = [round(value * scale, 6) for value in inverse]
Path(os.environ["MOWE_SKILL_CONFIG"]).write_text(
    json.dumps(cfg, indent=2, sort_keys=True) + "\n"
)
PY

python scripts/inspect_skill_experts.py \
  --data-root "$MOWE_DATA_ROOT" \
  --skill-config "$MOWE_SKILL_CONFIG" \
  --sidecar "$MOWE_SIDECAR_JSON" \
  | tee "$MOWE_REPORT_ROOT/skill_experts_h16_inspect.log"
```

达标条件：七个 route 均有正样本，unknown 保持 `-1`，七个 class weights 均为正。

### 6.4 原始 7B 双视角真实 backward preflight

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_flow_wam_training.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root "$MOWE_DATA_ROOT" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-sidecar "$MOWE_SIDECAR_JSON" \
  --precision bf16 \
  --backward \
  | tee "$MOWE_REPORT_ROOT/openvla_real_preflight.log"
```

达标条件：输出 `status=preflight_passed`、`mode=real_batch`、`backward=true`，所有 risk gates 通过，total loss 有限，无 CUDA OOM。

## 7. 构建 H=16 feature store

转换必须单进程运行。即使在 8 卡服务器上也只用 GPU 0；不要让 8 个进程同时写同一个 store。

### 7.1 两 episode smoke store

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/convert_rlds_to_mowe_store.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root "$MOWE_DATA_ROOT" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-sidecar "$MOWE_SIDECAR_JSON" \
  --output "$MOWE_STORE_SMOKE" \
  --encode-batch-size 8 \
  --episodes-per-shard 2 \
  --limit-episodes 2 \
  --device cuda:0 \
  --precision bf16 \
  | tee "$MOWE_REPORT_ROOT/store_smoke_conversion.log"

python scripts/audit_mowe_feature_store.py \
  --store "$MOWE_STORE_SMOKE" \
  --world-size 1 \
  --verify-all-checksums \
  --sample-windows 4 \
  --output "$MOWE_REPORT_ROOT/store_smoke_audit.json"
```

达标条件：audit `valid=true`，sample shape 中 actions 为 `[16,7]`、future targets 第一维为 4。Smoke store 的 `formal_training_ready=false` 是正确行为，绝不能用于训练。

### 7.2 全量 formal store

确认 smoke 通过后运行；命令中没有 `--limit-episodes`：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/convert_rlds_to_mowe_store.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --data-root "$MOWE_DATA_ROOT" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-sidecar "$MOWE_SIDECAR_JSON" \
  --output "$MOWE_STORE_FORMAL" \
  --encode-batch-size 8 \
  --episodes-per-shard 96 \
  --device cuda:0 \
  --precision bf16 \
  | tee "$MOWE_REPORT_ROOT/store_formal_conversion.log"
```

该转换可恢复；中断后用完全相同命令继续，不要删除已提交 shards。

达标条件：

- 输出 `formal_training_ready=true`。
- `completion_contract.counts_match=true`。
- actual episode/frame/window counts 与 converter 根据源数据计算的 expected counts 一致。
- 所有 shards 完成提交，无 pending/failed episode。

### 7.3 全 checksum 与 8-rank assignment 审计

```bash
python scripts/audit_mowe_feature_store.py \
  --store "$MOWE_STORE_FORMAL" \
  --world-size 8 \
  --seed 7 \
  --shuffle-block-size 256 \
  --verify-all-checksums \
  --sample-windows 32 \
  --max-window-imbalance-ratio 1.25 \
  --max-suite-imbalance-ratio 1.50 \
  --max-skill-imbalance-ratio 2.00 \
  --output "$MOWE_REPORT_ROOT/feature_store_audit.json" \
  | tee "$MOWE_REPORT_ROOT/feature_store_audit.stdout.log"
```

达标条件：report `valid=true`；episode union 完整、无 overlap、skill union 完整、八个 rank 均有 episode/window，三项 imbalance checks 均为 true。

如果某个 imbalance 限制失败，先检查罕见 skill 和 suite 分配；不要直接把阈值改成一个必过的大数。

### 7.4 100-window raw/cache 等价性

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/audit_feature_store_equivalence.py \
  --config configs/mowe_wam/train_nominal_flow_wam.yaml \
  --store "$MOWE_STORE_FORMAL" \
  --data-root "$MOWE_DATA_ROOT" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-sidecar "$MOWE_SIDECAR_JSON" \
  --samples 100 \
  --seed 1701 \
  --stage nominal_flow_pretrain \
  --feature-atol 0.03 \
  --output-atol 0.10 \
  --loss-atol 0.05 \
  --output "$MOWE_REPORT_ROOT/feature_equivalence_100.json" \
  | tee "$MOWE_REPORT_ROOT/feature_equivalence_100.stdout.log"
```

达标条件：`passed=true`、`compared_samples=100`、`missing_pairs=[]`、`masks_match=true`。
OpenVLA/language 在有效 history 位置使用 mean-absolute/cosine 门槛，DINO 使用与训练一致的
Smooth-L1/cosine 门槛；padding 只检查 raw/cache mask 一致性，不比较被 mask 的特征值。
`max_feature_abs_error` 仅用于诊断 BF16/FP16 极值，正式 feature 门槛看
`max_feature_gate_error <= tolerances.feature_atol`。输出门槛看连续模型输出和 gripper logits；
拼接后的 binary `actions` 与 `gripper_accuracy` 只作为离散诊断，不因临界 logit 的单步翻转否决缓存。

## 8. 单卡：三阶段完整工程调试

单卡 lane 只跑到每阶段 100 steps。所有命令显式传入 store、原始 backbone、revision、teacher、skill config、output 和 checkpoint。

### 8.1 Stage 1：0→25

```bash
export MOWE_S1_SINGLE="$MOWE_SINGLE_ROOT/stage1"

CUDA_VISIBLE_DEVICES=0 python scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/single_gpu_nominal_flow_wam_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S1_SINGLE" \
  --max-steps 50000 \
  --stop-step 25 \
  --grad-accumulation-steps 8 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 25 \
  --log-freq 5 \
  | tee "$MOWE_REPORT_ROOT/single_stage1_0_25.log"
```

### 8.2 Stage 1：25→100 resume

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/single_gpu_nominal_flow_wam_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S1_SINGLE" \
  --max-steps 50000 \
  --stop-step 100 \
  --grad-accumulation-steps 8 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 25 \
  --log-freq 5 \
  --resume "$MOWE_S1_SINGLE/checkpoint_latest.pt" \
  | tee "$MOWE_REPORT_ROOT/single_stage1_25_100.log"
```

Stage 1 硬门槛：

- checkpoint metadata 的 stage 为 `nominal_flow_pretrain`，step=100。
- train/validation JSONL 均存在，loss/gradient/LR 全部有限。
- `null_motion_zero_violation_count=0`。
- nominal flow、gripper、world model、memory 和 view fusion 的 gradient norm 非零。
- 无 OOM，GPU peak 未超过配置门限。

Stage 1 质量观察：验证集各 horizon 的 predicted `smooth_l1` 应开始下降；100 steps 只判断代码和趋势，不要求已经优于 copy-current。

### 8.3 Stage 2：从 Stage 1 初始化，0→25→100

```bash
export MOWE_S2_SINGLE="$MOWE_SINGLE_ROOT/stage2"

CUDA_VISIBLE_DEVICES=0 python scripts/warmstart_skill_flow_experts.py \
  --config configs/mowe_wam/single_gpu_warmstart_skill_flow_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S2_SINGLE" \
  --max-steps 50000 \
  --stop-step 25 \
  --grad-accumulation-steps 8 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --route-mode oracle \
  --save-freq 25 \
  --log-freq 5 \
  --init-wam "$MOWE_S1_SINGLE/checkpoint_latest.pt" \
  | tee "$MOWE_REPORT_ROOT/single_stage2_0_25.log"

CUDA_VISIBLE_DEVICES=0 python scripts/warmstart_skill_flow_experts.py \
  --config configs/mowe_wam/single_gpu_warmstart_skill_flow_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S2_SINGLE" \
  --max-steps 50000 \
  --stop-step 100 \
  --grad-accumulation-steps 8 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --route-mode oracle \
  --save-freq 25 \
  --log-freq 5 \
  --resume "$MOWE_S2_SINGLE/checkpoint_latest.pt" \
  | tee "$MOWE_REPORT_ROOT/single_stage2_25_100.log"
```

Stage 2 硬门槛：

- stage=`expert_warmstart`，step=100。
- route source 为 oracle。
- 六个 motor expert 在有覆盖的日志窗口内都有有限、非零 gradient norm。
- `null_motion_zero_violation_count=0`。
- `motion_residual_norm_max<=0.5`。
- route label coverage 和七路 confusion matrix 可见，无类别被错误映射到 expert 0。

### 8.4 Stage 3：从 Stage 2 初始化，0→25→100

```bash
export MOWE_S3_SINGLE="$MOWE_SINGLE_ROOT/stage3"

CUDA_VISIBLE_DEVICES=0 python scripts/train_flow_wam_skill_moe.py \
  --config configs/mowe_wam/single_gpu_train_flow_wam_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S3_SINGLE" \
  --max-steps 30000 \
  --stop-step 25 \
  --grad-accumulation-steps 8 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 25 \
  --log-freq 5 \
  --init-wam "$MOWE_S2_SINGLE/checkpoint_latest.pt" \
  | tee "$MOWE_REPORT_ROOT/single_stage3_0_25.log"

CUDA_VISIBLE_DEVICES=0 python scripts/train_flow_wam_skill_moe.py \
  --config configs/mowe_wam/single_gpu_train_flow_wam_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S3_SINGLE" \
  --max-steps 30000 \
  --stop-step 100 \
  --grad-accumulation-steps 8 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 25 \
  --log-freq 5 \
  --resume "$MOWE_S3_SINGLE/checkpoint_latest.pt" \
  | tee "$MOWE_REPORT_ROOT/single_stage3_25_100.log"
```

Stage 3 硬门槛：

- stage=`joint`，step=100。
- 前 100 steps 处于设计中的 oracle 区间；router gradient 必须非零。Scheduled ST-Gumbel 在总进度 20% 后开始出现，不能要求 step 100 已经出现。
- action、loss、route logits、world tokens 全部有限。
- null residual 精确为零，residual 最大范数不超过 0.5。
- execution histogram/reason histogram 可解析，prefix 始终在 1～8。

单卡三阶段全部通过后，重新从 8 卡 Stage 1 建立正式 lineage；不要把单卡 Stage 3 当作论文模型。

## 9. 8 卡正式训练前门禁

以下命令必须在将要训练的同一台机器、同一次 boot、同一个 cgroup 中运行。

### 9.1 cgroup v2 检查

```bash
test -f /sys/fs/cgroup/cgroup.controllers
test -f /sys/fs/cgroup/memory.current
test -f /sys/fs/cgroup/memory.max
test -f /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
```

如果文件不存在，当前正式 8 卡代码会 fail closed；不要关闭 `require_cgroup_metrics` 绕过。

### 9.2 8-rank feature-store soak

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/soak_mowe_feature_store.py \
  --store "$MOWE_STORE_FORMAL" \
  --steps 10000 \
  --warmup-steps 1000 \
  --sample-every 250 \
  --max-anon-growth-mib 512 \
  --max-working-set-growth-mib 2048 \
  --max-anon-slope-mib-per-1k-steps 64 \
  --max-working-set-slope-mib-per-1k-steps 256 \
  --max-open-feature-shards 2 \
  --shuffle-block-size 256 \
  --output "$MOWE_REPORT_ROOT/feature_store_soak_8rank.json" \
  | tee "$MOWE_REPORT_ROOT/feature_store_soak_8rank.stdout.log"
```

达标条件：顶层和每个 rank 的 `passed=true`，八个 rank 完整，无 TensorFlow import，无 OOM/OOM-kill，增长量和斜率均低于阈值。

### 9.3 8-GPU runtime audit

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/audit_ddp_runtime.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --memory-guard-fraction 0.80 \
  --gpu-memory-guard-fraction 0.85 \
  --output "$MOWE_REPORT_ROOT/ddp_runtime_8gpu.json" \
  | tee "$MOWE_REPORT_ROOT/ddp_runtime_8gpu.stdout.log"
```

达标条件：world size=8、effective global batch=8、local ranks 唯一、八张 GPU 均绑定，资源阈值为 0.80/0.85。

## 10. 8 卡 Stage 1：正式 lineage

正式配置的 `max_steps` 从第一步起固定为 50,000；只用 `--stop-step` 做阶梯，不能先用 `max_steps=1000` 再改成 50,000。

```bash
export MOWE_S1_DDP="$MOWE_DDP_ROOT/stage1"
```

### 10.1 0→2

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S1_DDP" \
  --max-steps 50000 \
  --stop-step 2 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 2 \
  --log-freq 1 \
  | tee "$MOWE_REPORT_ROOT/ddp_stage1_0_2.log"
```

达标：两个 optimizer steps、八 rank 参数同步、checkpoint step=2、无 episode overlap/OOM。

### 10.2 2→25→100

每次恢复都保持 `max_steps=50000`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S1_DDP" \
  --max-steps 50000 \
  --stop-step 25 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 25 \
  --log-freq 5 \
  --resume "$MOWE_S1_DDP/checkpoint_latest.pt" \
  | tee "$MOWE_REPORT_ROOT/ddp_stage1_2_25.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S1_DDP" \
  --max-steps 50000 \
  --stop-step 100 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 25 \
  --log-freq 5 \
  --resume "$MOWE_S1_DDP/checkpoint_latest.pt" \
  | tee "$MOWE_REPORT_ROOT/ddp_stage1_25_100.log"
```

### 10.3 为 step-100 checkpoint 签发 readiness

```bash
python scripts/audit_long_training_readiness.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --store "$MOWE_STORE_FORMAL" \
  --feature-audit "$MOWE_REPORT_ROOT/feature_store_audit.json" \
  --equivalence-report "$MOWE_REPORT_ROOT/feature_equivalence_100.json" \
  --soak-report "$MOWE_REPORT_ROOT/feature_store_soak_8rank.json" \
  --ddp-runtime-audit "$MOWE_REPORT_ROOT/ddp_runtime_8gpu.json" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --checkpoint "$MOWE_S1_DDP/checkpoint_latest.pt" \
  --checkpoint-mode resume \
  --world-size 8 \
  --min-equivalence-samples 100 \
  --min-soak-steps 10000 \
  --output "$MOWE_REPORT_ROOT/readiness_stage1_step100.json"
```

达标条件：`passed=true` 且所有 `checks` 为 true，errors 为空。

### 10.4 100→1000 pilot

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S1_DDP" \
  --max-steps 50000 \
  --stop-step 1000 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 100 \
  --log-freq 10 \
  --resume "$MOWE_S1_DDP/checkpoint_latest.pt" \
  --long-run-readiness-report "$MOWE_REPORT_ROOT/readiness_stage1_step100.json" \
  | tee "$MOWE_REPORT_ROOT/ddp_stage1_100_1000.log"
```

如果在 1000 主动停止，旧 step-100 readiness 不能用于从 step 1000 再恢复；必须用当前 checkpoint 重新运行 10.3，输出 `readiness_stage1_step1000.json`。

### 10.5 Stage 1 质量门槛与 1000→50000

运行日志汇总：

```bash
python scripts/analyze_flow_wam_logs.py "$MOWE_S1_DDP/train_log.jsonl" \
  > "$MOWE_REPORT_ROOT/ddp_stage1_analysis.json"
jq '.primary.latest.future_horizon_metrics' "$MOWE_REPORT_ROOT/ddp_stage1_analysis.json"
```

进入 Stage 1 正式余下训练前：

- 四个 horizon 的 validation predicted smooth-L1 均有限。
- H=4/8/16 的 predicted smooth-L1 相对 `current_copy_smooth_l1` 至少出现稳定改善；推荐初始 promotion 目标为平均改善 10%。
- validation loss 不连续三个 eval point 恶化。
- action-distance gate 未长期接近 0。
- view weights 有限、和为 1，且没有从训练初期立刻完全塌缩到单视角。

重新签发 step-1000 readiness 后，用下面的完整命令继续。尽量连续跑完；若主动停止，恢复前必须针对新的 `checkpoint_latest.pt` 重新签发 checkpoint-bound readiness：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/pretrain_nominal_flow_wam.py \
  --config configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S1_DDP" \
  --max-steps 50000 \
  --stop-step 50000 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 500 \
  --log-freq 10 \
  --resume "$MOWE_S1_DDP/checkpoint_latest.pt" \
  --long-run-readiness-report "$MOWE_REPORT_ROOT/readiness_stage1_step1000.json" \
  | tee "$MOWE_REPORT_ROOT/ddp_stage1_1000_50000.log"
```

## 11. 8 卡 Stage 2：oracle expert warm-start

Stage 1 通过 future-vs-copy-current 门槛后才能开始。

```bash
export MOWE_S2_DDP="$MOWE_DDP_ROOT/stage2"
export MOWE_S1_FINAL="$MOWE_S1_DDP/checkpoint_latest.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/warmstart_skill_flow_experts.py \
  --config configs/mowe_wam/ddp8_warmstart_skill_flow_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S2_DDP" \
  --max-steps 50000 \
  --stop-step 100 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --route-mode oracle \
  --save-freq 25 \
  --log-freq 5 \
  --init-wam "$MOWE_S1_FINAL" \
  | tee "$MOWE_REPORT_ROOT/ddp_stage2_0_100.log"
```

先按单卡 Stage 2 的硬门槛检查。通过后为 Stage 2 step-100 checkpoint 生成 readiness：

```bash
python scripts/audit_long_training_readiness.py \
  --config configs/mowe_wam/ddp8_warmstart_skill_flow_feature_store.yaml \
  --store "$MOWE_STORE_FORMAL" \
  --feature-audit "$MOWE_REPORT_ROOT/feature_store_audit.json" \
  --equivalence-report "$MOWE_REPORT_ROOT/feature_equivalence_100.json" \
  --soak-report "$MOWE_REPORT_ROOT/feature_store_soak_8rank.json" \
  --ddp-runtime-audit "$MOWE_REPORT_ROOT/ddp_runtime_8gpu.json" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --checkpoint "$MOWE_S2_DDP/checkpoint_latest.pt" \
  --checkpoint-mode resume \
  --world-size 8 \
  --output "$MOWE_REPORT_ROOT/readiness_stage2_step100.json"
```

继续到 50,000：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/warmstart_skill_flow_experts.py \
  --config configs/mowe_wam/ddp8_warmstart_skill_flow_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S2_DDP" \
  --max-steps 50000 \
  --stop-step 50000 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --route-mode oracle \
  --save-freq 500 \
  --log-freq 10 \
  --resume "$MOWE_S2_DDP/checkpoint_latest.pt" \
  --long-run-readiness-report "$MOWE_REPORT_ROOT/readiness_stage2_step100.json" \
  | tee "$MOWE_REPORT_ROOT/ddp_stage2_100_50000.log"
```

Stage 2 completion 建议目标：

- 六个 motor experts 全程都有覆盖和梯度。
- `route_mode_diagnostics.hard_predicted.route_accuracy` 推荐 ≥0.60；不要把 oracle 主分支中恒等于标签的 route accuracy 当成预测能力。
- 至少 5/6 motor skills 的 final endpoint L1 优于 nominal endpoint L1。
- residual clip fraction 推荐 <5%，否则说明 residual cap 经常饱和。
- null residual violation 必须为 0。

这些 accuracy 数字是首轮 promotion 目标，不是论文预注册结论；若真实数据表明不合理，应记录原因并在所有对比实验前冻结新门槛。

## 12. 8 卡 Stage 3：joint routing

```bash
export MOWE_S3_DDP="$MOWE_DDP_ROOT/stage3"
export MOWE_S2_FINAL="$MOWE_S2_DDP/checkpoint_latest.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train_flow_wam_skill_moe.py \
  --config configs/mowe_wam/ddp8_train_flow_wam_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S3_DDP" \
  --max-steps 30000 \
  --stop-step 100 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 25 \
  --log-freq 5 \
  --init-wam "$MOWE_S2_FINAL" \
  | tee "$MOWE_REPORT_ROOT/ddp_stage3_0_100.log"
```

生成 Stage 3 readiness：

```bash
python scripts/audit_long_training_readiness.py \
  --config configs/mowe_wam/ddp8_train_flow_wam_feature_store.yaml \
  --store "$MOWE_STORE_FORMAL" \
  --feature-audit "$MOWE_REPORT_ROOT/feature_store_audit.json" \
  --equivalence-report "$MOWE_REPORT_ROOT/feature_equivalence_100.json" \
  --soak-report "$MOWE_REPORT_ROOT/feature_store_soak_8rank.json" \
  --ddp-runtime-audit "$MOWE_REPORT_ROOT/ddp_runtime_8gpu.json" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --checkpoint "$MOWE_S3_DDP/checkpoint_latest.pt" \
  --checkpoint-mode resume \
  --world-size 8 \
  --output "$MOWE_REPORT_ROOT/readiness_stage3_step100.json"
```

继续到 30,000：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train_flow_wam_skill_moe.py \
  --config configs/mowe_wam/ddp8_train_flow_wam_feature_store.yaml \
  --feature-store "$MOWE_STORE_FORMAL" \
  --checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --teacher-checkpoint "$MOWE_DINO_SNAPSHOT" \
  --skill-expert-config "$MOWE_SKILL_CONFIG" \
  --output-dir "$MOWE_S3_DDP" \
  --max-steps 30000 \
  --stop-step 30000 \
  --grad-accumulation-steps 1 \
  --flow-solver-steps 4 \
  --precision bf16 \
  --save-freq 500 \
  --log-freq 10 \
  --resume "$MOWE_S3_DDP/checkpoint_latest.pt" \
  --long-run-readiness-report "$MOWE_REPORT_ROOT/readiness_stage3_step100.json" \
  | tee "$MOWE_REPORT_ROOT/ddp_stage3_100_30000.log"
```

Stage 3 completion 建议目标：

- hard-predicted endpoint 不劣于 nominal，且接近 oracle route endpoint。
- `route_mode_diagnostics.hard_predicted.route_accuracy` 推荐 ≥0.65；在 hard-predicted 离线评估中 boundary F1 推荐 ≥0.50。
- 六个 motor expert 均有使用，不能只剩 1～2 个 dominant experts。
- `null_motion_zero_violation_count=0`。
- residual norm≤0.5，clip fraction 推荐 <5%。
- execution reason histogram 同时可观测 default/caution；high-risk 不应几乎每次触发。
- 训练指标只是进入 simulator 的门槛，不是最终模型选择依据。

## 13. LIBERO simulator 评测

### 13.1 one-task/1-trial smoke

```bash
export MOWE_S3_FINAL="$MOWE_S3_DDP/checkpoint_latest.pt"

CUDA_VISIBLE_DEVICES=0 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python scripts/eval_libero_temporal_skill.py \
  --config configs/mowe_wam/train_flow_wam_skill_moe.yaml \
  --policy-checkpoint "$MOWE_S3_FINAL" \
  --backbone-checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --simulator \
  --task-suite libero_spatial \
  --task-id 0 \
  --trials 1 \
  --flow-seed 7 \
  --seed 7 \
  --output-jsonl "$MOWE_REPORT_ROOT/libero_spatial_task0_smoke.jsonl" \
  --summary-output "$MOWE_REPORT_ROOT/libero_spatial_task0_smoke_summary.json"
```

达标条件：环境 reset/step 无异常，action 全部有限，gripper sign 正确，policy query/prefix trace 可读，旧 suffix 未执行。单 trial 成败不作为性能判断。

### 13.2 每个 suite 的小样本 gate

先对四个 suite 各跑每任务 5 trials：

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python scripts/eval_libero_temporal_skill.py \
    --config configs/mowe_wam/train_flow_wam_skill_moe.yaml \
    --policy-checkpoint "$MOWE_S3_FINAL" \
    --backbone-checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
    --backbone-revision "$MOWE_OPENVLA_REVISION" \
    --simulator \
    --task-suite "$suite" \
    --all-tasks \
    --trials 5 \
    --flow-seed 7 \
    --seed 7 \
    --output-jsonl "$MOWE_REPORT_ROOT/${suite}_5trial.jsonl" \
    --summary-output "$MOWE_REPORT_ROOT/${suite}_5trial_summary.json"
done
```

达标条件：四个 suite 都无 crash/NaN，成功率不是全零，执行长度不是几乎全部 high-risk 短前缀。若全零，先查 action scale、gripper sign、backbone identity 和 observation orientation，不要先加训练步数。

### 13.3 正式 full-suite

将 `--trials 5` 改为 `--trials 50`。使用 `--resume-results` 恢复中断结果，不能混用不同 checkpoint、seed 或 backbone：

```bash
python scripts/eval_libero_temporal_skill.py \
  --config configs/mowe_wam/train_flow_wam_skill_moe.yaml \
  --policy-checkpoint "$MOWE_S3_FINAL" \
  --backbone-checkpoint "$MOWE_OPENVLA_SNAPSHOT" \
  --backbone-revision "$MOWE_OPENVLA_REVISION" \
  --simulator \
  --task-suite libero_spatial \
  --all-tasks \
  --trials 50 \
  --flow-seed 7 \
  --seed 7 \
  --output-jsonl "$MOWE_REPORT_ROOT/libero_spatial_50trial.jsonl" \
  --summary-output "$MOWE_REPORT_ROOT/libero_spatial_50trial_summary.json" \
  --resume-results
```

四个 suite 和至少三个随机种子完成后，才可报告正式 success rate。

## 14. 如何判断“可以进入下一步”

| 当前步骤 | 必须看到的输出 | 才能进入 |
|---|---|---|
| 环境 | CUDA/BF16/依赖检查通过 | 模型下载 |
| backbone identity | 40 位 revision + identity SHA-256 | real preflight |
| RLDS audit | exact join=100%，H=16 windows>0 | conversion |
| real preflight | `preflight_passed` + backward + finite loss | store smoke |
| store smoke | shape 正确、checksum 通过、formal=false | full conversion |
| formal store | counts match、formal=true、全 checksum | equivalence |
| equivalence | 100/100、passed=true | 单卡/8卡训练 |
| 单卡 Stage 1/2/3 | 各 100-step 硬门槛通过 | 8卡正式 lineage |
| 8-rank soak + runtime | 全 rank pass、同 node/boot/cgroup | readiness |
| Stage 1 | future 明显优于 copy-current | Stage 2 |
| Stage 2 | 六 expert 有效、hard-predicted routing 达到门槛 | Stage 3 |
| Stage 3 | predicted routing 不塌缩、endpoint 有效 | simulator smoke |
| simulator smoke | 无 crash/NaN/动作合同错误 | full-suite |

## 15. 常见错误的处理顺序

### CUDA OOM

1. 转换时先把 `--encode-batch-size 8` 降到 4 或 2。
2. 训练不要改变 effective global batch；单卡保持 batch1，通过 accumulation 保持8。
3. 检查是否误用 raw RLDS/online 7B 训练，而不是 feature store。
4. 不要先降低 action chunk 或模型宽度，这会改变论文合同。

### Feature store 被拒绝

检查：

- 是否误用了带 `--limit-episodes` 的 smoke store。
- H=16/action chunk=16 是否与 config 一致。
- teacher checkpoint 字符串和 OpenVLA identity 是否与 store manifest 一致。
- expected/actual counts 和 shard checksums 是否全部通过。

### Resume 被拒绝

只允许同 stage `--resume`；跨 stage 必须 `--init-wam`。同 stage 不得改变：

- `max_steps`。
- effective global batch。
- LR/warmup/loss/route/action-condition schedule。
- H=16 window 和 execution contract。

可以改变：`stop_step`、`save_freq`、`log_freq`。

### Readiness report 被拒绝

report 与 checkpoint step、stage、store、config、node、boot、cgroup 绑定。主动停在新 checkpoint 后必须重新签发，不能沿用旧 checkpoint 的 report。

### 训练 loss 正常但 simulator 全零

按顺序检查：

1. policy checkpoint 是否为 Stage 3 joint。
2. 在线 eval backbone 是否与 store/checkpoint identity 完全一致。
3. 6D motion 是否正确反归一化。
4. gripper canonical `0/1` 到 LIBERO `+1/-1` 是否正确。
5. primary/wrist 图像是否颠倒或旋转。
6. execution 是否几乎全被 high-risk 截成 1～2 步。
7. 最后才考虑学习率、训练步数和模型容量。

## 16. 每个阶段应保留的产物

```text
outputs/cloud_reports/
  environment/
  libero_rlds_h16_audit.json
  skill_experts_h16.json
  feature_store_audit.json
  feature_equivalence_100.json
  feature_store_soak_8rank.json
  ddp_runtime_8gpu.json
  readiness_stage*.json
  libero_*_summary.json

outputs/cloud_single_gpu/stage{1,2,3}/
  config_resolved.json
  train_log.jsonl
  validation_log.jsonl
  checkpoint_latest.pt
  checkpoint_latest.pt.metadata.json

outputs/cloud_ddp8/stage{1,2,3}/
  config_resolved.json
  train_log.jsonl
  validation_log.jsonl
  checkpoint_latest.pt
  checkpoint_latest.pt.metadata.json
```

不要只保存终端截图。正式证据以 resolved config、JSON/JSONL、checkpoint metadata、store manifest 和 simulator episode records 为准。
