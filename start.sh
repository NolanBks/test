#!/usr/bin/env bash

mkdir -p /opt/huawei/explorer-env/
ln -s /opt/huawei/dataset/ /opt/huawei/explorer-env/dataset
ln -s /opt/huawei/quoteModel/ /opt/huawei/explorer-env/quoteModel

# INIT CUDA
export CUDA_HOME=/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118
export PATH=/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118/bin:$PATH
export LD_LIBRARY_PATH=/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118/lib64$LD_LIBRARY_PATH

# INIT CONDA
export MINICONDA_PATH=/opt/huawei/explorer-env/dataset/Common_wl/miniconda3
export PATH=$MINICONDA_PATH/bin:$PATH
which conda
which python
$MINICONDA_PATH/bin/conda init bash
source "$MINICONDA_PATH/etc/profile.d/conda.sh"

# ACTIVATE CONDA
VIRTUAL_ENV="starVLA_flash_xmx"
if conda env list | grep -q "$VIRTUAL_ENV"; then
    conda activate "$VIRTUAL_ENV"
    echo "PYTHON PATH: $(which python)"
else
    echo "Warning: $VIRTUAL_ENV do not exists"
    echo "Avaiable envs:"
    conda env list
fi
which python

# Distributed settingd
export DEEPSPEED_TIMEOUT=7200
export NCCL_TIMEOUT=7200 
# export NCCL_CONNECT_TIMEOUT=54000
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TRITON_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1

export NCCL_TIMEOUT_S=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BUFFSIZE=8388608   # 8 MB
export NCCL_CUDA_MALLOC=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export STDBUF_ONESELF=1

# # Change to running directory
# cd starVLA
# echo "starVLA"

echo "Current Dir: $(pwd)"
export PYTHONPATH="${pwd}:${PYTHONPATH}"

#ln -s  /opt/huawei/explorer-env/dataset/ViTpretrained_wulan02/VLA_datasets/robocasa_365/tasks ./playground/Datasets/robocasa365/v1.0
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# Distributed training configuration
MASTER_ADDR=${VC_WORKER_HOSTS%%,*}
MASTER_PORT=8524
NNODES=$MA_NUM_HOSTS
NGPUS=$((NNODES*MA_NUM_GPUS))
# NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}

# ---- training knobs (edit here) ----
CONFIG_YAML=$1
BATCH=$4
MAX_STEPS=$5
EVAL_EVERY=$5
SAVE_EVERY=$6
LOG_EVERY=$7
run_root_dir=$2
run_id=$3
output_dir=${run_root_dir}/${run_id}

mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --multi_gpu \
  --num_machines $NNODES \
  --num_processes $NGPUS \
  --machine_rank $VC_TASK_INDEX \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml ${CONFIG_YAML} \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --trainer.eval_interval "${EVAL_EVERY}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
  
