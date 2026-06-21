#!/usr/bin/env bash
set -euo pipefail

# Single-node training. Override these from the shell if needed:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_PROCESSES=4 bash examples/LIBERO/train_files/run_libero_train.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Keep NCCL local and predictable for single-node runs.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
unset NCCL_IB_HCA
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

# Avoid interactive W&B prompts. Change to online if you want logging.
export WANDB_MODE=${WANDB_MODE:-disabled}

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=${Framework_name:-QwenGR00T}
freeze_module_list=''
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3.5-2B}
config_yaml=${config_yaml:-./examples/LIBERO/train_files/starvla_cotrain_libero.yaml}
libero_data_root=${libero_data_root:-playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${data_mix:-libero_all}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
run_id=${run_id:-libero4in1_qwen3_5_groot}
per_device_batch_size=${per_device_batch_size:-64}
video_backend=${video_backend:-torchcodec}
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

num_processes=${NUM_PROCESSES:-$(python - <<'PY'
import os
print(len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
PY
)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.video_backend "${video_backend}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
