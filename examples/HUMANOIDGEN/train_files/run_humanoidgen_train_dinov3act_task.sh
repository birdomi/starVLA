#!/usr/bin/env bash
set -euo pipefail

# Train DINOv3ACT on one HUMANOIDGEN task.
# Pick by TASK_INDEX=0..15 or exact TASK_NAME="Close the drawer."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
unset NCCL_IB_HCA
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}

Framework_name=${Framework_name:-DINOv3ACT}
config_yaml=${config_yaml:-./examples/HUMANOIDGEN/train_files/starvla_train_humanoidgen.yaml}
data_root_dir=${data_root_dir:-${HUMANOIDGEN_DATA:-playground/Datasets/HUMANOIDGEN_DATA}}
data_mix=${data_mix:-humanoidgen_all}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
TASK_INDEX=${TASK_INDEX:-0}
TASK_NAME=${TASK_NAME:-}
run_id=${run_id:-humanoidgen_dinov3act_task${TASK_INDEX}}
per_device_batch_size=${per_device_batch_size:-32}
video_backend=${video_backend:-pyav}
max_train_steps=${max_train_steps:-50000}
save_interval=${save_interval:-10000}
logging_frequency=${logging_frequency:-50}
eval_interval=${eval_interval:-1000}

DINO_BACKBONE=${DINO_BACKBONE:-dinov3_vits16}
DINO_CHECKPOINT=${DINO_CHECKPOINT:-playground/Pretrained_models/DINOv3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}
DINO_REPO_DIR=${DINO_REPO_DIR:-}
TRAIN_DINO_ENCODER=${TRAIN_DINO_ENCODER:-false}
MAX_VIEWS=${MAX_VIEWS:-3}
INCLUDE_CLS=${INCLUDE_CLS:-true}

action_dim=${action_dim:-26}
state_dim=${state_dim:-26}
action_horizon=${action_horizon:-16}
hidden_dim=${hidden_dim:-512}
num_encoder_layers=${num_encoder_layers:-2}
num_decoder_layers=${num_decoder_layers:-4}
num_heads=${num_heads:-8}
dim_feedforward=${dim_feedforward:-2048}
dropout=${dropout:-0.1}
loss_type=${loss_type:-l1}
learning_rate=${learning_rate:-1.0e-4}
dino_lr=${dino_lr:-1.0e-5}
num_warmup_steps=${num_warmup_steps:-1000}

if [[ ! -d "${data_root_dir}" ]]; then
  echo "[ERROR] HUMANOIDGEN data not found: ${data_root_dir}" >&2
  exit 1
fi

if [[ -n "${DINO_CHECKPOINT}" && ! -e "${DINO_CHECKPOINT}" ]]; then
  echo "[ERROR] DINO checkpoint not found: ${DINO_CHECKPOINT}" >&2
  exit 1
fi

if [[ -n "${NUM_PROCESSES:-}" ]]; then
  num_processes="${NUM_PROCESSES}"
else
  cuda_devices_no_spaces="${CUDA_VISIBLE_DEVICES// /}"
  if [[ -z "${cuda_devices_no_spaces}" ]]; then
    num_processes=1
  else
    IFS="," read -r -a cuda_device_array <<< "${cuda_devices_no_spaces}"
    num_processes="${#cuda_device_array[@]}"
  fi
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

task_args=(--datasets.vla_data.task_index "${TASK_INDEX}")
if [[ -n "${TASK_NAME}" ]]; then
  task_args=(--datasets.vla_data.task_name "${TASK_NAME}")
fi

repo_args=()
if [[ -n "${DINO_REPO_DIR}" ]]; then
  repo_args=(--framework.dino.repo_dir "${DINO_REPO_DIR}")
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port "${MAIN_PROCESS_PORT:-29505}" \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.dino.dino_backbone "${DINO_BACKBONE}" \
  --framework.dino.checkpoint_path "${DINO_CHECKPOINT}" \
  "${repo_args[@]}" \
  --framework.dino.train_encoder "${TRAIN_DINO_ENCODER}" \
  --framework.dino.max_views "${MAX_VIEWS}" \
  --framework.dino.include_cls "${INCLUDE_CLS}" \
  --framework.action_model.action_model_type ACT \
  --framework.action_model.action_dim "${action_dim}" \
  --framework.action_model.state_dim "${state_dim}" \
  --framework.action_model.action_horizon "${action_horizon}" \
  --framework.action_model.hidden_dim "${hidden_dim}" \
  --framework.action_model.num_encoder_layers "${num_encoder_layers}" \
  --framework.action_model.num_decoder_layers "${num_decoder_layers}" \
  --framework.action_model.num_heads "${num_heads}" \
  --framework.action_model.dim_feedforward "${dim_feedforward}" \
  --framework.action_model.dropout "${dropout}" \
  --framework.action_model.loss_type "${loss_type}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  "${task_args[@]}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.video_backend "${video_backend}" \
  --trainer.learning_rate.base "${learning_rate}" \
  --trainer.learning_rate.action_model "${learning_rate}" \
  --trainer.learning_rate.dino_encoder "${dino_lr}" \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.num_warmup_steps "${num_warmup_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval "${eval_interval}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
