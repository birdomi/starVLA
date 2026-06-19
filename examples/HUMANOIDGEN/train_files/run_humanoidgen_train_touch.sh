#!/usr/bin/env bash
set -euo pipefail

# Train QwenGR00T_touch on HUMANOIDGEN with a pretrained tactile encoder.
# Override from shell if needed:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash examples/HUMANOIDGEN/train_files/run_humanoidgen_train_touch.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
unset NCCL_IB_HCA
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}

Framework_name=${Framework_name:-QwenGR00T_touch}
# Freeze the VLM backbone by default. Override with freeze_module_list='' to train it.
freeze_module_list="${freeze_module_list-qwen_vl_interface}"
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3.5-2B}
config_yaml=${config_yaml:-./examples/HUMANOIDGEN/train_files/starvla_train_humanoidgen.yaml}
data_root_dir=${data_root_dir:-${HUMANOIDGEN_DATA:-playground/Datasets/HUMANOIDGEN_DATA}}
data_mix=${data_mix:-humanoidgen_all}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
run_id=${run_id:-humanoidgen_qwengroot_touch}
per_device_batch_size=${per_device_batch_size:-64}
video_backend=${video_backend:-pyav}
max_train_steps=${max_train_steps:-30000}
save_interval=${save_interval:-10000}
action_dim=${action_dim:-26}
state_dim=${state_dim:-26}
action_horizon=${action_horizon:-16}

tactile_encoder_ckpt=${tactile_encoder_ckpt:-playground/Pretrained_models/tactile_encoder/epoch-0200-all.ckpt}
touch_model_size=${touch_model_size:-tiny}
train_touch_encoder=${train_touch_encoder:-true}
touch_token_source=${touch_token_source:-x_tokens}
derive_finger_angles_from_state=${derive_finger_angles_from_state:-true}
joint_contact_key=${joint_contact_key:-joint_contact}

if [[ ! -f "${tactile_encoder_ckpt}" ]]; then
  echo "[ERROR] tactile encoder checkpoint not found: ${tactile_encoder_ckpt}" >&2
  exit 1
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

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

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port "${MAIN_PROCESS_PORT:-29501}" \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_dim "${action_dim}" \
  --framework.action_model.state_dim "${state_dim}" \
  --framework.action_model.action_horizon "${action_horizon}" \
  --framework.action_model.touch.enabled true \
  --framework.action_model.touch.checkpoint_encoder "${tactile_encoder_ckpt}" \
  --framework.action_model.touch.model_size "${touch_model_size}" \
  --framework.action_model.touch.train_encoder "${train_touch_encoder}" \
  --framework.action_model.touch.token_source "${touch_token_source}" \
  --framework.action_model.touch.require_touch true \
  --framework.action_model.touch.derive_finger_angles_from_state "${derive_finger_angles_from_state}" \
  --framework.action_model.touch.joint_contact_key "${joint_contact_key}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.video_backend "${video_backend}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
