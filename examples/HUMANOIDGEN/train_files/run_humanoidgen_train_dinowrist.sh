#!/usr/bin/env bash
set -euo pipefail

# Head camera -> Qwen-VL. Wrist cameras -> DINO -> GR00T DiT prefix.
# Override from shell if needed:
#   HUMANOIDGEN_DATA=/path/to/HUMANOIDGEN_DATA CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash examples/HUMANOIDGEN/train_files/run_humanoidgen_train_dinowrist.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
unset NCCL_IB_HCA
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}

Framework_name=${Framework_name:-QwenGR00T_DinoWrist}
freeze_module_list=${freeze_module_list:-}
base_vlm=${base_vlm:-playground/Pretrained_models/Qwen3.5-2B}
config_yaml=${config_yaml:-./examples/HUMANOIDGEN/train_files/starvla_train_humanoidgen_dinowrist.yaml}
data_root_dir=${data_root_dir:-${HUMANOIDGEN_DATA:-playground/Datasets/HUMANOIDGEN_DATA}}
data_mix=${data_mix:-humanoidgen_all}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
run_id=${run_id:-humanoidgen_qwengroot_dinowrist}
per_device_batch_size=${per_device_batch_size:-32}
video_backend=${video_backend:-pyav}
max_train_steps=${max_train_steps:-100000}
save_interval=${save_interval:-10000}
action_dim=${action_dim:-26}
state_dim=${state_dim:-26}
action_horizon=${action_horizon:-16}
dino_backbone=${dino_backbone:-dinov3_vits16}
dino_checkpoint_path=${dino_checkpoint_path:-playground/Pretrained_models/DINOv3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}
train_dino_encoder=${train_dino_encoder:-false}
num_wrist_query_tokens=${num_wrist_query_tokens:-16}

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
  --main_process_port "${MAIN_PROCESS_PORT:-29502}" \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.dino.dino_backbone "${dino_backbone}" \
  --framework.dino.checkpoint_path "${dino_checkpoint_path}" \
  --framework.dino.train_encoder "${train_dino_encoder}" \
  --framework.dino.num_wrist_query_tokens "${num_wrist_query_tokens}" \
  --framework.action_model.action_dim "${action_dim}" \
  --framework.action_model.state_dim "${state_dim}" \
  --framework.action_model.action_horizon "${action_horizon}" \
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
