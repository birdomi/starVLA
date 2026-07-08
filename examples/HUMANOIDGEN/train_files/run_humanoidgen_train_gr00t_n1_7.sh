#!/usr/bin/env bash
set -euo pipefail

# Train HUMANOIDGEN from GR00T N1.7-style pretrained action weights.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
unset NCCL_IB_HCA
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export GR00T_LOAD_LOG_LIMIT=${GR00T_LOAD_LOG_LIMIT:-800}

Framework_name=${Framework_name:-GR00T_N1_7}
freeze_module_list=${freeze_module_list:-qwen_vl_interface}
base_vlm=${base_vlm:-nvidia/Cosmos-Reason2-2B}
attn_implementation=${attn_implementation:-sdpa}
config_yaml=${config_yaml:-./examples/HUMANOIDGEN/train_files/starvla_train_humanoidgen.yaml}
data_root_dir=${data_root_dir:-${HUMANOIDGEN_DATA:-playground/Datasets/HUMANOIDGEN_DATA}}
data_mix=${data_mix:-humanoidgen_all}
run_root_dir=${run_root_dir:-./playground/Checkpoints}
run_id=${run_id:-humanoidgen_gr00t_n1_7}
per_device_batch_size=${per_device_batch_size:-32}
video_backend=${video_backend:-pyav}
max_train_steps=${max_train_steps:-100000}
save_interval=${save_interval:-10000}

action_model_type=${action_model_type:-DiT-L}
hidden_size=${hidden_size:-1024}
action_dim=${action_dim:-26}
state_dim=${state_dim:-26}
action_horizon=${action_horizon:-16}
repeated_diffusion_steps=${repeated_diffusion_steps:-4}
dit_num_layers=${dit_num_layers:-32}
vlm_hidden_state_index=${vlm_hidden_state_index:-16}

gr00t_pretrained_path=${gr00t_pretrained_path:-playground/Pretrained_models/GR00T-N1.7}
gr00t_load_prefixes=${gr00t_load_prefixes:-qwen_vl_interface.model,action_model.model}

if [[ ! -e "${gr00t_pretrained_path}" ]]; then
  echo "[ERROR] GR00T N1.7 checkpoint not found: ${gr00t_pretrained_path}" >&2
  echo "        prepare it with examples/HUMANOIDGEN/train_files/prepare_gr00t_n1_7_pretrained.sh" >&2
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
  --main_process_port "${MAIN_PROCESS_PORT:-29504}" \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.attn_implementation "${attn_implementation}" \
  --framework.action_model.action_model_type "${action_model_type}" \
  --framework.action_model.hidden_size "${hidden_size}" \
  --framework.action_model.action_dim "${action_dim}" \
  --framework.action_model.state_dim "${state_dim}" \
  --framework.action_model.action_horizon "${action_horizon}" \
  --framework.action_model.repeated_diffusion_steps "${repeated_diffusion_steps}" \
  --framework.action_model.diffusion_model_cfg.num_layers "${dit_num_layers}" \
  --framework.gr00t_n1_7.checkpoint_path "${gr00t_pretrained_path}" \
  --framework.gr00t_n1_7.load_prefixes "${gr00t_load_prefixes}" \
  --framework.gr00t_n1_7.vlm_hidden_state_index "${vlm_hidden_state_index}" \
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
