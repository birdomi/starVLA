#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WORKSPACE_DIR="$(cd "${STARVLA_DIR}/.." && pwd)"
HUMANOIDGEN_DIR="${HUMANOIDGEN_DIR:-${WORKSPACE_DIR}/HumanoidGen}"
HUMANOIDGEN_VENV="${HUMANOIDGEN_VENV:-${HUMANOIDGEN_DIR}/.venv}"
# CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/humanoidgen_qwengroot/checkpoints/steps_80000_pytorch_model.pt}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/humanoidgen_qwengroot_touch/checkpoints/steps_30000_pytorch_model.pt}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6695}"
ENV_ID="${ENV_ID:-open_drawer}"
TASK="${TASK:-}"

# HUMANOIDGEN eval tasks. If TASK is empty, the script automatically selects
# the matching instruction from ENV_ID. Uncomment/copy both lines below when
# you want to override the default task explicitly.
#
# ENV_ID=block_handover
# TASK="Hand over the block."
#
# ENV_ID=blocks_stack_easy
# TASK="Stack the blocks in the easy setting."
#
# ENV_ID=blocks_stack_hard
# TASK="Stack the blocks in the hard setting."
#
# ENV_ID=close_box_easy
# TASK="Close the box in the easy setting."
#
# ENV_ID=close_box_hard
# TASK="Close the box in the hard setting."
#
# ENV_ID=close_drawer
# TASK="Close the drawer."
#
# ENV_ID=close_laptop_easy
# TASK="Close the laptop in the easy setting."
#
# ENV_ID=close_laptop_hard
# TASK="Close the laptop in the hard setting."
#
# ENV_ID=dual_bottles_pick_easy
# TASK="Pick up the two bottles in the easy setting."
#
# ENV_ID=dual_bottles_pick_hard
# TASK="Pick up the two bottles in the hard setting."
#
# ENV_ID=handover_and_storage_cooperation
# TASK="Cooperatively hand over the object and place it into storage."
#
# ENV_ID=handover_and_storage
# TASK="Hand over the object and place it into storage."
#
# ENV_ID=open_box_hard
# TASK="Open the box in the hard setting."
#
# ENV_ID=open_drawer
# TASK="Open the drawer."
#
# ENV_ID=open_laptop_easy
# TASK="Open the laptop in the easy setting."
#
# ENV_ID=open_laptop_hard
# TASK="Open the laptop in the hard setting."

MAX_EPISODES="${MAX_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-600}"
MAX_DURATION_SECONDS="${MAX_DURATION_SECONDS:-20.0}"
ACTION_HORIZON="${ACTION_HORIZON:-0}"
UNNORM_KEY="${UNNORM_KEY:-}"
IMAGE_RESIZE="${IMAGE_RESIZE:-none}"
CAMERA_NAMES="${CAMERA_NAMES:-head_camera left_wrist_camera right_wrist_camera}"
INCLUDE_STATE="${INCLUDE_STATE:-1}"
INCLUDE_TOUCH="${INCLUDE_TOUCH:-1}"
TOUCH_FINGERTIP_INDICES="${TOUCH_FINGERTIP_INDICES:-2 4 6 8 10}"
RENDER_SCENE="${RENDER_SCENE:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SAVE_WHEN_FAIL="${SAVE_WHEN_FAIL:-0}"
CALCULATE_SUCCESS_RATE="${CALCULATE_SUCCESS_RATE:-1}"
RECORD_DATA="${RECORD_DATA:-0}"
RECORD_POINTCLOUD="${RECORD_POINTCLOUD:-0}"
SAVE_VIDEO_CAMERAS="${SAVE_VIDEO_CAMERAS:-right_wrist_camera left_wrist_camera}"
RENDER_SAVE_FREQ="${RENDER_SAVE_FREQ:-5}"

if [[ "${CKPT}" = /* ]]; then
  CKPT_ABS="${CKPT}"
else
  CKPT_ABS="$(cd "${STARVLA_DIR}" && realpath "${CKPT}")"
fi

if [[ "${CKPT_ABS}" == *"/checkpoints/"* ]]; then
  MODEL_ROOT="${CKPT_ABS%%/checkpoints/*}"
  FOLDER_NAME="$(basename "${MODEL_ROOT}")_checkpoints_$(basename "${CKPT_ABS}")"
elif [[ "${CKPT_ABS}" == *"/final_model/"* ]]; then
  MODEL_ROOT="${CKPT_ABS%%/final_model/*}"
  FOLDER_NAME="$(basename "${MODEL_ROOT}")_final_model_$(basename "${CKPT_ABS}")"
else
  MODEL_ROOT="$(dirname "$(dirname "${CKPT_ABS}")")"
  FOLDER_NAME="$(basename "$(dirname "${CKPT_ABS}")")_$(basename "${CKPT_ABS}")"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_ROOT}/results/${ENV_ID}/${FOLDER_NAME}}"
RESULTS_FILE="${RESULTS_FILE:-${OUTPUT_DIR}/success_rate.json}"

task_for_env() {
  case "$1" in
    block_handover) echo "Hand over the block." ;;
    blocks_stack_easy) echo "Stack the blocks in the easy setting." ;;
    blocks_stack_hard) echo "Stack the blocks in the hard setting." ;;
    close_box_easy) echo "Close the box in the easy setting." ;;
    close_box_hard) echo "Close the box in the hard setting." ;;
    close_drawer) echo "Close the drawer." ;;
    close_laptop_easy) echo "Close the laptop in the easy setting." ;;
    close_laptop_hard) echo "Close the laptop in the hard setting." ;;
    dual_bottles_pick_easy) echo "Pick up the two bottles in the easy setting." ;;
    dual_bottles_pick_hard) echo "Pick up the two bottles in the hard setting." ;;
    handover_and_storage_cooperation) echo "Cooperatively hand over the object and place it into storage." ;;
    handover_and_storage) echo "Hand over the object and place it into storage." ;;
    open_box_hard) echo "Open the box in the hard setting." ;;
    open_drawer) echo "Open the drawer." ;;
    open_laptop_easy) echo "Open the laptop in the easy setting." ;;
    open_laptop_hard) echo "Open the laptop in the hard setting." ;;
    *) echo "" ;;
  esac
}

if [[ -z "${TASK}" ]]; then
  TASK="$(task_for_env "${ENV_ID}")"
fi

if [[ ! -f "${HUMANOIDGEN_VENV}/bin/activate" ]]; then
  echo "HumanoidGen venv not found: ${HUMANOIDGEN_VENV}" >&2
  echo "Set HUMANOIDGEN_VENV=/path/to/venv or HUMANOIDGEN_DIR=/path/to/HumanoidGen." >&2
  exit 1
fi

source "${HUMANOIDGEN_VENV}/bin/activate"
export PYTHONPATH="${HUMANOIDGEN_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-humanoidgen}}"
mkdir -p "${MPLCONFIGDIR}"

USER_ARGS=("$@")

has_cli_arg() {
  local name
  local arg
  for name in "$@"; do
    for arg in "${USER_ARGS[@]}"; do
      if [[ "${arg}" == "${name}" || "${arg}" == "${name}="* ]]; then
        return 0
      fi
    done
  done
  return 1
}

add_value_arg() {
  local name="$1"
  local value="$2"
  if [[ -n "${value}" ]] && ! has_cli_arg "${name}"; then
    CMD+=("${name}" "${value}")
  fi
}

add_bool_arg() {
  local positive_name="$1"
  local negative_name="$2"
  local value="$3"
  if has_cli_arg "${positive_name}" "${negative_name}"; then
    return
  fi
  case "${value,,}" in
    1|true|yes|y|on) CMD+=("${positive_name}") ;;
    0|false|no|n|off) CMD+=("${negative_name}") ;;
    *) echo "Invalid boolean value for ${positive_name}: ${value}" >&2; exit 1 ;;
  esac
}

add_optional_bool_value_arg() {
  local name="$1"
  local value="$2"
  if has_cli_arg "${name}"; then
    return
  fi
  case "${value,,}" in
    1|true|yes|y|on) CMD+=("${name}" "True") ;;
    0|false|no|n|off) CMD+=("${name}" "False") ;;
    none|null) CMD+=("${name}" "None") ;;
    *) echo "Invalid boolean value for ${name}: ${value}" >&2; exit 1 ;;
  esac
}

add_list_arg() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]] || has_cli_arg "${name}"; then
    return
  fi
  local normalized="${value//,/ }"
  local items=()
  read -r -a items <<< "${normalized}"
  if [[ "${#items[@]}" -gt 0 ]]; then
    CMD+=("${name}" "${items[@]}")
  fi
}

CMD=(
  python "${HUMANOIDGEN_DIR}/humanoidgen/process/inference_starVLA.py"
)

if ! has_cli_arg "--env-id" "-env"; then
  CMD+=(--env-id "${ENV_ID}")
fi
add_value_arg "--host" "${HOST}"
add_value_arg "--port" "${PORT}"
add_value_arg "--task" "${TASK}"
add_value_arg "--max-episodes" "${MAX_EPISODES}"
add_value_arg "--max-steps" "${MAX_STEPS}"
add_value_arg "--max-duration-seconds" "${MAX_DURATION_SECONDS}"
add_value_arg "--action-horizon" "${ACTION_HORIZON}"
add_value_arg "--unnorm-key" "${UNNORM_KEY}"
add_value_arg "--image-resize" "${IMAGE_RESIZE}"
add_value_arg "--video-out-path" "${OUTPUT_DIR}"
add_value_arg "--results-file" "${RESULTS_FILE}"
add_value_arg "--render-save-freq" "${RENDER_SAVE_FREQ}"
add_list_arg "--camera-names" "${CAMERA_NAMES}"
add_list_arg "--touch-fingertip-indices" "${TOUCH_FINGERTIP_INDICES}"
add_list_arg "--save-video-cameras" "${SAVE_VIDEO_CAMERAS}"
add_optional_bool_value_arg "--render-scene" "${RENDER_SCENE}"
add_bool_arg "--include-state" "--no-include-state" "${INCLUDE_STATE}"
add_bool_arg "--include-touch" "--no-include-touch" "${INCLUDE_TOUCH}"
add_bool_arg "--save-video" "--no-save-video" "${SAVE_VIDEO}"
add_bool_arg "--save-when-fail" "--no-save-when-fail" "${SAVE_WHEN_FAIL}"
add_bool_arg "--calculate-success-rate" "--no-calculate-success-rate" "${CALCULATE_SUCCESS_RATE}"
add_bool_arg "--record-data" "--no-record-data" "${RECORD_DATA}"
add_bool_arg "--record-pointcloud" "--no-record-pointcloud" "${RECORD_POINTCLOUD}"

echo "[INFO] checkpoint : ${CKPT_ABS}"
echo "[INFO] output dir : ${OUTPUT_DIR}"
echo "[INFO] results    : ${RESULTS_FILE}"
echo "[INFO] server     : ${HOST}:${PORT}"
echo "[INFO] env/task   : ${ENV_ID} / ${TASK:-<empty>}"

cd "${HUMANOIDGEN_DIR}"
exec "${CMD[@]}" "$@"
