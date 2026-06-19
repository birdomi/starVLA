#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_humanoidgen.sh"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-}"
BASE_RESULTS_FILE="${RESULTS_FILE:-}"
# CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/humanoidgen_qwengroot/checkpoints/steps_80000_pytorch_model.pt}"
# CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/humanoidgen_qwengroot_touch/checkpoints/steps_30000_pytorch_model.pt}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/humanoidgen_qwengroot/checkpoints/steps_100000_pytorch_model.pt}"


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

if [[ -n "${TOTAL_RESULTS_FILE:-}" ]]; then
  SUMMARY_FILE="${TOTAL_RESULTS_FILE}"
elif [[ -n "${BASE_OUTPUT_DIR}" ]]; then
  SUMMARY_FILE="${BASE_OUTPUT_DIR%/}/total_success_rate.json"
elif [[ -n "${BASE_RESULTS_FILE}" ]]; then
  if [[ "${BASE_RESULTS_FILE}" == *.json ]]; then
    SUMMARY_FILE="${BASE_RESULTS_FILE%.json}_TOTAL.json"
  else
    SUMMARY_FILE="${BASE_RESULTS_FILE}_TOTAL"
  fi
else
  SUMMARY_FILE="${MODEL_ROOT}/results/TOTAL/${FOLDER_NAME}/success_rate.json"
fi

for arg in "$@"; do
  case "${arg}" in
    --env-id|-env|--env-id=*|-env=*|--task|--task=*|--results-file|--results-file=*|--video-out-path|--video-out-path=*)
      echo "eval_humanoidgen_all.sh controls task/output paths internally; do not pass ${arg}." >&2
      exit 1
      ;;
  esac
done

TASKS=(
  "block_handover|Hand over the block."
  "blocks_stack_easy|Stack the blocks in the easy setting."
  "blocks_stack_hard|Stack the blocks in the hard setting."
  "close_box_easy|Close the box in the easy setting."
  "close_box_hard|Close the box in the hard setting."
  "close_drawer|Close the drawer."
  "close_laptop_easy|Close the laptop in the easy setting."
  "close_laptop_hard|Close the laptop in the hard setting."
  "dual_bottles_pick_easy|Pick up the two bottles in the easy setting."
  "dual_bottles_pick_hard|Pick up the two bottles in the hard setting."
  "handover_and_storage_cooperation|Cooperatively hand over the object and place it into storage."
  "handover_and_storage|Hand over the object and place it into storage."
  "open_box_hard|Open the box in the hard setting."
  "open_drawer|Open the drawer."
  "open_laptop_easy|Open the laptop in the easy setting."
  "open_laptop_hard|Open the laptop in the hard setting."
)

failures=()
summary_args=()

for index in "${!TASKS[@]}"; do
  entry="${TASKS[$index]}"
  env_id="${entry%%|*}"
  task="${entry#*|}"

  env_args=(
    "ENV_ID=${env_id}"
    "TASK=${task}"
  )

  if [[ -n "${BASE_OUTPUT_DIR}" ]]; then
    task_output_dir="${BASE_OUTPUT_DIR%/}/${env_id}"
    env_args+=("OUTPUT_DIR=${task_output_dir}")
    if [[ -z "${BASE_RESULTS_FILE}" ]]; then
      task_results_file="${task_output_dir}/success_rate.json"
      env_args+=("RESULTS_FILE=${task_results_file}")
    fi
  fi

  if [[ -n "${BASE_RESULTS_FILE}" ]]; then
    if [[ "${BASE_RESULTS_FILE}" == *.json ]]; then
      task_results_file="${BASE_RESULTS_FILE%.json}_${env_id}.json"
    else
      task_results_file="${BASE_RESULTS_FILE}_${env_id}"
    fi
    env_args+=("RESULTS_FILE=${task_results_file}")
  fi

  if [[ -z "${BASE_OUTPUT_DIR}" && -z "${BASE_RESULTS_FILE}" ]]; then
    task_output_dir="${MODEL_ROOT}/results/${env_id}/${FOLDER_NAME}"
    task_results_file="${task_output_dir}/success_rate.json"
    env_args+=("OUTPUT_DIR=${task_output_dir}" "RESULTS_FILE=${task_results_file}")
  fi

  summary_args+=("${env_id}" "${task_results_file}")

  echo "[ALL] ($((index + 1))/${#TASKS[@]}) ${env_id} / ${task}"
  if env "${env_args[@]}" bash "${EVAL_SCRIPT}" "$@"; then
    echo "[ALL] done: ${env_id}"
  else
    status=$?
    failures+=("${env_id}:${status}")
    echo "[ALL] failed: ${env_id} (exit ${status})" >&2
    case "${CONTINUE_ON_ERROR,,}" in
      1|true|yes|y|on) ;;
      *) exit "${status}" ;;
    esac
  fi
done

"${PYTHON_BIN:-python3}" - "${SUMMARY_FILE}" "${summary_args[@]}" <<'PY'
import json
import os
import sys

summary_file = sys.argv[1]
pairs = sys.argv[2:]
task_summaries = []
missing_results = []
success_count = 0
episode_count = 0

for env_id, results_file in zip(pairs[0::2], pairs[1::2]):
    if not os.path.exists(results_file):
        missing_results.append({"env_id": env_id, "results_file": results_file})
        print(f"[TOTAL] missing result: {env_id} -> {results_file}", file=sys.stderr)
        continue

    with open(results_file, "r") as file:
        data = json.load(file)

    results = data.get("results", [])
    task_total = len(results)
    task_success = sum(1 for item in results if bool(item.get("success", False)))
    task_rate = (task_success / task_total * 100.0) if task_total else 0.0
    success_count += task_success
    episode_count += task_total
    task_summaries.append(
        {
            "env_id": env_id,
            "success_count": task_success,
            "episode_count": task_total,
            "success_rate": task_rate,
            "results_file": results_file,
        }
    )
    print(f"[TOTAL] {env_id}: {task_success}/{task_total} = {task_rate:.2f}%")

total_rate = (success_count / episode_count * 100.0) if episode_count else 0.0
summary = {
    "success_rate": total_rate,
    "success_count": success_count,
    "episode_count": episode_count,
    "tasks": task_summaries,
    "missing_results": missing_results,
}

summary_dir = os.path.dirname(summary_file)
if summary_dir:
    os.makedirs(summary_dir, exist_ok=True)
with open(summary_file, "w") as file:
    json.dump(summary, file, indent=4)

print(f"[TOTAL] overall: {success_count}/{episode_count} = {total_rate:.2f}%")
print(f"[TOTAL] summary saved to {summary_file}")
PY

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "[ALL] failures: ${failures[*]}" >&2
  exit 1
fi

echo "[ALL] completed all ${#TASKS[@]} HumanoidGen tasks."
