#!/usr/bin/env bash
set -euo pipefail

# Download GR00T N1.7 checkpoint into playground/Pretrained_models.
# GR00T N1.7 public HF repo name is not hardcoded; set GR00T_HF_REPO.


GR00T_HF_REPO=${GR00T_HF_REPO:-${1:-}}
GR00T_DIR=${GR00T_DIR:-playground/Pretrained_models/GR00T-N1.7}

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[ERROR] huggingface-cli not found. Install huggingface_hub first." >&2
  exit 1
fi


if [[ -z "${GR00T_HF_REPO}" ]]; then
  echo "[ERROR] GR00T_HF_REPO is empty. GR00T_DIR is only the local save path." >&2
  echo "        example: bash $0 nvidia/<actual-gr00t-n1.7-repo>" >&2
  exit 2
fi

echo "[INFO] downloading GR00T checkpoint: ${GR00T_HF_REPO} -> ${GR00T_DIR}"
hf download "${GR00T_HF_REPO}" --local-dir "${GR00T_DIR}"
