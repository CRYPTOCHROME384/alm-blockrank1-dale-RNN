#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/jingyi.xu/code_rnn_blockrank1_dale"
TRAIN_PY="${PROJECT_ROOT}/current_rnn/main_train_alm_current_blockrank1_dale.py"
DEFAULT_CONFIG="${PROJECT_ROOT}/current_rnn/parameters_ei_mixed_blockrank1_dale.json"
LOG_DIR="${PROJECT_ROOT}/logs"

CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
GPU_ID="${2:-0}"
RUN_TAG="${3:-gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR_SUFFIX="${4:-}"
LOG_PATH="${LOG_DIR}/train_blockrank1_dale_${RUN_TAG}.log"

STAMP="$(date +%Y%m%d_%H%M%S)"
CFG_BASE="$(basename "${CONFIG_PATH}" .json)"
CFG_SUFFIX="${CFG_BASE#parameters_ei_mixed_blockrank1_dale_}"
if [[ "${CFG_SUFFIX}" == "${CFG_BASE}" ]]; then
  CFG_SUFFIX="manual"
fi
if [[ -n "${OUT_DIR_SUFFIX}" ]]; then
  OUT_DIR="${PROJECT_ROOT}/results_current/blockrank1_dale_${OUT_DIR_SUFFIX}"
else
  OUT_DIR="${PROJECT_ROOT}/results_current/blockrank1_dale_${CFG_SUFFIX}_${STAMP}"
fi

mkdir -p "${LOG_DIR}"

nohup python -u "${TRAIN_PY}" \
  --config "${CONFIG_PATH}" \
  --out_dir "${OUT_DIR}" \
  --cuda_device "${GPU_ID}" \
  --device cuda \
  --require_cuda true \
  > "${LOG_PATH}" 2>&1 &

PID=$!
echo "[OK] launched pid=${PID}"
echo "[OK] log=${LOG_PATH}"
echo "[OK] gpu=${GPU_ID}"
echo "[OK] out_dir=${OUT_DIR}"
