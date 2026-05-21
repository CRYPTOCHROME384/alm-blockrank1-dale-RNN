#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/jingyi.xu/code_rnn_blockrank1_dale"
LAUNCH_ONE="${PROJECT_ROOT}/tools/launch_train_blockrank1_dale.sh"
CFG_121="${PROJECT_ROOT}/current_rnn/parameters_ei_mixed_blockrank1_dale_delay121.json"
CFG_131="${PROJECT_ROOT}/current_rnn/parameters_ei_mixed_blockrank1_dale_delay131.json"

STAMP="$(date +%Y%m%d_%H%M%S)"
"${LAUNCH_ONE}" "${CFG_121}" 0 "delay121_${STAMP}" "delay121_${STAMP}"
"${LAUNCH_ONE}" "${CFG_131}" 1 "delay131_${STAMP}" "delay131_${STAMP}"
