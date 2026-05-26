#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -u "${SCRIPT_DIR}/evaluation_davis.py" \
  --precomputed_root "${PRECOMPUTED_ROOT:-/path/to/c4g_precomputed_davis}" \
  --fps "${FPS:-5}" \
  --save_local_outputs \
  --diffsynth_path "${REPO_ROOT}" \
  --lora_path "${LORA_PATH:-/path/to/lora_checkpoint.safetensors}" \
  --lora_rank "${LORA_RANK:-128}" \
  --ref_pos_offset "${ROPE_OFFSET:-500}" \
  --wan_num_steps "${WAN_NUM_STEPS:-50}" \
  --wan_cfg_scale "${WAN_CFG_SCALE:-5.0}" \
  --device "${DEVICE:-cuda}" \
  --output_root "${OUTPUT_ROOT:-./outputs/davis_eval}" \
  --wandb_project "${WANDB_PROJECT:-wan_vace_eval}" \
  --wandb_run_name "${WANDB_RUN_NAME:-davis_wan_vace}"
