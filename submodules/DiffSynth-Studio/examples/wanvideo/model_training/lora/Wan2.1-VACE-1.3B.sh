#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

NUM_GPUS="${NUM_GPUS:-1}"
WANDB_NAME="${WANDB_NAME:-wan_vace_ref_keyframes}"
DATASET="${DATASET:-kubric}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-/path/to/Wan2.1-VACE-1.3B}"
MODEL_PATHS="["$WAN_MODEL_DIR/diffusion_pytorch_model.safetensors","$WAN_MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth","$WAN_MODEL_DIR/Wan2.1_VAE.pth"]"
TOKENIZER_PATH="$WAN_MODEL_DIR/google/umt5-xxl"

LORA_ARGS=()
if [ -n "${LORA_CKPT:-}" ]; then
  LORA_ARGS=(--lora_checkpoint "$LORA_CKPT")
fi

cd "${REPO_ROOT}"

accelerate launch --num_processes "$NUM_GPUS" \
  examples/wanvideo/model_training/train_vace.py \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --learning_rate 1e-4 \
  --num_epochs 15 \
  --dataset_num_workers 8 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./outputs/Wan2.1-VACE-1.3B_lora" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 128 \
  --dataset "$DATASET" \
  "${LORA_ARGS[@]}" \
  --wandb_name "$WANDB_NAME"
