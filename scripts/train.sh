#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-l40s_anysplat}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

cd "${REPO_ROOT}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m src.main +training=c4g "$@"
