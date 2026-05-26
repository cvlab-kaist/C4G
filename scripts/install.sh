#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./submodules/diff_gaussian_rasterization_w_pose
