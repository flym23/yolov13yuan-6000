#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/room305/ZZF/yolov13yuan-6000
PYTHON=/home/room305/.conda/envs/yolov13/bin/python3
RUN_ROOT="${1:?absolute run-root is required}"
shift
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PIN_MEMORY=false WANDB_DISABLED=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PYTHON" -u "$ROOT/tools/run_ucra_v2.py" --run-root "$RUN_ROOT" "$@"
