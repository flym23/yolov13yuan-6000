#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/room305/ZZF/yolov13yuan-6000
PYTHON=/home/room305/.conda/envs/yolov13/bin/python3
RUN_ROOT="${1:?absolute run-root is required}"
LAUNCHER_LOG="${2:?absolute launcher log is required}"
shift 2

[[ "$RUN_ROOT" = "$ROOT"/runs/sqanwd_urpc2020_* && "$LAUNCHER_LOG" = "$RUN_ROOT/train/launcher.log" ]] || exit 2
[[ -f "$RUN_ROOT/train/preflight.json" ]] || exit 3

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PIN_MEMORY=false WANDB_DISABLED=true CUBLAS_WORKSPACE_CONFIG=:4096:8

exec "$PYTHON" -u "$ROOT/tools/run_sqanwd_urpc2020.py" --run-root "$RUN_ROOT" "$@" >> "$LAUNCHER_LOG" 2>&1
