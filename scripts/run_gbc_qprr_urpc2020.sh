#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON=/home/room305/.conda/envs/yolov13/bin/python3
RUN_ROOT="${1:?absolute run root required}"
LAUNCHER_LOG="${2:?absolute launcher log path required}"
shift 2
[[ "${RUN_ROOT}" == /* && "${LAUNCHER_LOG}" == /* ]]
[[ "${RUN_ROOT}" == "${PROJECT_ROOT}/runs/gbc_qprr_urpc2020_20260928_r1" ]]
[[ -x "${PYTHON}" ]]

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PIN_MEMORY=false WANDB_DISABLED=true CUBLAS_WORKSPACE_CONFIG=:4096:8

printf '%s RUN_SCRIPT_START run_root=%s log=%s pid=%s\n' "$(date --iso-8601=seconds)" "${RUN_ROOT}" "${LAUNCHER_LOG}" "$$"
printf '%s\n' "$$" > "${RUN_ROOT}/launcher.pid"
exec "${PYTHON}" -u "${PROJECT_ROOT}/tools/run_gbc_qprr_urpc2020.py" --run-root "${RUN_ROOT}" "$@"
