#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUN_ROOT="${1:?absolute run root required}"
LAUNCHER_LOG="${2:?absolute launcher log path required}"
[[ "${RUN_ROOT}" == /* && "${LAUNCHER_LOG}" == /* ]]
[[ "${RUN_ROOT}" == "${PROJECT_ROOT}/runs/gbc_qprr_urpc2020_20260928_r1" ]]

printf '%s WAIT_SCRIPT_START run_root=%s log=%s pid=%s\n' "$(date --iso-8601=seconds)" "${RUN_ROOT}" "${LAUNCHER_LOG}" "$$"
printf '%s\n' "$$" > "${RUN_ROOT}/launcher.pid"
exec python3 "${PROJECT_ROOT}/tools/run_gbc_qprr_urpc2020.py" --run-root "${RUN_ROOT}" --wait
