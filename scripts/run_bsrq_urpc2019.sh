#!/usr/bin/env bash
# Direct, resumable BSRQ chain. Its completed DCRQ B1 parent is reused as S0.
set -Eeuo pipefail

ROOT=${1:?absolute project root is required}
LAUNCHER_LOG=${2:?absolute launcher log is required}
RUN_ID=${3:?immutable run id is required}
DATA=/home/room305/ZZF/URPC2019/data.yaml
PARENT_RUN_ROOT=/home/room305/ZZF/yolov13yuan-6000/runs/dcrq_urpc2019_20260914_r1_20260914_0415
PYTHON_BIN=${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python3}

[[ ${ROOT} = /* && ${LAUNCHER_LOG} = /* ]] || { echo "ROOT and LAUNCHER_LOG must be absolute" >&2; exit 2; }
[[ ${RUN_ID} =~ ^[A-Za-z0-9][A-Za-z0-9_-]{2,80}$ ]] || { echo "invalid immutable RUN_ID" >&2; exit 2; }
ROOT=$(cd "${ROOT}" && pwd -P)
RUN_ROOT="${ROOT}/runs/bsrq_urpc2019_20260915_${RUN_ID}"
EXPECTED_LOG="${RUN_ROOT}/train/launcher.log"
[[ ${LAUNCHER_LOG} == "${EXPECTED_LOG}" ]] || { echo "launcher log must be ${EXPECTED_LOG}" >&2; exit 2; }
[[ -x ${PYTHON_BIN} ]] || { echo "Python runtime missing: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f ${DATA} && -f ${ROOT}/yolov13n.pt && -d ${PARENT_RUN_ROOT} ]] || { echo "dataset, yolov13n.pt, or completed B1 parent is missing" >&2; exit 2; }
[[ ! -e ${RUN_ROOT}/state.json || $("${PYTHON_BIN}" -c "import json; print(json.load(open('${RUN_ROOT}/state.json', encoding='utf-8')).get('status', ''))") != running ]] || { echo "BSRQ run already active: ${RUN_ROOT}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/train"
echo "$(date -Is) BSRQ direct chain starts; parent B1_D is completed and reused as S0." >> "${LAUNCHER_LOG}"
"${PYTHON_BIN}" "${ROOT}/tools/generate_bsrq_yamls.py" --root "${ROOT}"
"${PYTHON_BIN}" "${ROOT}/tools/run_bsrq_selfcheck.py" --root "${ROOT}"
"${PYTHON_BIN}" "${ROOT}/tools/validate_bsrq_models.py" --root "${ROOT}" --pretrained "${ROOT}/yolov13n.pt" --report-dir "${RUN_ROOT}/preflight"
BSRQ_LAUNCHER_PID=$$ exec "${PYTHON_BIN}" "${ROOT}/tools/run_bsrq_ablation.py" \
  --root "${ROOT}" --run-id "${RUN_ID}" --data "${DATA}" --parent-run-root "${PARENT_RUN_ROOT}" --force-all-phase2
