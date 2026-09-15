#!/usr/bin/env bash
# Validate, archive, clean only old B3/B4 artifacts, then rerun repaired B3 and B4 with three concurrent seeds each.
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
RUN_ROOT="${ROOT}/runs/dcrq_b3b4_repaired_urpc2019_${RUN_ID}"
EXPECTED_LOG="${RUN_ROOT}/train/launcher.log"
AUDIT_ROOT="${ROOT}/runs/repair_audit_20260915"
[[ ${LAUNCHER_LOG} == "${EXPECTED_LOG}" ]] || { echo "launcher log must be ${EXPECTED_LOG}" >&2; exit 2; }
[[ -x ${PYTHON_BIN} && -f ${DATA} && -f ${ROOT}/yolov13n.pt && -f ${PARENT_RUN_ROOT}/state.json ]] || { echo "runtime, dataset, weight, or completed B2 parent is missing" >&2; exit 2; }
[[ ! -e ${RUN_ROOT}/state.json || $("${PYTHON_BIN}" -c "import json; print(json.load(open('${RUN_ROOT}/state.json', encoding='utf-8')).get('status', ''))") != running ]] || { echo "DCRQ repaired run already active" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/train" "${AUDIT_ROOT}"
echo "$(date -Is) DCRQ B3/B4 repair chain starts; old artifacts are untouched until all validations pass." >> "${LAUNCHER_LOG}"
"${PYTHON_BIN}" "${ROOT}/tools/generate_dcrq_repaired_yamls.py" --root "${ROOT}" --archive-dir "${AUDIT_ROOT}/yaml_before_repair"
"${PYTHON_BIN}" "${ROOT}/tools/run_dcrq_repaired_selfcheck.py" --root "${ROOT}"
"${PYTHON_BIN}" "${ROOT}/tools/validate_dcrq_b3b4_repaired.py" --root "${ROOT}" --pretrained "${ROOT}/yolov13n.pt" --report-dir "${RUN_ROOT}/preflight"
"${PYTHON_BIN}" "${ROOT}/tools/repair_dcrq_b3b4_cleanup.py" \
  --old-run-root "${PARENT_RUN_ROOT}" --coverage-report "${RUN_ROOT}/preflight/pretrained_coverage.json" \
  --old-yaml-manifest "${AUDIT_ROOT}/yaml_before_repair/old_yaml_manifest.json" \
  --manifest-out "${AUDIT_ROOT}/old_b3_b4_manifest.json"
DCRQ_B3B4_LAUNCHER_PID=$$ exec "${PYTHON_BIN}" "${ROOT}/tools/run_dcrq_b3b4_repaired.py" \
  --root "${ROOT}" --run-id "${RUN_ID}" --data "${DATA}" --parent-run-root "${PARENT_RUN_ROOT}" \
  --coverage-report "${RUN_ROOT}/preflight/pretrained_coverage.json"
