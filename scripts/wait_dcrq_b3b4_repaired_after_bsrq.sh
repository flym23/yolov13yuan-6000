#!/usr/bin/env bash
# Wait for the active BSRQ chain, force any skipped S4--S7 stages, then launch repaired DCRQ B3/B4.
set -Eeuo pipefail

ROOT=${1:?absolute project root is required}
LAUNCHER_LOG=${2:?absolute launcher log is required}
RUN_ID=${3:?immutable DCRQ run id is required}
BSRQ_RUN_ID=r1_20260915_130524
BSRQ_RUN_ROOT=/home/room305/ZZF/yolov13yuan-6000/runs/bsrq_urpc2019_20260915_${BSRQ_RUN_ID}
PARENT_STATE=${BSRQ_RUN_ROOT}/state.json
DCRQ_RUN_ROOT=/home/room305/ZZF/yolov13yuan-6000/runs/dcrq_b3b4_repaired_urpc2019_${RUN_ID}
PYTHON_BIN=${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python3}

[[ ${ROOT} = /* && ${LAUNCHER_LOG} = /* ]] || { echo "ROOT and LAUNCHER_LOG must be absolute" >&2; exit 2; }
[[ ${RUN_ID} =~ ^[A-Za-z0-9][A-Za-z0-9_-]{2,80}$ ]] || { echo "invalid immutable RUN_ID" >&2; exit 2; }
ROOT=$(cd "${ROOT}" && pwd -P)
EXPECTED_LOG="${DCRQ_RUN_ROOT}/train/wait_launcher.log"
[[ ${LAUNCHER_LOG} == "${EXPECTED_LOG}" ]] || { echo "wait launcher log must be ${EXPECTED_LOG}" >&2; exit 2; }
[[ -f ${PARENT_STATE} && -x ${PYTHON_BIN} ]] || { echo "active BSRQ state or runtime is missing" >&2; exit 2; }
mkdir -p "${DCRQ_RUN_ROOT}/train"

write_state() {
  "${PYTHON_BIN}" - "${DCRQ_RUN_ROOT}/state.json" "${1}" "${2}" "${3:-}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, status, detail, reason = map(str, sys.argv[1:])
payload = {"run_id": "r2_20260915_b3b4_repaired", "status": status, "upstream_state_path": "/home/room305/ZZF/yolov13yuan-6000/runs/bsrq_urpc2019_20260915_r1_20260915_130524/state.json", "detail": detail, "failure_reason": reason, "updated_at": datetime.now(timezone.utc).isoformat()}
target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + ".tmp"); temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); temporary.replace(target)
PY
}

write_state waiting "Waiting for BSRQ terminal state before DCRQ repaired B3/B4 launch."
echo "$(date -Is) waiting for ${PARENT_STATE}" >> "${LAUNCHER_LOG}"
while true; do
  if ! readout=$("${PYTHON_BIN}" - "${PARENT_STATE}" 2>/dev/null <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data.get("status", "")); print(data.get("failure_reason", ""))
PY
); then
    sleep 30
    continue
  fi
  status=$(printf '%s\n' "${readout}" | sed -n '1p')
  reason=$(printf '%s\n' "${readout}" | sed -n '2p')
  if [[ ${status} == completed || ${status} == failed || ${status} == cancelled ]]; then
    break
  fi
  sleep 30
done
echo "$(date -Is) observed BSRQ terminal status=${status}; reason=${reason}" >> "${LAUNCHER_LOG}"
write_state running "Observed BSRQ terminal state; preparing unconditional S4-S7 continuation and DCRQ rerun." "${reason}"
if [[ ${status} == completed ]]; then
  if ! "${PYTHON_BIN}" "${ROOT}/tools/run_bsrq_ablation.py" --root "${ROOT}" --run-id "${BSRQ_RUN_ID}" \
      --data /home/room305/ZZF/URPC2019/data.yaml --parent-run-root /home/room305/ZZF/yolov13yuan-6000/runs/dcrq_urpc2019_20260914_r1_20260914_0415 \
      --force-all-phase2 >> "${BSRQ_RUN_ROOT}/train/launcher.log" 2>&1; then
    echo "$(date -Is) forced BSRQ S4-S7 continuation failed; DCRQ still starts after its terminal state." >> "${LAUNCHER_LOG}"
  fi
fi
exec bash "${ROOT}/scripts/run_dcrq_b3b4_repaired.sh" "${ROOT}" "${DCRQ_RUN_ROOT}/train/launcher.log" "${RUN_ID}"
