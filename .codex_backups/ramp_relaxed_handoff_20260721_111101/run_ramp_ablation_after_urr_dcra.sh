#!/usr/bin/env bash
# Start RAMP only after URR-DCRA has recorded a successful U8 test for all three seeds.
set -Ee -o pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DEPENDENCY_ROOT=/home/room305/ZZF/yolov13-6000
DEPENDENCY_STATE=/home/room305/ZZF/yolov13-6000/runs/urr_dcra_u2_u8/state.json
LOG_DIR="$ROOT/runs/ramp_ablation"

mkdir -p "$LOG_DIR"
dependency_alive() {
  [[ -f "$DEPENDENCY_STATE" ]] || return 1
  local pid cmdline
  pid=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("launcher_pid", ""))' "$DEPENDENCY_STATE" 2>/dev/null || true)
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$cmdline" == *"$DEPENDENCY_ROOT"/run_urr_dcra_* ]]
}

dependency_complete() {
  "$PY" - "$DEPENDENCY_STATE" "$DEPENDENCY_ROOT" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
root = Path(sys.argv[2])
try:
    state = json.loads(state_path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
run_id = state.get("run_id", "")
if not run_id or state.get("status") != "complete" or state.get("stage") != "complete" or state.get("exit_code") != 0:
    raise SystemExit(1)
for seed in range(3):
    test_dir = root / "runs" / "test" / f"{run_id}_u8_none_seed{seed}"
    if not (test_dir / "summary_metrics.json").is_file() or not (test_dir / "scale_ap_metrics.json").is_file():
        raise SystemExit(1)
PY
}

last_status="unavailable"
while ! dependency_complete; do
  if [[ -f "$DEPENDENCY_STATE" ]]; then
    last_status=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "unknown"))' "$DEPENDENCY_STATE" 2>/dev/null || echo unreadable)
  fi
  if dependency_alive; then
    echo "$(date -u +%FT%TZ) waiting for active URR-DCRA recovery (state=$last_status)"
  else
    echo "$(date -u +%FT%TZ) waiting for verified U7/U8 completion (last state=$last_status)"
  fi
  sleep 60
done

echo "$(date -u +%FT%TZ) verified U8 training and testing for all seeds; starting RAMP K1--K4"
exec "$ROOT/run_ramp_ablation.sh"
