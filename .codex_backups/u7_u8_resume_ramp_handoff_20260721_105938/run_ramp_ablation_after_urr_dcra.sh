#!/usr/bin/env bash
# Wait for the active URR-DCRA chain, then hand off to the local RAMP chain.
set -Ee -o pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DEPENDENCY_SCRIPT=/home/room305/ZZF/yolov13-6000/run_urr_dcra_u2_u8_after_dgdr.sh
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
  [[ "$cmdline" == *"$DEPENDENCY_SCRIPT"* ]]
}

while dependency_alive; do
  echo "$(date -u +%FT%TZ) waiting for URR-DCRA chain to finish"
  sleep 60
done

status=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "unknown"))' "$DEPENDENCY_STATE" 2>/dev/null || echo unavailable)
echo "$(date -u +%FT%TZ) URR-DCRA launcher is absent (last state=$status); starting RAMP K1--K4"
exec "$ROOT/run_ramp_ablation.sh"
