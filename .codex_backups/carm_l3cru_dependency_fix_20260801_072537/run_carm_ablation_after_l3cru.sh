#!/usr/bin/env bash
# Wait for the active L3-CRU chain, then hand over to the CARM A0--A7 chain.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
L3CRU_STATE_FILE=${1:-/home/room305/ZZF/yolov13-6000/runs/train/l3cru_20260801_025537.state.json}
WAIT_LOG="$ROOT/runs/carm_ablation/after_l3cru_wait.log"

mkdir -p "$(dirname "$WAIT_LOG")"
[[ -x "$PY" && -f "$L3CRU_STATE_FILE" ]] || { echo "L3-CRU dependency or Python runtime is unavailable" >&2; exit 78; }
while true; do
  status=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$L3CRU_STATE_FILE")
  case "$status" in
    complete)
      echo "$(date -u +%FT%TZ) L3-CRU complete; starting CARM." >>"$WAIT_LOG"
      exec bash "$ROOT/run_carm_ablation.sh" "$L3CRU_STATE_FILE"
      ;;
    failed)
      echo "$(date -u +%FT%TZ) L3-CRU failed; CARM will not start." >>"$WAIT_LOG"
      exit 75
      ;;
    *)
      echo "$(date -u +%FT%TZ) waiting for L3-CRU: $status" >>"$WAIT_LOG"
      sleep 120
      ;;
  esac
done
