#!/usr/bin/env bash
# Start once; wait for the completed CMRF chain, then hand over to the CARM A0--A7 chain.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
CMRF_STATE_FILE=${CMRF_STATE_FILE:-$ROOT/runs/cmrf_ablation/cmrf_20260729_123654.state.json}
WAIT_LOG="$ROOT/runs/carm_ablation/after_cmrf_wait.log"

mkdir -p "$(dirname "$WAIT_LOG")"
[[ -x "$PY" && -f "$CMRF_STATE_FILE" ]] || { echo "CMRF dependency or Python runtime is unavailable" >&2; exit 78; }
while true; do
  status=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$CMRF_STATE_FILE")
  case "$status" in
    complete)
      echo "CMRF complete; starting CARM." | tee -a "$WAIT_LOG"
      exec bash "$ROOT/run_carm_ablation.sh" "$CMRF_STATE_FILE"
      ;;
    failed)
      echo "CMRF failed; CARM will not start." | tee -a "$WAIT_LOG" >&2
      exit 75
      ;;
    *)
      echo "$(date -u +%FT%TZ) waiting for CMRF: $status" >>"$WAIT_LOG"
      sleep 120
      ;;
  esac
done
