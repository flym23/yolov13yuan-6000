#!/usr/bin/env bash
# Launch this once: it waits for the currently active TDR chain, then execs the CMRF C0--C5 chain.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
# The completed TDR chain and authorized historical baseline summaries remain in the original project.
TDR_STATE_FILE=${TDR_STATE_FILE:-/home/room305/ZZF/yolov13-6000/runs/tdr_yolov13_t1_t7/tdr_yolov13_20260728_085954.state.json}
WAIT_LOG="$ROOT/runs/cmrf_ablation/after_tdr_wait.log"

mkdir -p "$(dirname "$WAIT_LOG")"
[[ -x "$PY" && -f "$TDR_STATE_FILE" ]] || { echo "TDR dependency or Python runtime is unavailable" >&2; exit 78; }
while true; do
  status=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$TDR_STATE_FILE")
  case "$status" in
    complete) echo "TDR complete; starting CMRF." | tee -a "$WAIT_LOG"; exec bash "$ROOT/run_cmrf_ablation.sh" "$TDR_STATE_FILE" ;;
    failed) echo "TDR failed; CMRF will not start." | tee -a "$WAIT_LOG" >&2; exit 75 ;;
    *) echo "$(date -u +%FT%TZ) waiting for TDR: $status" >>"$WAIT_LOG"; sleep 120 ;;
  esac
done
