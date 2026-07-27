#!/usr/bin/env bash
# Start GSD as soon as the sibling DGMR-LCER-DCRA launcher and its workers have exited.
set -Eeuo pipefail

GSD_ROOT=$(cd "$(dirname "$0")" && pwd)
DGMR_ROOT=/home/room305/ZZF/yolov13-6000
PY=/home/room305/.conda/envs/yolov13/bin/python
STATE_DIR="$GSD_ROOT/runs/gsd_ablation"
LOCK="$STATE_DIR/.wait_for_dgmr_lcer_dcra.lock"
PID=$$

mkdir -p "$STATE_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "GSD DGMR waiter is already running" >&2; exit 73; }

write_wait_state() {
  GSD_WAIT_DIR="$STATE_DIR" GSD_WAIT_STATUS="$1" GSD_WAIT_PID="$PID" GSD_DGMR_ROOT="$DGMR_ROOT" "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["GSD_WAIT_DIR"]) / "wait_for_dgmr_lcer_dcra.json"
payload = {
    "status": os.environ["GSD_WAIT_STATUS"], "launcher_pid": int(os.environ["GSD_WAIT_PID"]),
    "dgmr_root": os.environ["GSD_DGMR_ROOT"],
    "policy": "start GSD G1-G7 after no active DGMR launcher/train/test process remains; no score gate",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temp.replace(path)
'
}

cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT
trap 'write_wait_state interrupted; exit 130' INT
trap 'write_wait_state interrupted; exit 143' TERM
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }

while true; do
  if pgrep -f "[r]un_dgmr_lcer_dcra.*\.sh|[t]rain_dgmr_lcer_dcra_worker.py|[t]est_dgmr_lcer_dcra.py" >/dev/null; then
    write_wait_state waiting_dgmr_running
    sleep 60
    continue
  fi
  write_wait_state dgmr_inactive_starting_gsd
  rmdir "$LOCK" 2>/dev/null || true
  exec /usr/bin/env bash "$GSD_ROOT/run_gsd_ablation.sh"
done
