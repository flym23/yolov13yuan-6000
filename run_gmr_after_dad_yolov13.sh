#!/usr/bin/env bash
# Start GMR after the sibling DAD-YOLOv13 launcher and all of its workers have exited.
set -Eeuo pipefail

GMR_ROOT=$(cd "$(dirname "$0")" && pwd)
DAD_ROOT=/home/room305/ZZF/yolov13-6000
PY=/home/room305/.conda/envs/yolov13/bin/python
STATE_DIR="$GMR_ROOT/runs/gmr_ablation"
LOCK="$STATE_DIR/.wait_for_dad_yolov13.lock"
PID=$$

mkdir -p "$STATE_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "GMR DAD waiter is already running" >&2; exit 73; }

write_wait_state() {
  GMR_WAIT_DIR="$STATE_DIR" GMR_WAIT_STATUS="$1" GMR_WAIT_PID="$PID" GMR_DAD_ROOT="$DAD_ROOT" "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["GMR_WAIT_DIR"]) / "wait_for_dad_yolov13.json"
payload = {
    "status": os.environ["GMR_WAIT_STATUS"], "launcher_pid": int(os.environ["GMR_WAIT_PID"]),
    "dad_root": os.environ["GMR_DAD_ROOT"],
    "policy": "start GMR after no active DAD launcher/train/test process remains; no score gate",
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
  if pgrep -f "[r]un_dad_yolov13.*\\.sh|[t]rain_dad_yolov13_worker.py|[t]est_dad_yolov13.*\\.py" >/dev/null; then
    write_wait_state waiting_dad_running
    sleep 60
    continue
  fi
  write_wait_state dad_inactive_starting_gmr
  rmdir "$LOCK" 2>/dev/null || true
  exec /usr/bin/env bash "$GMR_ROOT/run_gmr_ablation.sh"
done
