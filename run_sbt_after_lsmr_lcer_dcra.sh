#!/usr/bin/env bash
# Start SBT after the sibling LSMR-LCER-DCRA chain has no active launcher, train, or test process.
set -Eeuo pipefail

SBT_ROOT=$(cd "$(dirname "$0")" && pwd)
LSMR_ROOT=/home/room305/ZZF/yolov13-6000
LSMR_STATE="$LSMR_ROOT/runs/lsmr_lcer_dcra_r1_r3/state.json"
PY=/home/room305/.conda/envs/yolov13/bin/python
STATE_DIR="$SBT_ROOT/runs/sbt_ablation"
LOCK="$STATE_DIR/.wait_for_lsmr_lcer_dcra.lock"
PID=$$

mkdir -p "$STATE_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "SBT LSMR waiter is already running" >&2; exit 73; }

write_wait_state() {
  SBT_WAIT_STATE="$STATE_DIR" SBT_WAIT_STATUS="$1" SBT_WAIT_PID="$PID" SBT_WAIT_LSMR_STATE="$LSMR_STATE" \
    "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["SBT_WAIT_STATE"]) / "wait_for_lsmr_lcer_dcra.json"
payload = {
    "status": os.environ["SBT_WAIT_STATUS"], "launcher_pid": int(os.environ["SBT_WAIT_PID"]),
    "lsmr_state": os.environ["SBT_WAIT_LSMR_STATE"],
    "policy": "start SBT T1-T7 after no active LSMR launcher/train/test process remains; T4-T7 have no result gate",
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
  lsmr_status=state_unavailable
  if [[ -f "$LSMR_STATE" ]]; then
    lsmr_status=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "unknown"))' "$LSMR_STATE" 2>/dev/null || echo state_unreadable)
  fi
  if pgrep -f "[r]un_lsmr_lcer_dcra_r1_r3.sh|[t]rain_lsmr_lcer_dcra_worker.py|[t]est_lsmr_lcer_dcra.py" >/dev/null; then
    write_wait_state "waiting_lsmr_${lsmr_status}"
    sleep 60
    continue
  fi
  write_wait_state "lsmr_inactive_${lsmr_status}_starting_sbt"
  rmdir "$LOCK" 2>/dev/null || true
  exec /usr/bin/env bash "$SBT_ROOT/run_sbt_ablation.sh"
done
