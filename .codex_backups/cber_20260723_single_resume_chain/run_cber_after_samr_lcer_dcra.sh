#!/usr/bin/env bash
# Wait for the active SAMR-LCER-DCRA chain to release GPU 0, then resume CBER from completed N3.
set -Eeuo pipefail

CBER_ROOT=$(cd "$(dirname "$0")" && pwd)
SAMR_ROOT=/home/room305/ZZF/yolov13-6000
SAMR_STATE="$SAMR_ROOT/runs/samr_lcer_dcra_s1_s3/state.json"
PY=/home/room305/.conda/envs/yolov13/bin/python
STATE="$CBER_ROOT/runs/cber_ablation"
RUN_ID=cber_20260723_104127
LOCK="$STATE/.wait_for_samr_lcer_dcra.lock"
PID=$$

mkdir -p "$STATE"
mkdir "$LOCK" 2>/dev/null || { echo "CBER SAMR waiter is already running" >&2; exit 73; }

write_wait_state() {
  CBER_WAIT_STATE="$STATE" CBER_WAIT_STATUS="$1" CBER_WAIT_PID="$PID" CBER_WAIT_RUN_ID="$RUN_ID" CBER_WAIT_SAMR_STATE="$SAMR_STATE" \
    "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["CBER_WAIT_STATE"]) / "wait_for_samr_lcer_dcra.json"
payload = {
    "status": os.environ["CBER_WAIT_STATUS"],
    "launcher_pid": int(os.environ["CBER_WAIT_PID"]),
    "cber_run_id": os.environ["CBER_WAIT_RUN_ID"],
    "samr_state": os.environ["CBER_WAIT_SAMR_STATE"],
    "policy": "start CBER N4-N6 as soon as no SAMR-LCER-DCRA launcher, train, or test process remains; terminal state is recorded when available",
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
[[ -f "$SAMR_STATE" ]] || { echo "SAMR state is unavailable: $SAMR_STATE" >&2; exit 79; }
for seed in 0 1 2; do
  [[ -f "$CBER_ROOT/runs/test/${RUN_ID}_n3_seed${seed}/summary_metrics.json" ]] || { echo "N3 summary missing for seed $seed" >&2; exit 80; }
done

while true; do
  samr_status=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", "unknown"))' "$SAMR_STATE")
  if pgrep -f "[r]un_samr_lcer_dcra_s1_s3_after_cber.sh|[t]rain_samr_lcer_dcra_worker.py|[t]est.py.*samr_lcer_dcra" >/dev/null; then
    write_wait_state "waiting_samr_${samr_status}"
    sleep 60
    continue
  fi
  write_wait_state "samr_inactive_${samr_status}_starting_cber"
  rmdir "$LOCK" 2>/dev/null || true
  exec /usr/bin/env bash "$CBER_ROOT/run_cber_after_n3.sh" "$RUN_ID"
done
