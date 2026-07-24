#!/usr/bin/env bash
# CBER N3--N6 chain. Each stage trains and tests seed 0/1/2 concurrently on GPU 0.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE="$ROOT/runs/cber_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RESUME=false
COMPLETED=""

case $# in
  0) RUN_ID="cber_$(date -u +%Y%m%d_%H%M%S)" ;;
  2)
    [[ "$1" == "--resume" && "$2" =~ ^cber_[0-9]{8}_[0-9]{6}$ ]] || { echo "usage: $0 [--resume <cber-run-id>]" >&2; exit 64; }
    RUN_ID="$2"
    RESUME=true
    COMPLETED="n3"
    ;;
  *) echo "usage: $0 [--resume <cber-run-id>]" >&2; exit 64 ;;
esac

LOCK="$STATE/.${RUN_ID}.chain_lock"
PID=$$
CURRENT=initializing

mkdir -p "$STATE" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "CBER chain already running for $RUN_ID" >&2; exit 73; }

write_state() {
  CBER_RUN_ID="$RUN_ID" CBER_STATUS="$1" CBER_STAGE="$2" CBER_CODE="$3" CBER_PID="$PID" CBER_STATE="$STATE" CBER_DATA="$DATA" CBER_DONE="$COMPLETED" CBER_RESUME="$RESUME" \
    "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["CBER_STATE"]) / "state.json"
payload = {
    "run_id": os.environ["CBER_RUN_ID"], "status": os.environ["CBER_STATUS"], "stage": os.environ["CBER_STAGE"],
    "exit_code": int(os.environ["CBER_CODE"]), "launcher_pid": int(os.environ["CBER_PID"]),
    "dataset": os.environ["CBER_DATA"], "epochs": 300, "patience": 40, "amp": False, "workers": 2,
    "plots": False, "pin_memory": False, "stage_order": ["n3", "n4", "n5", "n6"],
    "completed_stages": os.environ["CBER_DONE"].split(), "resume": os.environ["CBER_RESUME"] == "true",
    "n5_n6_policy": "always start after N4; the documented N4 gate is recorded for analysis only",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temp.replace(path)
'
}

cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
on_error() { code=$?; write_state failed "$CURRENT" "$code" || true; exit "$code"; }
trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

collect_completed() {
  "$PY" "$ROOT/tools/collect_cber_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $COMPLETED
}

run_stage() {
  local stage failed pids seed name weights child
  stage="$1"
  failed=0
  pids=""
  CURRENT="$stage"
  write_state running "$CURRENT" 0
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/train_cber_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
    pids="$pids $!"
    echo "$!" >"$STATE/${stage}_seed${seed}.train.pid"
  done
  for child in $pids; do wait "$child" || failed=1; done
  ((failed == 0)) || return 75

  failed=0
  pids=""
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    weights="$ROOT/runs/train/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "missing checkpoint: $weights" >&2; return 76; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 >"$TEST_DIR/$name.log" 2>&1 &
    pids="$pids $!"
    echo "$!" >"$STATE/${stage}_seed${seed}.test.pid"
  done
  for child in $pids; do wait "$child" || failed=1; done
  ((failed == 0)) || return 77

  COMPLETED="$COMPLETED $stage"
  collect_completed >"$STATE/${RUN_ID}_${stage}.collect.log" 2>&1
}

export PIN_MEMORY=false
echo "$PID" >"$STATE/${RUN_ID}.launcher.pid"
write_state initializing "$CURRENT" 0
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }

if [[ "$RESUME" == true ]]; then
  for seed in 0 1 2; do
    [[ -f "$ROOT/runs/test/${RUN_ID}_n3_seed${seed}/summary_metrics.json" ]] || { echo "N3 test summary is missing for seed $seed" >&2; exit 80; }
  done
fi

"$PY" "$ROOT/tools/generate_cber_yamls.py" >"$STATE/${RUN_ID}.yaml.log" 2>&1
"$PY" "$ROOT/tools/validate_cber_models.py" --root "$ROOT" --device cpu --imgsz 128 >"$STATE/${RUN_ID}.model_preflight.log" 2>&1
"$PY" "$ROOT/tools/validate_cber_checkpoint.py" --root "$ROOT" \
  --weights "$ROOT/runs/train/scpg_20260720_055026_h3_seed0/weights/best.pt" \
  --output "$STATE/${RUN_ID}.checkpoint_preflight.json" >"$STATE/${RUN_ID}.checkpoint_preflight.log" 2>&1

if [[ "$RESUME" == false ]]; then
  run_stage "n3"
fi
run_stage "n4"
"$PY" "$ROOT/tools/collect_cber_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $COMPLETED --check-n4 >"$STATE/${RUN_ID}.n4_gate.log" 2>&1
write_state n4_gate_recorded_continue n4 0
run_stage "n5"
run_stage "n6"
CURRENT=complete
write_state complete "$CURRENT" 0
echo "CBER N3--N6 chain completed: run_id=$RUN_ID"
