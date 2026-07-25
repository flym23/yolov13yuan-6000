#!/usr/bin/env bash
# SBT T1--T7 full-factorial chain: three independent seed processes run concurrently for every stage.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE_DIR="$ROOT/runs/sbt_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=${SBT_RUN_ID:-sbt_$(date -u +%Y%m%d_%H%M%S)}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
CURRENT_STAGE=initializing
COMPLETED=""

mkdir -p "$STATE_DIR" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "SBT chain already running for $RUN_ID" >&2; exit 73; }
export PIN_MEMORY=false
export WANDB_DISABLED=true

write_state() {
  SBT_STATE_FILE="$STATE_FILE" SBT_RUN_ID="$RUN_ID" SBT_STATUS="$1" SBT_STAGE="$CURRENT_STAGE" SBT_DETAIL="${2:-}" SBT_COMPLETED="$COMPLETED" SBT_PID="$$" \
    "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["SBT_STATE_FILE"])
payload = {
    "run_id": os.environ["SBT_RUN_ID"], "status": os.environ["SBT_STATUS"], "stage": os.environ["SBT_STAGE"],
    "detail": os.environ["SBT_DETAIL"], "launcher_pid": int(os.environ["SBT_PID"]),
    "completed_stages": os.environ["SBT_COMPLETED"].split(),
    "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
    "settings": {"epochs": 300, "patience": 40, "workers": 2, "amp": False, "plots": False, "pin_memory": False, "deterministic": True},
    "t0_policy": "reuse designated L0 and P0 original-YOLOv13 summaries; do not retrain T0",
    "t4_t7_policy": "unconditional sequential execution; no result gate or launch threshold",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temp.replace(path)
'
}

cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
on_error() { code=$?; write_state failed "exit_code=$code" || true; exit "$code"; }
trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_group() {
  local label failed pid other
  label="$1"
  shift
  failed=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
      for other in "$@"; do
        if [[ "$other" != "$pid" ]] && kill -0 "$other" 2>/dev/null; then kill "$other" 2>/dev/null || true; fi
      done
    fi
  done
  (( failed == 0 )) || { echo "SBT $CURRENT_STAGE $label failed" >&2; return 75; }
}

run_stage() {
  local stage seed name weights
  local -a train_pids=() test_pids=()
  stage="$1"
  CURRENT_STAGE="$stage"
  write_state training
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/train_sbt_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
    train_pids+=("$!")
  done
  printf '%s\n' "${train_pids[@]}" >"$STATE_DIR/$RUN_ID.$stage.train.pids"
  wait_group training "${train_pids[@]}"
  write_state testing
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    weights="$TRAIN_DIR/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "Missing checkpoint: $weights" >&2; return 76; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 >"$TEST_DIR/$name.log" 2>&1 &
    test_pids+=("$!")
  done
  printf '%s\n' "${test_pids[@]}" >"$STATE_DIR/$RUN_ID.$stage.test.pids"
  wait_group testing "${test_pids[@]}"
  COMPLETED="$COMPLETED $stage"
  "$PY" "$ROOT/tools/collect_sbt_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $COMPLETED >"$STATE_DIR/$RUN_ID.$stage.collect.log" 2>&1
}

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
CURRENT_STAGE=preflight
write_state running "launcher_pid=$$"
"$PY" "$ROOT/tools/validate_sbt_models.py" --root "$ROOT" --device cpu --imgsz 128 >"$STATE_DIR/$RUN_ID.model_preflight.log" 2>&1
"$PY" "$ROOT/tools/validate_sbt_checkpoint.py" --root "$ROOT" --weights "$ROOT/yolov13n.pt" --output "$STATE_DIR/$RUN_ID.checkpoint_preflight.json" >"$STATE_DIR/$RUN_ID.checkpoint_preflight.log" 2>&1

for stage in t1 t2 t3 t4 t5 t6 t7; do
  run_stage "$stage"
done
CURRENT_STAGE=complete
write_state complete "t1_t7_finished=true; t4_t7_started_without_result_gates=true"
echo "SBT T1--T7 chain completed: run_id=$RUN_ID"
