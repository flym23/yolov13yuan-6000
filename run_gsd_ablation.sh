#!/usr/bin/env bash
# GSD G1--G7 full-factorial chain: exactly three independent seed processes run concurrently per stage.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE_DIR="$ROOT/runs/gsd_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=${GSD_RUN_ID:-gsd_$(date -u +%Y%m%d_%H%M%S)}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
CURRENT_STAGE=initializing
COMPLETED=""

mkdir -p "$STATE_DIR" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "GSD chain already running for $RUN_ID" >&2; exit 73; }
export PIN_MEMORY=false
export WANDB_DISABLED=true

write_state() {
  GSD_STATE_FILE="$STATE_FILE" GSD_RUN_ID="$RUN_ID" GSD_STATUS="$1" GSD_STAGE="$CURRENT_STAGE" GSD_DETAIL="${2:-}" GSD_COMPLETED="$COMPLETED" GSD_PID="$$" "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["GSD_STATE_FILE"])
payload = {
    "run_id": os.environ["GSD_RUN_ID"], "status": os.environ["GSD_STATUS"], "stage": os.environ["GSD_STAGE"],
    "detail": os.environ["GSD_DETAIL"], "launcher_pid": int(os.environ["GSD_PID"]),
    "completed_stages": os.environ["GSD_COMPLETED"].split(), "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
    "settings": {"epochs": 300, "patience": 40, "workers": 2, "amp": False, "plots": False, "pin_memory": False, "deterministic": True},
    "g0_policy": "reuse designated L0 and P0 original-YOLOv13 summaries; do not retrain G0",
    "chain_policy": "unconditional G1-G7 sequential execution; no result threshold or performance gate",
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
  label="$1"; shift; failed=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
      for other in "$@"; do
        if [[ "$other" != "$pid" ]] && kill -0 "$other" 2>/dev/null; then kill "$other" 2>/dev/null || true; fi
      done
    fi
  done
  (( failed == 0 )) || { echo "GSD $CURRENT_STAGE $label failed" >&2; return 75; }
}

run_stage() {
  local stage seed name weights
  local -a train_pids=() test_pids=()
  stage="$1"; CURRENT_STAGE="$stage"; write_state training
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/train_gsd_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
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
  "$PY" "$ROOT/tools/collect_gsd_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $COMPLETED >"$STATE_DIR/$RUN_ID.$stage.collect.log" 2>&1
}

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
CURRENT_STAGE=preflight
write_state running "launcher_pid=$$"
"$PY" "$ROOT/tools/validate_gsd_models.py" --root "$ROOT" --device cpu --imgsz 128 >"$STATE_DIR/$RUN_ID.model_preflight.log" 2>&1
"$PY" "$ROOT/tools/validate_gsd_checkpoint.py" --root "$ROOT" --weights "$ROOT/yolov13n.pt" --output "$STATE_DIR/$RUN_ID.checkpoint_preflight.json" >"$STATE_DIR/$RUN_ID.checkpoint_preflight.log" 2>&1

for stage in g1 g2 g3 g4 g5 g6 g7; do
  run_stage "$stage"
done
CURRENT_STAGE=complete
write_state complete "g1_g7_finished=true; all_stages_started_without_result_gates=true"
echo "GSD G1--G7 chain completed: run_id=$RUN_ID"
