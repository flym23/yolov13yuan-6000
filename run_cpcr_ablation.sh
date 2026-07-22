#!/usr/bin/env bash
# CPCR L1--L5 chain: train three seed processes per stage, with L5 launched directly after L4.
set -Eeo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE_DIR="$ROOT/runs/cpcr_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=${CPCR_RUN_ID:-cpcr_$(date -u +%Y%m%d_%H%M%S)}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
CURRENT_STAGE=initializing

mkdir -p "$STATE_DIR" "$TRAIN_DIR" "$TEST_DIR"
export PIN_MEMORY=false
export WANDB_DISABLED=true

state() {
  "$PY" "$ROOT/tools/cpcr_chain_state.py" --path "$STATE_FILE" --run-id "$RUN_ID" --status "$1" --stage "$CURRENT_STAGE" --detail "${2:-}"
}

on_error() {
  local code=$?
  state failed "exit_code=$code" || true
  exit "$code"
}
trap on_error ERR

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 1; }
echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
state running "launcher_pid=$$"

wait_group() {
  local label=$1
  shift
  local failed=0 pid other
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
      for other in "$@"; do
        if [[ "$other" != "$pid" ]] && kill -0 "$other" 2>/dev/null; then
          kill "$other" 2>/dev/null || true
        fi
      done
    fi
  done
  (( failed == 0 )) || { echo "CPCR $CURRENT_STAGE $label failed" >&2; return 1; }
}

run_stage() {
  local stage=$1 seed name weights
  local -a train_pids=() test_pids=()
  CURRENT_STAGE=$stage
  state training
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/train_cpcr_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 \
      >"$TRAIN_DIR/$name.log" 2>&1 &
    train_pids+=("$!")
  done
  printf '%s\n' "${train_pids[@]}" >"$STATE_DIR/$RUN_ID.$stage.train.pids"
  wait_group training "${train_pids[@]}"
  state testing
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    weights="$TRAIN_DIR/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "Missing checkpoint: $weights" >&2; return 1; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 \
      >"$TEST_DIR/$name.log" 2>&1 &
    test_pids+=("$!")
  done
  printf '%s\n' "${test_pids[@]}" >"$STATE_DIR/$RUN_ID.$stage.test.pids"
  wait_group testing "${test_pids[@]}"
}

CURRENT_STAGE=preflight
state preflight
"$PY" "$ROOT/tools/validate_cpcr_models.py" --root "$ROOT" --device cpu --imgsz 128 >"$STATE_DIR/$RUN_ID.model_preflight.log" 2>&1
"$PY" "$ROOT/tools/validate_cpcr_checkpoint.py" --root "$ROOT" --weights "$ROOT/yolov13n.pt" --output "$STATE_DIR/$RUN_ID.checkpoint_preflight.json" \
  >"$STATE_DIR/$RUN_ID.checkpoint_preflight.log" 2>&1

COMPLETED=()
for stage in l1 l2 l3 l4; do
  run_stage "$stage"
  COMPLETED+=("$stage")
  "$PY" "$ROOT/tools/collect_cpcr_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}" >"$STATE_DIR/$RUN_ID.$stage.collect.log" 2>&1
done

CURRENT_STAGE=l4_analysis
state evaluating_l4 "analysis_only"
"$PY" "$ROOT/tools/collect_cpcr_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}" >"$STATE_DIR/$RUN_ID.l4_gate_analysis.log" 2>&1

run_stage l5
COMPLETED+=(l5)
"$PY" "$ROOT/tools/collect_cpcr_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}" >"$STATE_DIR/$RUN_ID.l5.collect.log" 2>&1
CURRENT_STAGE=complete
state complete "l5_launched_unconditionally=true"
echo "CPCR L1--L5 chain completed: run_id=$RUN_ID"
