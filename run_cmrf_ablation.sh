#!/usr/bin/env bash
# CMRF C0--C5: exactly three deterministic seed processes per stage, with an existing TDR chain as a hard dependency.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
BASELINE_ROOT=${BASELINE_ROOT:-/home/room305/ZZF/yolov13-6000}
STATE_DIR="$ROOT/runs/cmrf_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=${CMRF_RUN_ID:-cmrf_$(date -u +%Y%m%d_%H%M%S)}
TDR_STATE_FILE=${1:?usage: run_cmrf_ablation.sh /absolute/path/to/completed-tdr-state.json}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
STAGES=(c0 c1 c2 c3 c4 c5)
COMPLETED=()
CURRENT_STAGE=initializing
L0_REFERENCE="$BASELINE_ROOT/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json"
P0_REFERENCE="$BASELINE_ROOT/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json"

mkdir -p "$STATE_DIR" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "CMRF chain already running for $RUN_ID" >&2; exit 73; }
export WANDB_DISABLED=true PIN_MEMORY=false

write_state() {
  "$PY" "$ROOT/tools/cmrf_chain_state.py" --path "$STATE_FILE" --run-id "$RUN_ID" --status "$1" --stage "$CURRENT_STAGE" \
    --detail "${2:-}" --launcher-pid "$$" --completed "${COMPLETED[@]}" --tdr-state "$TDR_STATE_FILE"
}

cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
on_error() { code=$?; write_state failed "exit_code=$code" || true; exit "$code"; }
trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_group() {
  local label="$1" failed=0 pid other
  shift
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
      for other in "$@"; do
        if [[ "$other" != "$pid" ]] && kill -0 "$other" 2>/dev/null; then kill "$other" 2>/dev/null || true; fi
      done
    fi
  done
  (( failed == 0 )) || { echo "CMRF $CURRENT_STAGE $label failed" >&2; return 75; }
}

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" && -f "$ROOT/yolov13n.pt" && -f "$L0_REFERENCE" && -f "$P0_REFERENCE" ]] || { echo "CMRF input file missing" >&2; exit 79; }
[[ -f "$TDR_STATE_FILE" ]] || { echo "TDR state unavailable: $TDR_STATE_FILE" >&2; exit 80; }
TDR_STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$TDR_STATE_FILE")
[[ "$TDR_STATUS" == "complete" ]] || { echo "TDR has not completed: status=$TDR_STATUS" >&2; exit 81; }

echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
CURRENT_STAGE=preflight
write_state running "tdr_complete=true"
"$PY" -m py_compile "$ROOT/ultralytics/nn/modules/block.py" "$ROOT/ultralytics/nn/modules/__init__.py" \
  "$ROOT/ultralytics/nn/tasks.py" "$ROOT/tools/train_cmrf_worker.py" "$ROOT/tools/collect_cmrf_ablation.py" \
  "$ROOT/tools/validate_cmrf_models.py" "$ROOT/tools/cmrf_chain_state.py" >"$STATE_DIR/$RUN_ID.py_compile.log" 2>&1
"$PY" "$ROOT/tools/validate_cmrf_models.py" --root "$ROOT" --imgsz 128 --all-scales \
  --output "$STATE_DIR/$RUN_ID.model_preflight.json" >"$STATE_DIR/$RUN_ID.model_preflight.log" 2>&1

for stage in "${STAGES[@]}"; do
  CURRENT_STAGE="$stage"
  write_state training
  TRAIN_PIDS=()
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/train_cmrf_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" \
      --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
    TRAIN_PIDS+=("$!")
  done
  printf '%s\n' "${TRAIN_PIDS[@]}" >"$STATE_DIR/$RUN_ID.$stage.train.pids"
  wait_group training "${TRAIN_PIDS[@]}"

  write_state testing
  TEST_PIDS=()
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    weights="$TRAIN_DIR/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "Missing checkpoint: $weights" >&2; exit 76; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 \
      >"$TEST_DIR/$name.log" 2>&1 &
    TEST_PIDS+=("$!")
  done
  printf '%s\n' "${TEST_PIDS[@]}" >"$STATE_DIR/$RUN_ID.$stage.test.pids"
  wait_group testing "${TEST_PIDS[@]}"
  COMPLETED+=("$stage")
  "$PY" "$ROOT/tools/collect_cmrf_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}" \
    --l0-reference "$L0_REFERENCE" --p0-reference "$P0_REFERENCE" >"$STATE_DIR/$RUN_ID.$stage.collect.log" 2>&1
done

CURRENT_STAGE=complete
write_state complete "c0_c5_finished=true; three_parallel_seeds_per_stage=true"
echo "CMRF C0--C5 chain completed: run_id=$RUN_ID"
