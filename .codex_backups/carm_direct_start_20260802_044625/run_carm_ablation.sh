#!/usr/bin/env bash
# CARM A0--A7: after L3-CRU finishes, run exactly three deterministic seed workers per group.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
BASELINE_ROOT=${BASELINE_ROOT:-/home/room305/ZZF/yolov13-6000}
L3CRU_STATE_FILE=${1:?usage: run_carm_ablation.sh /absolute/path/to/completed-l3cru-state.json}
PRETRAINED="$ROOT/yolov13n.pt"
T7_REFERENCE=${T7_REFERENCE:-}
C4_REFERENCE=${C4_REFERENCE:-$ROOT/runs/test/cmrf_20260729_123654_summary.json}
STATE_DIR="$ROOT/runs/carm_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=${CARM_RUN_ID:-carm_$(date -u +%Y%m%d_%H%M%S)}
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
STAGES=(a0 a1 a2 a3 a4 a5 a6 a7)
COMPLETED=()
CURRENT_STAGE=initializing
L0_REFERENCE="$BASELINE_ROOT/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json"
P0_REFERENCE="$BASELINE_ROOT/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json"

mkdir -p "$STATE_DIR" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "CARM chain already running for $RUN_ID" >&2; exit 73; }
export WANDB_DISABLED=true PIN_MEMORY=false

write_state() {
  "$PY" "$ROOT/tools/carm_chain_state.py" --path "$STATE_FILE" --run-id "$RUN_ID" --status "$1" \
    --stage "$CURRENT_STAGE" --detail "${2:-}" --launcher-pid "$$" --completed "${COMPLETED[@]}" \
    --l3cru-state "$L3CRU_STATE_FILE" --pretrained "$PRETRAINED"
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
        if [[ "$other" != "$pid" ]] && kill -0 "$other" 2>/dev/null; then
          kill "$other" 2>/dev/null || true
        fi
      done
    fi
  done
  (( failed == 0 )) || { echo "CARM $CURRENT_STAGE $label failed" >&2; return 75; }
}

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" && -f "$PRETRAINED" && -f "$L0_REFERENCE" && -f "$P0_REFERENCE" ]] || {
  echo "CARM input file missing" >&2
  exit 79
}
[[ -f "$L3CRU_STATE_FILE" ]] || { echo "L3-CRU state unavailable: $L3CRU_STATE_FILE" >&2; exit 80; }
L3CRU_STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$L3CRU_STATE_FILE")
[[ "$L3CRU_STATUS" == "complete" ]] || { echo "L3-CRU has not completed: status=$L3CRU_STATUS" >&2; exit 81; }

echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
CURRENT_STAGE=preflight
write_state running "l3cru_complete=true; initialization=yolov13n.pt; sha256=$(sha256sum "$PRETRAINED" | awk '{print $1}')"
"$PY" -m py_compile "$ROOT/ultralytics/nn/modules/block.py" "$ROOT/ultralytics/nn/modules/__init__.py" \
  "$ROOT/ultralytics/nn/tasks.py" "$ROOT/tests/test_carm_modules.py" "$ROOT/tools/run_carm_ablation.py" \
  "$ROOT/tools/validate_carm_models.py" "$ROOT/tools/summarize_carm_results.py" "$ROOT/tools/carm_chain_state.py" \
  >"$STATE_DIR/$RUN_ID.py_compile.log" 2>&1
"$PY" -m pytest "$ROOT/tests/test_carm_modules.py" -q >"$STATE_DIR/$RUN_ID.unit_tests.log" 2>&1
"$PY" "$ROOT/tools/validate_carm_models.py" --root "$ROOT" --imgsz 128 --all-scales \
  --output "$STATE_DIR/$RUN_ID.model_preflight.json" >"$STATE_DIR/$RUN_ID.model_preflight.log" 2>&1

for stage in "${STAGES[@]}"; do
  CURRENT_STAGE="$stage"
  write_state training
  TRAIN_PIDS=()
  for seed in 0 1 2; do
    name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/run_carm_ablation.py" --root "$ROOT" --stage "$stage" --data "$DATA" \
      --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
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
  REFERENCE_ARGS=(--l0-reference "$L0_REFERENCE" --p0-reference "$P0_REFERENCE")
  [[ -f "$C4_REFERENCE" ]] && REFERENCE_ARGS+=(--c4-reference "$C4_REFERENCE")
  [[ -n "$T7_REFERENCE" && -f "$T7_REFERENCE" ]] && REFERENCE_ARGS+=(--t7-reference "$T7_REFERENCE")
  "$PY" "$ROOT/tools/summarize_carm_results.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}" \
    "${REFERENCE_ARGS[@]}" >"$STATE_DIR/$RUN_ID.$stage.collect.log" 2>&1
done

CURRENT_STAGE=complete
write_state complete "a0_a7_finished=true; three_parallel_seeds_per_stage=true"
echo "CARM A0--A7 chain completed: run_id=$RUN_ID"
