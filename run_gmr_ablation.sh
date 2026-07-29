#!/usr/bin/env bash
# GMR R1--R7 chain: exactly three seeds and exactly three concurrent processes per stage.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE_DIR="$ROOT/runs/gmr_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
G4_WEIGHTS="$ROOT/runs/train/gsd_20260726_022443_g4_seed0/weights/best.pt"
RUN_ID=${GMR_RUN_ID:-gmr_$(date -u +%Y%m%d_%H%M%S)}
START_STAGE=${GMR_START_STAGE:-r1}
STAGES=(r1 r2 r3 r4 r5 r6 r7)
STATE_FILE="$STATE_DIR/$RUN_ID.state.json"
LOCK="$STATE_DIR/.${RUN_ID}.chain_lock"
CURRENT_STAGE=initializing
COMPLETED=""
START_INDEX=-1

for index in "${!STAGES[@]}"; do
  if [[ "${STAGES[$index]}" == "$START_STAGE" ]]; then
    START_INDEX=$index
    break
  fi
done
(( START_INDEX >= 0 )) || { echo "Invalid GMR_START_STAGE: $START_STAGE" >&2; exit 72; }
for ((index=0; index < START_INDEX; index++)); do
  COMPLETED+="${COMPLETED:+ }${STAGES[$index]}"
done

mkdir -p "$STATE_DIR" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "GMR chain already running for $RUN_ID" >&2; exit 73; }
export PIN_MEMORY=false
export WANDB_DISABLED=true

write_state() {
  GMR_STATE_FILE="$STATE_FILE" GMR_RUN_ID="$RUN_ID" GMR_STATUS="$1" GMR_STAGE="$CURRENT_STAGE" GMR_DETAIL="${2:-}" GMR_COMPLETED="$COMPLETED" GMR_PID="$$" GMR_START_STAGE="$START_STAGE" "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["GMR_STATE_FILE"])
payload = {
    "run_id": os.environ["GMR_RUN_ID"], "status": os.environ["GMR_STATUS"], "stage": os.environ["GMR_STAGE"],
    "detail": os.environ["GMR_DETAIL"], "launcher_pid": int(os.environ["GMR_PID"]),
    "completed_stages": os.environ["GMR_COMPLETED"].split(), "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
    "settings": {"epochs": 300, "patience": 40, "workers": 2, "amp": False, "plots": False, "pin_memory": False, "deterministic": True},
    "seed_policy": [0, 1, 2], "max_concurrent_processes": 3, "resume_from": os.environ["GMR_START_STAGE"],
    "r0_policy": "reuse designated L0 and P0 original-YOLOv13 summaries; do not retrain R0",
    "chain_policy": "unconditional R1-R7 sequential execution after one-epoch R7 smoke test; no result threshold or performance gate",
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
  (( failed == 0 )) || { echo "GMR $CURRENT_STAGE $label failed" >&2; return 75; }
}

run_seed_batch() {
  local stage mode seed name weights
  local -a pids=()
  stage="$1"; mode="$2"; shift 2
  if [[ "$mode" == train ]]; then
    for seed in "$@"; do
      name="${RUN_ID}_${stage}_seed${seed}"
      "$PY" "$ROOT/tools/train_gmr_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
      pids+=("$!")
    done
  else
    for seed in "$@"; do
      name="${RUN_ID}_${stage}_seed${seed}"
      weights="$TRAIN_DIR/$name/weights/best.pt"
      [[ -f "$weights" ]] || { echo "Missing checkpoint: $weights" >&2; return 76; }
      "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 >"$TEST_DIR/$name.log" 2>&1 &
      pids+=("$!")
    done
  fi
  printf '%s\n' "${pids[@]}" >>"$STATE_DIR/$RUN_ID.$stage.$mode.pids"
  wait_group "$mode" "${pids[@]}"
}

run_stage() {
  local stage
  stage="$1"; CURRENT_STAGE="$stage"; : >"$STATE_DIR/$RUN_ID.$stage.train.pids"; : >"$STATE_DIR/$RUN_ID.$stage.test.pids"
  write_state training "seeds=0,1,2"
  run_seed_batch "$stage" train 0 1 2
  write_state testing "seeds=0,1,2"
  run_seed_batch "$stage" test 0 1 2
  COMPLETED="$COMPLETED $stage"
  "$PY" "$ROOT/tools/collect_gmr_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $COMPLETED >"$STATE_DIR/$RUN_ID.$stage.collect.log" 2>&1
}

run_smoke_trial() {
  local name weights
  CURRENT_STAGE=smoke_r7
  name="${RUN_ID}_smoke_r7_seed0"
  write_state smoke_training "epochs=1; formal R1-R7 does not reuse this checkpoint"
  "$PY" "$ROOT/tools/train_gmr_worker.py" --root "$ROOT" --stage r7 --data "$DATA" --name "$name" --seed 0 --epochs 1 --patience 1 >"$TRAIN_DIR/$name.log" 2>&1
  weights="$TRAIN_DIR/$name/weights/best.pt"
  [[ -f "$weights" ]] || { echo "Smoke trial did not create $weights" >&2; return 77; }
  write_state smoke_testing "epochs=1"
  "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 >"$TEST_DIR/$name.log" 2>&1
  [[ -f "$TEST_DIR/$name/summary_metrics.json" ]] || { echo "Smoke trial test summary missing" >&2; return 77; }
}

verify_resume_prerequisites() {
  local stage seed name
  (( START_INDEX > 0 )) || return 0
  for ((index=0; index < START_INDEX; index++)); do
    stage="${STAGES[$index]}"
    for seed in 0 1 2; do
      name="${RUN_ID}_${stage}_seed${seed}"
      [[ -f "$TEST_DIR/$name/summary_metrics.json" ]] || {
        echo "Cannot resume at $START_STAGE: missing completed test summary $TEST_DIR/$name/summary_metrics.json" >&2
        return 81
      }
    done
  done
  [[ -f "$TEST_DIR/${RUN_ID}_smoke_r7_seed0/summary_metrics.json" ]] || {
    echo "Cannot resume at $START_STAGE: missing smoke-test summary" >&2
    return 82
  }
}

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
[[ -f "$G4_WEIGHTS" ]] || { echo "Required GSD-G4 checkpoint unavailable: $G4_WEIGHTS" >&2; exit 80; }
echo "$$" >"$STATE_DIR/$RUN_ID.launcher.pid"
CURRENT_STAGE=preflight
write_state running "launcher_pid=$$"
"$PY" "$ROOT/tools/validate_gmr_models.py" --root "$ROOT" --device cpu --imgsz 128 >"$STATE_DIR/$RUN_ID.model_preflight.log" 2>&1
"$PY" "$ROOT/tools/validate_gmr_checkpoint.py" --root "$ROOT" --weights "$ROOT/yolov13n.pt" --output "$STATE_DIR/$RUN_ID.checkpoint_preflight_original.json" >"$STATE_DIR/$RUN_ID.checkpoint_preflight_original.log" 2>&1
"$PY" "$ROOT/tools/validate_gmr_checkpoint.py" --root "$ROOT" --weights "$G4_WEIGHTS" --output "$STATE_DIR/$RUN_ID.checkpoint_preflight_gsd_g4.json" >"$STATE_DIR/$RUN_ID.checkpoint_preflight_gsd_g4.log" 2>&1
verify_resume_prerequisites
if (( START_INDEX == 0 )); then
  run_smoke_trial
fi

for ((index=START_INDEX; index < ${#STAGES[@]}; index++)); do
  run_stage "${STAGES[$index]}"
done
CURRENT_STAGE=complete
write_state complete "r1_r7_finished=true; three_seed_protocol=true; all_stages_started_without_result_gates=true"
echo "GMR R1--R7 chain completed: run_id=$RUN_ID"
