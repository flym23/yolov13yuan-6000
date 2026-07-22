#!/usr/bin/env bash
# RAMP K1--K4 chain: train three seeds concurrently, then test three seeds concurrently.
set -Ee -o pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE="$ROOT/runs/ramp_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=ramp_$(date -u +%Y%m%d_%H%M%S)

mkdir -p "$STATE" "$TRAIN_DIR" "$TEST_DIR"
export PIN_MEMORY=false

[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 1; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 1; }

"$PY" "$ROOT/tools/generate_ramp_yamls.py" >"$STATE/$RUN_ID.yaml.log" 2>&1
"$PY" "$ROOT/tools/validate_ramp_models.py" --root "$ROOT" --device cpu --imgsz 128 >"$STATE/$RUN_ID.model_preflight.log" 2>&1
"$PY" "$ROOT/tools/validate_ramp_checkpoint.py" --root "$ROOT" --weights "$ROOT/yolov13n.pt" \
  --output "$STATE/$RUN_ID.checkpoint_preflight.json" >"$STATE/$RUN_ID.checkpoint_preflight.log" 2>&1

completed=""
for stage in k1 k2 k3 k4; do
  train_pids=""
  for seed in 0 1 2; do
    name="$RUN_ID"_"$stage"_seed"$seed"
    "$PY" "$ROOT/tools/train_ramp_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" --name "$name" \
      --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
    train_pids="$train_pids $!"
  done
  failed=0
  for pid in $train_pids; do wait "$pid" || failed=1; done
  ((failed == 0)) || { echo "RAMP $stage training failed" >&2; exit 1; }

  test_pids=""
  for seed in 0 1 2; do
    name="$RUN_ID"_"$stage"_seed"$seed"
    weights="$TRAIN_DIR/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "Missing checkpoint: $weights" >&2; exit 1; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 --workers 2 \
      >"$TEST_DIR/$name.log" 2>&1 &
    test_pids="$test_pids $!"
  done
  failed=0
  for pid in $test_pids; do wait "$pid" || failed=1; done
  ((failed == 0)) || { echo "RAMP $stage testing failed" >&2; exit 1; }

  completed="$completed $stage"
  "$PY" "$ROOT/tools/collect_ramp_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $completed
done

echo "RAMP K1--K4 chain completed: run_id=$RUN_ID"
