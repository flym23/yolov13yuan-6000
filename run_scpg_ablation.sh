#!/usr/bin/env bash
# SCPG H1--H5 chain: each stage trains seed 0/1/2 concurrently, then tests them concurrently.
set -Ee -o pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
PY=/home/room305/.conda/envs/yolov13/bin/python
DATA=/home/room305/ZZF/URPC2020half/data.yaml
STATE="$ROOT/runs/scpg_ablation"
TRAIN_DIR="$ROOT/runs/train"
TEST_DIR="$ROOT/runs/test"
RUN_ID=scpg_$(date -u +%Y%m%d_%H%M%S)
LOCK="$STATE/.chain_lock"
PID=$$
CURRENT=initializing

mkdir -p "$STATE" "$TRAIN_DIR" "$TEST_DIR"
mkdir "$LOCK" 2>/dev/null || { echo "SCPG chain already running" >&2; exit 73; }

write_state() {
  SCPG_RUN_ID="$RUN_ID" SCPG_STATUS="$1" SCPG_STAGE="$2" SCPG_CODE="$3" SCPG_PID="$PID" SCPG_STATE="$STATE" SCPG_DATA="$DATA" \
    "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["SCPG_STATE"]) / "state.json"
payload = {
    "run_id": os.environ["SCPG_RUN_ID"],
    "status": os.environ["SCPG_STATUS"],
    "stage": os.environ["SCPG_STAGE"],
    "exit_code": int(os.environ["SCPG_CODE"]),
    "launcher_pid": int(os.environ["SCPG_PID"]),
    "dataset": os.environ["SCPG_DATA"],
    "epochs": 300, "patience": 40, "amp": False, "workers": 2,
    "plots": False, "pin_memory": False,
    "stage_order": ["h1", "h2", "h3", "h4", "h5"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
temp.replace(path)
'
}

cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
on_error() {
  code=$?
  write_state failed "$CURRENT" "$code" || true
  exit "$code"
}
trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_stage() {
  stage=$1
  CURRENT=$stage
  write_state running "$CURRENT" 0
  failed=0
  pids=""

  for seed in 0 1 2; do
    name="$RUN_ID"_"$stage"_seed"$seed"
    "$PY" "$ROOT/tools/train_scpg_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" \
      --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$TRAIN_DIR/$name.log" 2>&1 &
    pids="$pids $!"
    echo "$!" >"$STATE/$stage"_seed"$seed.train.pid"
  done
  for child in $pids; do wait "$child" || failed=1; done
  ((failed == 0)) || return 75

  failed=0
  pids=""
  for seed in 0 1 2; do
    name="$RUN_ID"_"$stage"_seed"$seed"
    weights="$ROOT/runs/train/$name/weights/best.pt"
    [[ -f "$weights" ]] || { echo "missing checkpoint: $weights" >&2; return 76; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 \
      --workers 2 >"$TEST_DIR/$name.log" 2>&1 &
    pids="$pids $!"
    echo "$!" >"$STATE/$stage"_seed"$seed.test.pid"
  done
  for child in $pids; do wait "$child" || failed=1; done
  ((failed == 0)) || return 77

  completed="$completed $stage"
  "$PY" "$ROOT/tools/collect_scpg_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages $completed
}

export PIN_MEMORY=false
echo "$PID" >"$STATE/launcher.pid"
write_state initializing "$CURRENT" 0
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }

"$PY" "$ROOT/tools/generate_scpg_yamls.py" >"$STATE/$RUN_ID.yaml.log" 2>&1
"$PY" "$ROOT/tools/validate_scpg_models.py" --root "$ROOT" --device cpu --imgsz 128 >"$STATE/$RUN_ID.model_preflight.log" 2>&1
"$PY" "$ROOT/tools/validate_scpg_checkpoint.py" --root "$ROOT" --weights "$ROOT/yolov13n.pt" \
  --output "$STATE/$RUN_ID.checkpoint_preflight.json" >"$STATE/$RUN_ID.checkpoint_preflight.log" 2>&1

completed=""
for stage in h1 h2 h3 h4 h5; do
  run_stage "$stage"
done

CURRENT=complete
write_state complete "$CURRENT" 0
echo "SCPG H1--H5 chain completed: run_id=$RUN_ID"

