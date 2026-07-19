#!/usr/bin/env bash
# Run URPC2020 baseline -> DGDR G1..G7 only after the ME-DCRA M0--M7 launcher finishes successfully.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/home/room305/.conda/envs/yolov13/bin/python}"
DATA="${DGDR_DATA:-/home/room305/ZZF/URPC2020/data.yaml}"
PREDECESSOR_ROOT="${MEDCRA_ROOT:-/home/room305/ZZF/yolov13-6000}"
PREDECESSOR_SCRIPT="$PREDECESSOR_ROOT/run_medcra_m0_m7_after_durc.sh"
PREDECESSOR_STATE="$PREDECESSOR_ROOT/runs/medcra_ablation/state.json"
STATE="$ROOT/runs/dgdr_ablation"
RUN_ID="${DGDR_RUN_ID:-dgdr_$(date -u +%Y%m%d_%H%M%S)}"
LOCK="$STATE/.chain_lock"
CURRENT="initializing"
STATUS="initializing"
CODE=0
PID="$$"
STAGES=(baseline g1 g2 g3 g4 g5 g6 g7)
COMPLETED=()

mkdir -p "$STATE"
mkdir "$LOCK" 2>/dev/null || { echo "DGDR chain already running" >&2; exit 73; }

write_state() {
  STATUS="$1"
  CURRENT="$2"
  CODE="$3"
  DGDR_RUN_ID="$RUN_ID" DGDR_STATUS="$STATUS" DGDR_STAGE="$CURRENT" DGDR_CODE="$CODE" DGDR_PID="$PID" \
    DGDR_STATE="$STATE" DGDR_DATA="$DATA" DGDR_PREDECESSOR="$PREDECESSOR_STATE" "$PY" -c '
import json, os
from datetime import datetime, timezone
from pathlib import Path
p = Path(os.environ["DGDR_STATE"]) / "state.json"
d = {
    "run_id": os.environ["DGDR_RUN_ID"], "status": os.environ["DGDR_STATUS"],
    "stage": os.environ["DGDR_STAGE"], "exit_code": int(os.environ["DGDR_CODE"]),
    "launcher_pid": int(os.environ["DGDR_PID"]), "dataset": os.environ["DGDR_DATA"],
    "epochs": 300, "patience": 40, "amp": False, "workers": 2, "plots": False,
    "dependency_state": os.environ["DGDR_PREDECESSOR"],
    "stage_order": ["baseline", "g1", "g2", "g3", "g4", "g5", "g6", "g7"],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
t = p.with_suffix(".tmp")
t.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
t.replace(p)
'
}

cleanup() {
  rmdir "$LOCK" 2>/dev/null || true
}

on_error() {
  local rc="$?"
  write_state failed "$CURRENT" "$rc" || true
  exit "$rc"
}

trap on_error ERR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

dependency_status() {
  "$PY" - "$PREDECESSOR_STATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("missing")
print(str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")).lower())
PY
}

wait_for_predecessor() {
  CURRENT="waiting_for_medcra"
  while true; do
    if pgrep -f "$PREDECESSOR_SCRIPT" >/dev/null; then
      write_state waiting_for_predecessor "$CURRENT" 0
    elif [[ ! -f "$PREDECESSOR_STATE" ]]; then
      # Account/project migration may precede the new ME-DCRA launch.  Keep the
      # DGDR launcher dormant instead of treating an as-yet-uncreated state as success.
      write_state waiting_for_predecessor "$CURRENT" 0
    else
      local result
      result="$(dependency_status)"
      case "$result" in
        complete|completed|success)
          return 0
          ;;
        initializing|pending|queued|running|preflight_complete|stage_complete)
          write_state waiting_for_predecessor "$CURRENT" 0
          ;;
        *)
          echo "ME-DCRA predecessor ended with status '$result'; DGDR training will not start." >&2
          return 74
          ;;
      esac
    fi
    sleep 30
  done
}

run_stage() {
  local stage="$1"
  local failed=0
  local -a pids=()
  CURRENT="$stage"
  write_state running "$CURRENT" 0

  for seed in 0 1 2; do
    local name="${RUN_ID}_${stage}_seed${seed}"
    "$PY" "$ROOT/tools/train_dgdr_worker.py" --root "$ROOT" --stage "$stage" --data "$DATA" \
      --name "$name" --seed "$seed" --epochs 300 --patience 40 >"$STATE/${name}.train.log" 2>&1 &
    pids+=("$!")
    echo "$!" >"$STATE/${stage}_seed${seed}.train.pid"
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || return 75

  pids=()
  for seed in 0 1 2; do
    local name="${RUN_ID}_${stage}_seed${seed}"
    local weights="$ROOT/runs/train/${name}/weights/best.pt"
    [[ -f "$weights" ]] || { echo "missing checkpoint: $weights" >&2; return 76; }
    "$PY" "$ROOT/test.py" --weights "$weights" --data "$DATA" --name "$name" --device 0 --batch 16 --imgsz 640 \
      --workers 2 >"$STATE/${name}.test.log" 2>&1 &
    pids+=("$!")
    echo "$!" >"$STATE/${stage}_seed${seed}.test.pid"
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || return 77

  COMPLETED+=("$stage")
  "$PY" "$ROOT/tools/collect_dgdr_ablation.py" --root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}"
}

echo "$$" >"$STATE/launcher.pid"
write_state initializing "$CURRENT" 0
[[ -x "$PY" ]] || { echo "Python runtime unavailable: $PY" >&2; exit 78; }
[[ -f "$DATA" ]] || { echo "Dataset YAML unavailable: $DATA" >&2; exit 79; }
"$PY" "$ROOT/tools/generate_dgdr_yamls.py" >"$STATE/${RUN_ID}.yaml.log" 2>&1
"$PY" "$ROOT/tools/validate_dgdr_models.py" --root "$ROOT" --device cpu --imgsz 640 >"$STATE/${RUN_ID}.model_preflight.log" 2>&1
"$PY" "$ROOT/tools/validate_dgdr_checkpoint.py" --root "$ROOT" --weights "$ROOT/yolov13n.pt" \
  --output "$STATE/${RUN_ID}.checkpoint_preflight.json" >"$STATE/${RUN_ID}.checkpoint_preflight.log" 2>&1
wait_for_predecessor

for stage in "${STAGES[@]}"; do
  run_stage "$stage"
done

CURRENT="complete"
write_state complete "$CURRENT" 0
echo "DGDR baseline + G1--G7 chain completed: run_id=$RUN_ID"
