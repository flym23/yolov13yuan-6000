#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/rtx6000/ZZF/yolov13yuan-6000
PY=/home/rtx6000/.conda/envs/yolov13/bin/python
TRAIN=$ROOT/runs/train
TEST=$ROOT/runs/test
STATE=$ROOT/runs/durc_ablation
RUN_ID=${DURC_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOCK=$STATE/chain.lock
CURRENT=initialization
DEFAULT_STAGES=(A1 A2 A3 B1 B2 B3 B4 C1 C2 C3 C4)
STAGES=("${DEFAULT_STAGES[@]}")
SEEDS=(0 1 2)
COMPLETED=()
SKIP_TRAIN_STAGES=${DURC_SKIP_TRAIN_STAGES:-}
COMPLETED_STAGES=${DURC_COMPLETED_STAGES:-}
SEED_OVERRIDES=${DURC_SEED_OVERRIDES:-}

if [[ -n ${DURC_STAGES:-} ]]; then
  read -r -a STAGES <<< "${DURC_STAGES//,/ }"
  for stage in "${STAGES[@]}"; do
    [[ " ${DEFAULT_STAGES[*]} " == *" ${stage} "* ]] || {
      echo "unknown DURC stage requested: ${stage}" >&2
      exit 64
    }
  done
fi
if [[ -n "$COMPLETED_STAGES" ]]; then
  read -r -a COMPLETED <<< "${COMPLETED_STAGES//,/ }"
fi

mkdir -p "$TRAIN" "$TEST" "$STATE"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/tools"
mkdir "$LOCK" 2>/dev/null || { echo "DURC chain already running" >&2; exit 73; }

write_state() {
  DURC_STATUS=$1 DURC_STAGE=$2 DURC_CODE=${3:-0} DURC_RUN_ID=$RUN_ID DURC_PID=$$ DURC_STATE=$STATE \
    "$PY" -c 'import json,os; from pathlib import Path; from datetime import datetime,timezone; p=Path(os.environ["DURC_STATE"])/"state.json"; d={"run_id":os.environ["DURC_RUN_ID"],"status":os.environ["DURC_STATUS"],"stage":os.environ["DURC_STAGE"],"exit_code":int(os.environ["DURC_CODE"]),"launcher_pid":int(os.environ["DURC_PID"]),"updated_at":datetime.now(timezone.utc).isoformat(),"precision":"fp32","amp":False,"physical_gpu":"cuda:0","parallel_seed_processes":3,"seeds":[0,1,2],"stage_order":["A1","A2","A3","B1","B2","B3","B4","C1","C2","C3","C4"]}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(d,indent=2),encoding="utf-8"); t.replace(p)'
}

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  children=$(jobs -pr || true)
  if [[ -n "$children" ]]; then
    kill $children 2>/dev/null || true
    wait $children 2>/dev/null || true
  fi
  rmdir "$LOCK" 2>/dev/null || true
  ((rc == 0)) || write_state failed "$CURRENT" "$rc" || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

prepare_concurrent_launch() {
  if ! nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits; then
    echo "nvidia-smi availability query failed" >&2
    return 1
  fi
  CURRENT=concurrent_gpu_launch
  write_state running "$CURRENT"
}

wait_group() {
  local bad=0 pid
  for pid in "$@"; do
    wait "$pid" || bad=1
  done
  ((bad == 0))
}

wait_for_dataset_caches() {
  local seed0_pid=$1 elapsed=0
  local train_cache="$ROOT/.durc_dataset/labels/train.cache"
  local val_cache="$ROOT/.durc_dataset/labels/val.cache"
  while ((elapsed < 120)); do
    if [[ -s "$train_cache" && -s "$val_cache" ]]; then
      return 0
    fi
    if ! kill -0 "$seed0_pid" 2>/dev/null; then
      echo "seed-0 exited before DURC dataset caches were prepared" >&2
      return 1
    fi
    sleep 1
    ((elapsed += 1))
  done
  echo "timed out waiting for DURC dataset caches" >&2
  return 1
}

skip_train_stage() {
  [[ ",${SKIP_TRAIN_STAGES}," == *",$1,"* ]]
}

collect_results() {
  local args=(--root "$ROOT" --run-id "$RUN_ID" --stages "${COMPLETED[@]}" --seeds "${SEEDS[@]}")
  if [[ -n "$SEED_OVERRIDES" ]]; then
    args+=(--seed-overrides "$SEED_OVERRIDES")
  fi
  "$PY" "$ROOT/tools/collect_durc_ablation.py" "${args[@]}"
}

echo $$ > "$STATE/launcher.pid"
echo "$RUN_ID" > "$STATE/run_id.txt"
write_state running "$CURRENT"

prepare_concurrent_launch

CURRENT=prepare_durc_dataset
write_state running "$CURRENT"
"$PY" "$ROOT/tools/prepare_durc_dataset.py" \
  --source-data "$ROOT/data.yaml" \
  --target-root "$ROOT/.durc_dataset" \
  --target-data "$ROOT/data_durc.yaml" \
  > "$STATE/prepare_dataset_${RUN_ID}.log" 2>&1

CURRENT=preflight_runtime
write_state running "$CURRENT"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/tools/run_durc_selfcheck.py" --device cuda:0 \
  > "$STATE/preflight_runtime_${RUN_ID}.log" 2>&1

CURRENT=preflight_models
write_state running "$CURRENT"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/tools/validate_durc_models.py" \
  --root "$ROOT" --device cuda:0 --imgsz 640 --batch 2 \
  --output "$STATE/complexity.json" > "$STATE/preflight_models_${RUN_ID}.log" 2>&1
write_state preflight_complete "$CURRENT"

CACHE_READY=false
for stage in "${STAGES[@]}"; do
  stage_lower=${stage,,}
  if skip_train_stage "$stage"; then
    CURRENT=${stage}.seeds0_1_2.reuse_weights
    write_state running "$CURRENT"
  else
    CURRENT=${stage}.seeds0_1_2.train
    write_state running "$CURRENT"
    PIDS=()
    for seed in "${SEEDS[@]}"; do
      name=durc_${RUN_ID}_${stage_lower}_seed${seed}
      CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/tools/train_durc_worker.py" \
        --root "$ROOT" --stage "$stage" --name "$name" --seed "$seed" --epochs 200 \
        > "$TRAIN/${name}.log" 2>&1 &
      PIDS+=("$!")
      echo "$!" > "$STATE/${stage_lower}_seed${seed}.train.pid"
      if [[ "$CACHE_READY" == false && "$seed" == 0 ]]; then
        wait_for_dataset_caches "$!"
        CACHE_READY=true
      fi
    done
    wait_group "${PIDS[@]}"
  fi
  for seed in "${SEEDS[@]}"; do
    name=durc_${RUN_ID}_${stage_lower}_seed${seed}
    test -s "$TRAIN/$name/weights/best.pt"
    test -s "$TRAIN/$name/weights/last.pt"
  done

  CURRENT=${stage}.seeds0_1_2.test
  write_state running "$CURRENT"
  PIDS=()
  for seed in "${SEEDS[@]}"; do
    name=durc_${RUN_ID}_${stage_lower}_seed${seed}
    CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true "$PY" "$ROOT/test.py" \
      --weights "$TRAIN/$name/weights/best.pt" --name "$name" --device 0 --batch 16 --imgsz 640 \
      > "$TEST/${name}.log" 2>&1 &
    PIDS+=("$!")
    echo "$!" > "$STATE/${stage_lower}_seed${seed}.test.pid"
  done
  wait_group "${PIDS[@]}"

  CURRENT=${stage}.seeds0_1_2.audit
  write_state running "$CURRENT"
  PIDS=()
  for seed in "${SEEDS[@]}"; do
    name=durc_${RUN_ID}_${stage_lower}_seed${seed}
    test -s "$TEST/$name/summary_metrics.json"
    test -s "$TEST/$name/scale_ap_metrics.json"
    CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/tools/audit_durc_checkpoint.py" \
      --stage "$stage" --best "$TRAIN/$name/weights/best.pt" --last "$TRAIN/$name/weights/last.pt" \
      --device cuda:0 --imgsz 160 --output "$TEST/${name}.durc_audit.json" \
      > "$TEST/${name}.durc_audit.log" 2>&1 &
    PIDS+=("$!")
  done
  wait_group "${PIDS[@]}"

  COMPLETED+=("$stage")
  collect_results
  write_state stage_complete "$stage"
done

CURRENT=reporting
write_state running "$CURRENT"
collect_results
CURRENT=complete
write_state complete "$CURRENT"
echo "DURC ablation chain completed: run_id=$RUN_ID"
