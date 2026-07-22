#!/usr/bin/env bash
# One-time GPU handoff: wait only while the verified LCER-DCRA launcher remains alive, then exec CPCR.
set -Ee -o pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
DEPENDENCY_PID=${LCER_DCRA_LAUNCHER_PID:-463031}
DEPENDENCY_COMMAND=/home/room305/ZZF/yolov13-6000/run_lcer_dcra_l0_l3.sh
LOG_DIR="$ROOT/runs/cpcr_ablation"

mkdir -p "$LOG_DIR"
if [[ -r "/proc/$DEPENDENCY_PID/cmdline" ]] && tr '\0' ' ' <"/proc/$DEPENDENCY_PID/cmdline" | grep -Fq "$DEPENDENCY_COMMAND"; then
  echo "$(date -u +%FT%TZ) waiting for LCER-DCRA launcher pid=$DEPENDENCY_PID"
  while kill -0 "$DEPENDENCY_PID" 2>/dev/null; do
    sleep 60
  done
  echo "$(date -u +%FT%TZ) LCER-DCRA launcher exited; starting CPCR"
else
  echo "$(date -u +%FT%TZ) dependency is absent or no longer matches; starting CPCR"
fi
exec "$ROOT/run_cpcr_ablation.sh"
