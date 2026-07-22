#!/usr/bin/env bash
# Wait for the active URR-DCRA chain, then hand off to the local RAMP chain.
set -Ee -o pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
DEPENDENCY='[r]un_urr_dcra_u2_u8_after_dgdr.sh'
LOG_DIR="$ROOT/runs/ramp_ablation"

mkdir -p "$LOG_DIR"
while pgrep -f "$DEPENDENCY" >/dev/null; do
  echo "$(date -u +%FT%TZ) waiting for URR-DCRA chain to finish"
  sleep 60
done

echo "$(date -u +%FT%TZ) URR-DCRA finished; starting RAMP K1--K4"
exec "$ROOT/run_ramp_ablation.sh"
