#!/usr/bin/env bash
# Execute a resumable DCRQ B0→B4 chain. Arguments are root, launcher log, and immutable run ID.
set -Eeuo pipefail

ROOT_DIR=${1:?absolute project root is required}
LAUNCHER_LOG=${2:?absolute launcher log is required}
RUN_ID=${3:?immutable run ID is required}
DATA_YAML=/home/room305/ZZF/URPC2019/data.yaml

[[ "$ROOT_DIR" = /* && "$LAUNCHER_LOG" = /* ]] || { echo "root and launcher log must be absolute" >&2; exit 2; }
[[ -f "$DATA_YAML" && -f "$ROOT_DIR/yolov13n.pt" ]] || { echo "required data YAML or yolov13n.pt is missing" >&2; exit 2; }
mkdir -p "$(dirname "$LAUNCHER_LOG")"
export DCRQ_LAUNCHER_PID=$$
exec python3 "$ROOT_DIR/tools/run_dcrq_ablation.py" --root "$ROOT_DIR" --run-id "$RUN_ID" --data "$DATA_YAML"
