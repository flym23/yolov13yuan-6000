#!/usr/bin/env python3
"""Atomically record the compact state of the SEAL ablation chain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("initializing", "running", "complete", "failed"), required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--completed", nargs="*", default=[])
    args = parser.parse_args()
    payload = {
        "run_id": args.run_id,
        "status": args.status,
        "stage": args.stage,
        "completed_stages": args.completed,
        "exit_code": args.exit_code,
        "launcher_pid": args.launcher_pid,
        "dataset": "/home/room305/ZZF/URPC2019/data.yaml",
        "settings": {
            "epochs": 300, "patience": 40, "batch": 16, "imgsz": 640, "device": 0, "workers": 2,
            "amp": False, "deterministic": True, "plots": False, "parallel_seed_processes": 3,
        },
        "initialization": "YOLO(model_yaml).load(yolov13n.pt)",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.path)


if __name__ == "__main__":
    main()
