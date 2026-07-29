#!/usr/bin/env python3
"""Atomically write the small state record used by the CMRF training chain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--detail", default="")
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--completed", nargs="*", default=[])
    parser.add_argument("--tdr-state", required=True)
    args = parser.parse_args()
    payload = {
        "run_id": args.run_id,
        "status": args.status,
        "stage": args.stage,
        "detail": args.detail,
        "launcher_pid": args.launcher_pid,
        "completed_stages": args.completed,
        "dataset": "/home/room305/ZZF/URPC2020half/data.yaml",
        "settings": {"epochs": 300, "patience": 40, "batch": 16, "imgsz": 640, "device": 0, "workers": 2,
                     "amp": False, "deterministic": True, "plots": False, "parallel_seed_processes": 3},
        "baseline_policy": {
            "l0_reference": "/home/room305/ZZF/yolov13-6000/runs/test/lcer_dcra_20260722_045426_l0_baseline_summary.json",
            "p0_reference": "/home/room305/ZZF/yolov13-6000/runs/test/spc_lcer_dcra_20260722_162019_p0_baseline_summary.json",
            "note": "Historical L0/P0 references are retained as supplied; C0 SCPG-H3 is trained in this chain.",
        },
        "tdr_dependency_state": args.tdr_state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.path)


if __name__ == "__main__":
    main()
