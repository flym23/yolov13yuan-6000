#!/usr/bin/env python3
"""Build and structurally audit every CMRF ablation YAML without touching training data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


YAMLS = {
    "l0": "yolov13-cmrf-l0.yaml",
    "c0": "yolov13-cmrf-c0-h3.yaml",
    "c1": "yolov13-cmrf-c1-crr.yaml",
    "c2": "yolov13-cmrf-c2-smpgr.yaml",
    "c3": "yolov13-cmrf-c3-h3-rfa.yaml",
    "c4": "yolov13-cmrf-c4-crr-smpg.yaml",
    "c5": "yolov13-cmrf-c5-full.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--all-scales", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    records = {}
    scales = ("n", "s", "l", "x") if args.all_scales else ("n",)
    for stage, filename in YAMLS.items():
        path = root / "ultralytics" / "cfg" / "models" / "v13" / "cmrf_ablation" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        stage_records = {}
        for scale in scales:
            config = yaml_load(path)
            config["scale"] = scale
            model = DetectionModel(config, ch=3, nc=4, verbose=False)
            assert len(model.model) == 33
            assert list(model.model[-1].stride.cpu().tolist()) == [8.0, 16.0, 32.0]
            expected_layer4 = "DSC3k2" if stage == "l0" else "SCPGDSC3k2" if stage in {"c0", "c3"} else "CMRFDSC3k2"
            expected_layer19 = "ReliabilityFrequencyAlignUp" if stage in {"c3", "c5"} else "Upsample"
            assert model.model[4].__class__.__name__ == expected_layer4
            assert model.model[19].__class__.__name__ == expected_layer19
            assert model.model[20].__class__.__name__ == "Concat"
            stage_records[scale] = {
                "yaml": str(path), "layer4": expected_layer4, "layer19": expected_layer19,
                "parameters": int(get_num_params(model)), "gflops": float(get_flops(model, imgsz=args.imgsz)),
                "stride": [float(value) for value in model.model[-1].stride.cpu().tolist()],
            }
        records[stage] = stage_records
    payload = {"imgsz": args.imgsz, "scales": list(scales), "models": records}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
