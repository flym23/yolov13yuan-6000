#!/usr/bin/env python3
"""Build every MESA ablation at n/s/l/x and record topology and complexity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_mesa_ablation import STAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--all-scales", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root)) if str(root) not in sys.path else None
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import yaml_load
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    records = {}
    for stage, (filename, _) in STAGES.items():
        path = root / "ultralytics/cfg/models/v13/mesa_ablation" / filename
        records[stage] = {}
        for scale in ("n", "s", "l", "x") if args.all_scales else ("n",):
            config = yaml_load(path)
            config["scale"] = scale
            model = DetectionModel({**config, "scale": scale}, ch=3, nc=4, verbose=False)
            classes = [model.model[index].__class__.__name__ for index in (4, 19, 20, 21, 32)]
            expected = "MACRDSC3k2" if stage == "m0" else "MESADSC3k2"
            if classes != ["CMRFDSC3k2", "Upsample", "Concat", expected, "Detect"]:
                raise AssertionError(f"{stage}/{scale}: unexpected topology {classes}")
            if list(model.stride.cpu().tolist()) != [8.0, 16.0, 32.0]:
                raise AssertionError(f"{stage}/{scale}: unexpected stride {model.stride}")
            records[stage][scale] = {"yaml": str(path), "layers": len(model.model), "topology": classes, "parameters": int(get_num_params(model)), "gflops": float(get_flops(model, imgsz=args.imgsz)), "stride": [float(value) for value in model.stride.cpu().tolist()]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
