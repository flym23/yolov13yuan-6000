#!/usr/bin/env python3
"""Build and structurally audit every CARM ablation YAML without touching training data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_carm_ablation import STAGES


EXPECTED = {
    "a0": ("CMRFDSC3k2", "Upsample", "DSC3k2"),
    "a1": ("CARMDSC3k2", "Upsample", "DSC3k2"),
    "a2": ("CMRFDSC3k2", "OrthogonalComplementaryAlignUp", "DSC3k2"),
    "a3": ("CMRFDSC3k2", "Upsample", "MACRDSC3k2"),
    "a4": ("CARMDSC3k2", "OrthogonalComplementaryAlignUp", "DSC3k2"),
    "a5": ("CARMDSC3k2", "Upsample", "MACRDSC3k2"),
    "a6": ("CMRFDSC3k2", "OrthogonalComplementaryAlignUp", "MACRDSC3k2"),
    "a7": ("CARMDSC3k2", "OrthogonalComplementaryAlignUp", "MACRDSC3k2"),
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
    for stage, (filename, structure) in STAGES.items():
        path = root / "ultralytics" / "cfg" / "models" / "v13" / "carm_ablation" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        stage_records = {}
        for scale in scales:
            config = yaml_load(path)
            config["scale"] = scale
            model = DetectionModel(config, ch=3, nc=4, verbose=False)
            expected_layer4, expected_layer19, expected_layer21 = EXPECTED[stage]
            assert len(model.model) == 33
            assert list(model.model[-1].stride.cpu().tolist()) == [8.0, 16.0, 32.0]
            assert model.model[4].__class__.__name__ == expected_layer4
            assert model.model[19].__class__.__name__ == expected_layer19
            assert model.model[20].__class__.__name__ == "Concat"
            assert model.model[21].__class__.__name__ == expected_layer21
            stage_records[scale] = {
                "yaml": str(path),
                "structure": structure,
                "layer4": expected_layer4,
                "layer19": expected_layer19,
                "layer20": "Concat",
                "layer21": expected_layer21,
                "parameters": int(get_num_params(model)),
                "gflops": float(get_flops(model, imgsz=args.imgsz)),
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
