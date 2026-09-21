from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ultralytics
assert Path(ultralytics.__file__).resolve().is_relative_to(ROOT)
import torch
import yaml

from ultralytics import YOLO
from ultralytics.nn.modules.ucra_v2 import UCRA1v2, UCRA2v2


ROOT = Path(__file__).resolve().parents[1]
MODEL_A3 = ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v2-a3-no-offset.yaml"
MODEL_A4 = ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml"


def check_module_identity():
    torch.manual_seed(0)
    u1 = UCRA1v2(256, 128).eval()
    d1 = torch.randn(1, 256, 20, 20)
    l1 = torch.randn(1, 128, 40, 40)
    ref1 = torch.nn.functional.interpolate(d1, size=(40, 40), mode="nearest")
    out1 = u1([d1, l1])
    assert torch.equal(out1, ref1)

    u2 = UCRA2v2(128, 128).eval()
    d2 = torch.randn(1, 128, 40, 40)
    l2 = torch.randn(1, 128, 80, 80)
    ref2 = torch.nn.functional.interpolate(d2, size=(80, 80), mode="nearest")
    out2 = u2([d2, l2])
    assert torch.equal(out2, ref2)


def check_model_yaml():
    for path in (MODEL_A3, MODEL_A4):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["nc"] == 4
        assert len(cfg["backbone"]) + len(cfg["head"]) == 33

        model = YOLO(str(path))
        layers = model.model.model
        assert len(layers) == 33
        assert layers[15].__class__.__name__ == "UCRA1v2"
        assert layers[19].__class__.__name__ == "UCRA2v2"
        assert layers[32].f == [23, 27, 31]

        x = torch.randn(1, 3, 640, 640)
        model.model.eval()
        with torch.no_grad():
            _ = model.model(x)


if __name__ == "__main__":
    check_module_identity()
    check_model_yaml()
    print("UCRA-v2 repository verification passed.")
