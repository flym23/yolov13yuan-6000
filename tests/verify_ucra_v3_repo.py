from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ultralytics
assert Path(ultralytics.__file__).resolve().is_relative_to(ROOT)

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "B1": ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v3-b1-context-only.yaml",
    "B2": ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v3-b2-hybrid-full.yaml",
    "B3": ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v3-b3-full.yaml",
}

for run_id, cfg in MODELS.items():
    model = YOLO(str(cfg))
    layers = model.model.model
    assert len(layers) == 33, (run_id, len(layers))
    assert layers[32].f == [23, 27, 31], (run_id, layers[32].f)

    if run_id in {"B1", "B2"}:
        assert layers[15].__class__.__name__ == "UCRA1v2"
    else:
        assert layers[15].__class__.__name__ == "UCRA1v3"
    assert layers[19].__class__.__name__ == "UCRA2v3"

    model.model.eval()
    with torch.no_grad():
        _ = model.model(torch.randn(1, 3, 640, 640))
    print(run_id, "PASS", sum(p.numel() for p in model.model.parameters()))
