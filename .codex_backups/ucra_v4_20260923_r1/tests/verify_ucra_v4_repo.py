from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "C0": ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v4-c0-equivalence.yaml",
    "C1": ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v4-c1-gipr.yaml",
    "C2": ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v4-c2-conservative.yaml",
}

for run_id, cfg in MODELS.items():
    model = YOLO(str(cfg))
    layers = model.model.model
    assert len(layers) == 33, (run_id, len(layers))
    assert layers[15].__class__.__name__ == "UCRA1v2"
    assert layers[19].__class__.__name__ == "UCRA2v4"
    assert layers[32].f == [23, 27, 31], (run_id, layers[32].f)

    model.model.eval()
    with torch.no_grad():
        _ = model.model(torch.randn(1, 3, 640, 640))

    print(
        run_id,
        "PASS",
        "params=",
        sum(p.numel() for p in model.model.parameters()),
    )
