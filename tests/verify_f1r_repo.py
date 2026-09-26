from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ultralytics
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f"External Ultralytics: {ultralytics.__file__}")
A4 = ROOT / "ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml"
C0 = ROOT / "ultralytics/cfg/models/v13/yolov13n-a4-f1r-c0-off.yaml"
C1 = ROOT / "ultralytics/cfg/models/v13/yolov13n-a4-f1r-c1.yaml"
C2 = ROOT / "ultralytics/cfg/models/v13/yolov13n-f1r-c2-head-only.yaml"


def clone_batch(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def compare_shared_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    for key, value in sa.items():
        assert key in sb, key
        assert value.shape == sb[key].shape, key
        assert torch.equal(value, sb[key]), key


def main():
    # A4 vs C0/C1: same-seed shared weights and initial loss must match.
    seed = 20260925
    torch.manual_seed(seed)
    a4 = YOLO(str(A4)).model

    torch.manual_seed(seed)
    c0 = YOLO(str(C0)).model

    torch.manual_seed(seed)
    c1 = YOLO(str(C1)).model

    for model in (a4, c0, c1):
        model.args = DEFAULT_CFG
        assert len(model.model) == 33
        assert model.model[15].__class__.__name__ == "UCRA1v2"
        assert model.model[19].__class__.__name__ == "UCRA2v2"
        assert model.model[32].f == [23, 27, 31]
        assert torch.equal(model.model[32].stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))

    assert c0.model[32].__class__.__name__ == "F1ReconcileDetect"
    assert c1.model[32].__class__.__name__ == "F1ReconcileDetect"

    compare_shared_state(a4, c0)
    compare_shared_state(a4, c1)

    batch = {
        "img": torch.randn(2, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor(
            [
                [0.30, 0.35, 0.04, 0.04],
                [0.70, 0.65, 0.12, 0.10],
            ]
        ),
    }

    a4.train()
    c0.train()
    c1.train()

    la4, ia4 = a4.loss(clone_batch(batch))
    lc0, ic0 = c0.loss(clone_batch(batch))
    lc1, ic1 = c1.loss(clone_batch(batch))

    assert torch.equal(la4, lc0)
    assert torch.equal(ia4, ic0)
    # Active C1 is also exact at initialization because gain_raw == 0.
    assert torch.equal(la4, lc1)
    assert torch.equal(ia4, ic1)

    # C2 verifies the head can also be isolated on the original backbone/neck.
    c2 = YOLO(str(C2)).model
    assert len(c2.model) == 33
    assert c2.model[15].__class__.__name__ == "Upsample"
    assert c2.model[19].__class__.__name__ == "Upsample"
    assert c2.model[32].__class__.__name__ == "F1ReconcileDetect"
    assert c2.model[32].f == [23, 27, 31]

    for model in (c1, c2):
        model.eval()
        with torch.no_grad():
            pred = model(torch.randn(1, 3, 640, 640))
        assert pred is not None

    print("F1-Reconcile repository verification passed.")


if __name__ == "__main__":
    main()
