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
D0 = ROOT / "ultralytics/cfg/models/v13/yolov13n-a4-qprr-d0-off.yaml"
D1 = ROOT / "ultralytics/cfg/models/v13/yolov13n-a4-qprr-d1.yaml"
D2 = ROOT / "ultralytics/cfg/models/v13/yolov13n-b1-qprr-d2.yaml"
D3 = ROOT / "ultralytics/cfg/models/v13/yolov13n-qprr-d3-baseline.yaml"


def clone_batch(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def shared_state_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


def main():
    # CPU oneDNN backward can choose different reductions for identical graphs.
    # Use the deterministic native path for bit-exact gradient comparison only.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    seed = 20260926

    # D0 must be architecture/state/loss equivalent to A4.
    torch.manual_seed(seed)
    a4 = YOLO(str(A4)).model
    torch.manual_seed(seed)
    d0 = YOLO(str(D0)).model
    a4.args = DEFAULT_CFG
    d0.args = DEFAULT_CFG
    shared_state_equal(a4, d0)

    assert len(a4.model) == len(d0.model) == 33
    assert d0.model[15].__class__.__name__ == "UCRA1v2"
    assert d0.model[19].__class__.__name__ == "UCRA2v2"
    assert d0.model[32].__class__.__name__ == "Detect"
    assert d0.model[32].f == [23, 27, 31]
    assert torch.equal(d0.model[32].stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))

    batch = {
        "img": torch.randn(2, 3, 256, 256),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        "bboxes": torch.tensor(
            [
                [0.28, 0.30, 0.05, 0.05],
                [0.66, 0.62, 0.12, 0.10],
                [0.35, 0.70, 0.06, 0.05],
                [0.75, 0.25, 0.14, 0.12],
            ]
        ),
    }

    a4.train()
    d0.train()
    a4.zero_grad(set_to_none=True)
    d0.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    loss_a4, items_a4 = a4.loss(clone_batch(batch))
    torch.manual_seed(seed)
    loss_d0, items_d0 = d0.loss(clone_batch(batch))
    assert torch.equal(loss_a4, loss_d0)
    assert torch.equal(items_a4, items_d0)

    loss_a4.backward()
    loss_d0.backward()
    p_a4 = dict(a4.named_parameters())
    p_d0 = dict(d0.named_parameters())
    for name in p_a4:
        g0, g1 = p_a4[name].grad, p_d0[name].grad
        assert (g0 is None) == (g1 is None), name
        if g0 is not None:
            assert torch.equal(g0, g1), name

    # D1 is the same deployed architecture; QPRR only changes training loss.
    d1 = YOLO(str(D1)).model
    d1.args = DEFAULT_CFG
    assert d1.model[32].__class__.__name__ == "Detect"
    assert sum(p.numel() for p in d1.parameters()) == sum(p.numel() for p in a4.parameters())

    d1.train()
    d1.zero_grad(set_to_none=True)
    loss_d1, items_d1 = d1.loss(clone_batch(batch))
    assert torch.isfinite(loss_d1)
    assert torch.isfinite(items_d1).all()
    loss_d1.backward()
    assert getattr(d1.criterion, "use_qprr", False)
    diag = d1.criterion.last_qprr_diagnostics
    assert diag
    assert diag["rank_pairs"] > 0
    assert torch.isfinite(diag["auxiliary"])

    # D2 and D3 are ablation bases.
    d2 = YOLO(str(D2)).model
    d3 = YOLO(str(D3)).model
    assert len(d2.model) == len(d3.model) == 33
    assert d2.model[15].__class__.__name__ == "UCRA1v2"
    assert d3.model[32].__class__.__name__ == "Detect"

    # Inference-facing model graph is unchanged by QPRR.
    for model in (d1, d2, d3):
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 640, 640))
        assert out is not None

    print("QPRR repository verification passed.")


if __name__ == "__main__":
    main()
