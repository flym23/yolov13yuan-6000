from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG

if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f"External Ultralytics: {ultralytics.__file__}")
A0 = ROOT / "ultralytics/cfg/models/v13/yolov13.yaml"
E0 = ROOT / "ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e0-off.yaml"
E1 = ROOT / "ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e1-instance.yaml"
E2 = ROOT / "ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e2-full.yaml"
E3 = ROOT / "ultralytics/cfg/models/v13/yolov13n-a4-gbc-qprr-e3.yaml"


def clone_batch(batch):
    return {k:(v.clone() if torch.is_tensor(v) else v) for k,v in batch.items()}


def state_equal(a,b):
    sa,sb=a.state_dict(),b.state_dict()
    assert sa.keys()==sb.keys()
    for k in sa:
        assert torch.equal(sa[k],sb[k]),k


def main():
    seed=20260928
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(seed)
    a0=YOLO(str(A0)).model
    torch.manual_seed(seed)
    e0=YOLO(str(E0)).model
    a0.args = DEFAULT_CFG
    e0.args = DEFAULT_CFG
    state_equal(a0,e0)

    assert len(a0.model)==len(e0.model)==33
    assert e0.model[-1].__class__.__name__=="Detect"
    assert e0.model[-1].f==[23,27,31]
    assert torch.equal(e0.model[-1].stride.cpu(),torch.tensor([8.,16.,32.]))

    batch={
        "img":torch.randn(2,3,256,256),
        "batch_idx":torch.tensor([0.,0.,0.,1.,1.]),
        "cls":torch.tensor([[0.],[0.],[1.],[2.],[3.]]),
        "bboxes":torch.tensor([
            [0.25,0.25,0.08,0.08],
            [0.55,0.25,0.07,0.07],  # repeated same-class GT in image0
            [0.70,0.65,0.14,0.12],
            [0.35,0.70,0.08,0.07],
            [0.75,0.25,0.14,0.12],
        ]),
    }

    a0.train(); e0.train()
    a0.zero_grad(set_to_none=True); e0.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    la,ia=a0.loss(clone_batch(batch))
    torch.manual_seed(seed)
    le,ie=e0.loss(clone_batch(batch))
    assert torch.equal(la,le)
    assert torch.equal(ia,ie)
    la.backward(); le.backward()
    pa=dict(a0.named_parameters()); pe=dict(e0.named_parameters())
    for k in pa:
        ga,ge=pa[k].grad,pe[k].grad
        assert (ga is None)==(ge is None),k
        if ga is not None:
            assert torch.equal(ga,ge),k

    # E1/E2: training-only loss path, no model parameters or inference graph added.
    e1=YOLO(str(E1)).model
    e2=YOLO(str(E2)).model
    assert sum(p.numel() for p in e1.parameters())==sum(p.numel() for p in a0.parameters())
    assert sum(p.numel() for p in e2.parameters())==sum(p.numel() for p in a0.parameters())

    for model in (e1,e2):
        model.args = DEFAULT_CFG
        model.train()
        model.zero_grad(set_to_none=True)
        loss,items=model.loss(clone_batch(batch))
        assert torch.isfinite(loss)
        assert torch.isfinite(items).all()
        loss.backward()
        assert getattr(model.criterion,"use_gbc_qprr",False)
        diag=model.criterion.last_gbc_qprr_diagnostics
        assert diag
        assert diag["valid_gts"] >= 5
        assert torch.isfinite(diag["auxiliary"])

    # The synthetic unit test separately guarantees nonzero ranking/coverage paths;
    # this model-level batch verifies criterion wiring and finite backward behavior.

    # E3 uses A4 neck but still original Detect.
    e3=YOLO(str(E3)).model
    e3.args = DEFAULT_CFG
    assert len(e3.model)==33
    assert e3.model[15].__class__.__name__=="UCRA1v2"
    assert e3.model[19].__class__.__name__=="UCRA2v2"
    assert e3.model[-1].__class__.__name__=="Detect"

    # Inference-facing graph is unchanged by the training-only loss.
    for model in (e1,e2,e3):
        model.eval()
        with torch.no_grad():
            out=model(torch.randn(1,3,640,640))
        assert out is not None

    print("GBC-QPRR repository verification passed.")


if __name__=="__main__":
    main()
