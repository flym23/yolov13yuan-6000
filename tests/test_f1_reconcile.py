import torch
import torch.nn.functional as F

from ultralytics.nn.modules import Detect, F1ReconcileAdapter, F1ReconcileDetect


def test_round1_exact_init_and_math():
    Detect.legacy = False
    F1ReconcileDetect.legacy = False

    seed = 12345
    torch.manual_seed(seed)
    baseline = Detect(nc=4, ch=(64, 128, 256))
    baseline_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(seed)
    candidate = F1ReconcileDetect(
        nc=4,
        max_delta=0.75,
        loc_floor=0.50,
        reduction=4,
        detach_context=True,
        ch=(64, 128, 256),
    )
    candidate_rng = torch.random.get_rng_state().clone()

    # Adapter construction must not perturb the global RNG stream.
    assert torch.equal(baseline_rng, candidate_rng)

    base_params = dict(baseline.named_parameters())
    cand_params = dict(candidate.named_parameters())
    for name, param in base_params.items():
        assert name in cand_params, name
        assert torch.equal(param, cand_params[name]), name

    baseline.stride = torch.tensor([8.0, 16.0, 32.0])
    candidate.stride = torch.tensor([8.0, 16.0, 32.0])
    baseline.bias_init()
    candidate.bias_init()

    # Main Detect parameters remain bit-exact after bias initialization.
    cand_params = dict(candidate.named_parameters())
    for name, param in baseline.named_parameters():
        assert torch.equal(param, cand_params[name]), name

    baseline.train()
    candidate.train()
    p3 = torch.randn(2, 64, 32, 32)
    p4 = torch.randn(2, 128, 16, 16)
    p5 = torch.randn(2, 256, 8, 8)

    out_base = baseline([p3.clone(), p4.clone(), p5.clone()])
    out_cand = candidate([p3.clone(), p4.clone(), p5.clone()])
    for a, b in zip(out_base, out_cand):
        assert torch.equal(a, b)

    # Exact shared-gradient equality at initialization.
    baseline.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    target = [torch.randn_like(x) for x in out_base]
    loss_base = sum(F.mse_loss(x, y) for x, y in zip(out_base, target))
    loss_cand = sum(F.mse_loss(x, y) for x, y in zip(out_cand, target))
    assert torch.equal(loss_base, loss_cand)
    loss_base.backward()
    loss_cand.backward()

    cand_params = dict(candidate.named_parameters())
    for name, param in baseline.named_parameters():
        g0, g1 = param.grad, cand_params[name].grad
        assert (g0 is None) == (g1 is None), name
        if g0 is not None:
            assert torch.equal(g0, g1), name

    # Adapter mathematical guards.
    adapter = F1ReconcileAdapter(64, 128, 4, 16, max_delta=0.75)
    uniform = torch.zeros(1, 64, 4, 4)
    peaked = torch.zeros(1, 64, 4, 4)
    peaked.view(1, 4, 16, 4, 4)[:, :, 0] = 8.0
    assert adapter._localization_guard(peaked).mean() > adapter._localization_guard(uniform).mean()

    logits = torch.tensor([[-6.0, -2.0, 0.0, 2.0, 6.0]]).view(1, 5, 1, 1)
    ambiguity = adapter._miss_ambiguity(logits).flatten()
    assert ambiguity[0] < ambiguity[1] < ambiguity[2]
    assert ambiguity[4] < ambiguity[3] < ambiguity[2]

    parts = adapter.compute_components(
        torch.zeros(1, 64, 10, 10),
        torch.zeros(1, 128, 5, 5),
        torch.zeros(1, 4, 10, 10),
        torch.zeros(1, 64, 10, 10),
    )
    assert torch.isfinite(parts["output_logits"]).all()
    assert torch.equal(parts["delta_logits"], torch.zeros_like(parts["delta_logits"]))


def test_round2_staged_gradient_and_hard_bound():
    torch.manual_seed(77)
    module = F1ReconcileDetect(
        nc=4,
        max_delta=0.75,
        loc_floor=0.50,
        reduction=4,
        detach_context=True,
        ch=(64, 128, 256),
    )
    module.train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)

    aux_grad_seen = False
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        p3 = torch.randn(2, 64, 16, 16)
        p4 = torch.randn(2, 128, 8, 8)
        p5 = torch.randn(2, 256, 4, 4)
        outputs = module([p3, p4, p5])
        target = torch.randn_like(outputs[0])
        loss = F.mse_loss(outputs[0], target)
        assert torch.isfinite(loss)
        loss.backward()

        assert module.f1_adapter.gain_raw.grad is not None
        assert torch.isfinite(module.f1_adapter.gain_raw.grad).all()

        if step == 0:
            # Exact-zero gain: only gain_raw receives the first unlock signal.
            assert module.f1_adapter.gain_raw.grad.abs().sum() > 0
            assert module.f1_adapter.aux_logits.weight.grad is not None
            assert module.f1_adapter.aux_logits.weight.grad.abs().sum() == 0
        else:
            grad = module.f1_adapter.aux_logits.weight.grad
            if grad is not None and grad.abs().sum() > 0:
                aux_grad_seen = True

        optimizer.step()

    assert aux_grad_seen

    with torch.no_grad():
        module.f1_adapter.gain_raw.fill_(3.0)
        module.f1_adapter.aux_logits.weight.normal_(0.0, 0.5)

    p3 = torch.randn(1, 64, 16, 16)
    p4 = torch.randn(1, 128, 8, 8)
    box_logits = module.cv2[0](p3)
    base_logits = module.cv3[0](p3)
    parts = module.f1_adapter.compute_components(p3, p4, base_logits, box_logits)

    assert parts["delta_logits"].abs().max() <= 0.750001
    assert parts["support"].min() >= 0
    assert parts["support"].max() <= 1.000001
