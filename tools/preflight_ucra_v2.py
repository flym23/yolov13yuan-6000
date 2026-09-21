#!/usr/bin/env python3
"""Audit dataset, invariants, pretrained transfer and a CUDA backward before launch."""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ucra_v2_common import ROOT, DATA, MODELS, SETTINGS, model_path, now, require_run_root, sha256, transfer_report, write_json

def dataset_audit(data):
    import yaml
    from PIL import Image
    from ultralytics.data.utils import check_det_dataset, img2label_paths, IMG_FORMATS
    resolved = check_det_dataset(str(data), autodownload=False)
    assert resolved['nc'] == 4
    result = dict(data=str(data), dataset_yaml_sha256=sha256(data), names=resolved['names'], splits={})
    split_paths = {}
    for split in ('train', 'val', 'test'):
        if not resolved.get(split):
            continue
        values = resolved[split] if isinstance(resolved[split], list) else [resolved[split]]
        images = []
        for value in values:
            path = Path(value)
            if path.is_dir():
                images.extend(p.resolve() for p in path.rglob('*') if p.suffix[1:].lower() in IMG_FORMATS)
            elif path.is_file() and path.suffix == '.txt':
                images.extend((path.parent / line[2:]).resolve() if line.startswith('./') else Path(line).resolve()
                              for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
            else:
                raise FileNotFoundError(path)
        assert images and len(images) == len(set(images)), f'Empty/duplicate split: {split}'
        split_paths[split] = set(images)
        counts = [0] * 4
        original = [0] * 3
        resized = [0] * 3
        missing = empty = 0
        import hashlib
        digest = hashlib.sha256()
        for image in sorted(images):
            with Image.open(image) as im:
                width, height = im.size
                im.verify()
            label = Path(img2label_paths([str(image)])[0])
            digest.update(str(image).encode())
            digest.update(sha256(image).encode())
            if not label.exists():
                missing += 1
                continue
            content = label.read_text(encoding='utf-8')
            digest.update(content.encode())
            rows = [line.split() for line in content.splitlines() if line.strip()]
            empty += not bool(rows)
            for fields in rows:
                assert len(fields) == 5, f'Invalid detection label: {label}'
                cls, x, y, w, h = map(float, fields)
                assert all(math.isfinite(z) for z in (cls, x, y, w, h)), label
                assert cls.is_integer() and 0 <= cls < 4, label
                assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1, label
                counts[int(cls)] += 1
                area = w * width * h * height
                original[0 if area < 32**2 else 1 if area < 96**2 else 2] += 1
                area *= (640 / max(width, height)) ** 2
                resized[0 if area < 32**2 else 1 if area < 96**2 else 2] += 1
        assert missing == 0, f'Missing label files in {split}: {missing}'
        result['splits'][split] = dict(images=len(images), instances_per_class=counts,
             coco_sml_original=original, coco_sml_at_640=resized, missing_labels=missing,
             empty_labels=empty, content_sha256=digest.hexdigest(), paths=values)
    assert not split_paths['train'].intersection(split_paths['val']), 'Train/val leakage'
    result['val_equals_test'] = split_paths.get('test') == split_paths['val']
    result['evaluation_note'] = 'Existing val/test split preserved; val and test are the same held-out images.'
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    if (run / 'state.json').exists():
        raise FileExistsError('Run already registered; preflight cannot overwrite a live contract')
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.modules.ucra_v2 import UCRA1v2, UCRA2v2
    from ultralytics.utils.torch_utils import init_seeds, get_flops
    torch.set_num_threads(2)
    print('LOCAL_PACKAGE', str(ROOT), 'TORCH', torch.__version__, flush=True)
    audit = dataset_audit(DATA)
    write_json(run / 'train/dataset_audit.json', audit)
    print('DATASET_AUDIT', json.dumps(audit), flush=True)
    import runpy
    runpy.run_path(str(ROOT / 'tests/test_ucra_v2_rdas.py'), run_name='__main__')
    runpy.run_path(str(ROOT / 'tests/verify_ucra_v2_repo.py'), run_name='__main__')
    baseline_cfg = yaml.safe_load(model_path('A0').read_text(encoding='utf-8'))
    baseline_layers = baseline_cfg['backbone'] + baseline_cfg['head']
    reports = {}
    reference = None
    image = torch.linspace(0, 1, 3 * 640 * 640).reshape(1, 3, 640, 640)
    for stage in MODELS:
        cfg = yaml.safe_load(model_path(stage).read_text(encoding='utf-8'))
        layers = cfg['backbone'] + cfg['head']
        assert len(layers) == 33 and layers[-1][0] == [23, 27, 31] and cfg['nc'] == 4
        assert all(layers[i] == baseline_layers[i] for i in range(33) if i not in (15, 19))
        init_seeds(0, deterministic=True)
        model = YOLO(str(model_path(stage)), verbose=False)
        transfer = transfer_report(model)
        model.model.eval()
        with torch.no_grad():
            prediction = model.model(image)[0]
        assert torch.isfinite(prediction).all()
        if reference is None:
            reference = prediction.clone()
        else:
            assert torch.equal(prediction, reference), f'{stage}: full-model zero-start differs from A0'
        reports[stage] = dict(transfer=transfer, params=sum(p.numel() for p in model.model.parameters()),
                              gflops=get_flops(model.model, imgsz=640), forward_640=True,
                              exact_baseline_start=True, layers=33)
        print('MODEL_REPORT', stage, json.dumps(reports[stage]), flush=True)
        del model
    # Two optimizer steps exercise zero-start and subsequently nonzero alignment gradients.
    # This is a tensor-level preflight, not a shortened training experiment.
    assert torch.cuda.is_available()
    for cls in (UCRA1v2, UCRA2v2):
        for max_offset in (0.0, 0.25 if cls is UCRA1v2 else 0.5):
            module = cls(64, 32, max_offset=max_offset).cuda()
            optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
            for step in range(2):
                optimizer.zero_grad()
                deep = torch.randn(2, 64, 10, 10, device='cuda', requires_grad=True)
                lateral = torch.randn(2, 32, 20, 20, device='cuda', requires_grad=True)
                loss = (module([deep, lateral]) - torch.randn(2, 64, 20, 20, device='cuda')).square().mean()
                loss.backward()
                assert torch.isfinite(loss) and torch.isfinite(deep.grad).all()
                assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
                assert module.residual_out.weight.grad.abs().sum() > 0
                if step and max_offset:
                    assert module.offset_head[-1].weight.grad.abs().sum() > 0
                optimizer.step()
    defaults = yaml.safe_load((ROOT / 'ultralytics/cfg/default.yaml').read_text(encoding='utf-8'))
    keys = ('optimizer lr0 lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr '
            'box cls dfl nbs hsv_h hsv_s hsv_v degrees translate scale shear perspective flipud fliplr bgr '
            'mosaic mixup copy_paste copy_paste_mode close_mosaic cos_lr single_cls rect multi_scale '
            'fraction freeze cache save save_period val iou max_det').split()
    recipe = {k: defaults[k] for k in keys if k in defaults}
    recipe.update(SETTINGS)
    files = ['tools/ucra_v2_common.py', 'tools/train_ucra_v2_worker.py', 'tools/run_ucra_v2.py',
             'tools/preflight_ucra_v2.py', 'test.py', 'yolov13n.pt', 'ultralytics/cfg/default.yaml',
             'experiments/urpc2020half_ucra_v2_ablation.yaml']
    files += [str(p.relative_to(ROOT)) for p in (ROOT / 'ultralytics').rglob('*.py')]
    files += [str(model_path(stage).relative_to(ROOT)) for stage in MODELS]
    for stage in MODELS:
        snapshot = run / 'train/model_snapshots' / model_path(stage).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(model_path(stage).read_bytes())
    contract = dict(created_at=now(), data=str(DATA), dataset_yaml_sha256=sha256(DATA),
        recipe=recipe, early_stopping=False, inherited_patience_ignored=defaults.get('patience'),
        stages=list(MODELS), seeds=[0,1,2], phase2=False,
        initialization='YOLO(model_yaml).load(yolov13n.pt)',
        hashes={rel: sha256(ROOT / rel) for rel in sorted(set(files))})
    write_json(run / 'train/contract.json', contract)
    write_json(run / 'train/preflight.json', dict(status='passed', at=now(), models=reports,
               cuda_backward=True, dataset=True, module_tests=True))
    print('PREFLIGHT_PASSED', str(run), flush=True)

if __name__ == '__main__':
    main()
