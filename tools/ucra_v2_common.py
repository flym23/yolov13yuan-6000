"""Shared immutable experiment contract for UCRA-v2 structural ablations."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['WANDB_DISABLED'] = 'true'
os.environ['PIN_MEMORY'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ultralytics
if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f'External Ultralytics: {ultralytics.__file__}')

MODELS = {
    'A0': 'yolov13.yaml',
    'A3': 'yolov13n-ucra-v2-a3-no-offset.yaml',
    'A4': 'yolov13n-ucra-v2-a4-full.yaml',
}
SETTINGS = dict(epochs=180, device=0, workers=2, amp=False, deterministic=True,
                plots=False, imgsz=640, batch=16)
DATA = Path('/home/room305/ZZF/URPC2020half/data.yaml')

def now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def model_path(stage):
    return ROOT / 'ultralytics/cfg/models/v13' / MODELS[stage]

def transfer_report(model):
    from ultralytics.nn.tasks import torch_safe_load
    ckpt, _ = torch_safe_load(str(ROOT / 'yolov13n.pt'))
    source = (ckpt.get('ema') or ckpt['model']).float().state_dict()
    target = model.model.state_dict()
    matched = sorted(k for k in target if k in source and target[k].shape == source[k].shape)
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    mismatch = {k: [list(source[k].shape), list(target[k].shape)] for k in target
                if k in source and source[k].shape != target[k].shape}
    report = dict(loaded_tensors=len(matched), target_tensors=len(target), missing_keys=missing,
                  unexpected_keys=unexpected, shape_mismatch_keys=mismatch,
                  pretrained_sha256=sha256(ROOT / 'yolov13n.pt'))
    # COCO -> 4 classes may resize classification tensors only. All other original layers must load.
    invalid = [k for k in missing if not k.startswith(('model.15.', 'model.19.'))]
    invalid += [k for k in mismatch if not k.startswith('model.32.cv3.')]
    invalid += [k for k in unexpected if not k.startswith('model.32.cv3.')]
    if invalid:
        raise RuntimeError(f'Unexpected pretrained incompatibility: {invalid}')
    model.load(str(ROOT / 'yolov13n.pt'))
    loaded = model.model.state_dict()
    import torch
    assert all(torch.equal(loaded[k], source[k]) for k in matched)
    return report

def require_run_root(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('run-root must be absolute')
    path = path.resolve()
    if path.parent != ROOT / 'runs' or not path.name.startswith('ucra_v2_'):
        raise ValueError(f'Unexpected run-root: {path}')
    return path
