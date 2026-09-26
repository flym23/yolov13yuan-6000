"""UCRA-v3 contract; reuse the verified v2 IO and transfer audit without modifying v2."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.ucra_v2_common import DATA, SETTINGS, now, sha256, transfer_report, write_json
import ultralytics
if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f'External Ultralytics: {ultralytics.__file__}')

REFERENCE_RUN = ROOT / 'runs/ucra_v2_20260920_r1'
MODELS = {
    'B1': 'yolov13n-ucra-v3-b1-context-only.yaml',
    'B2': 'yolov13n-ucra-v3-b2-hybrid-full.yaml',
    'B3': 'yolov13n-ucra-v3-b3-full.yaml',
}

def model_path(stage):
    return ROOT / 'ultralytics/cfg/models/v13' / MODELS[stage]

def require_run_root(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('run-root must be absolute')
    path = path.resolve()
    if path.parent != ROOT / 'runs' or not re.fullmatch(r'ucra_v3_[A-Za-z0-9][A-Za-z0-9_-]{2,80}', path.name):
        raise ValueError(f'Unexpected run-root: {path}')
    return path

def verify_contract(run):
    frozen = json.loads((run / 'train/contract.json').read_text(encoding='utf-8'))
    for rel, expected in frozen['hashes'].items():
        if sha256(ROOT / rel) != expected:
            raise RuntimeError(f'Contract file changed after preflight: {rel}')
    if sha256(Path(frozen['data'])) != frozen['dataset_yaml_sha256']:
        raise RuntimeError('Dataset YAML changed after preflight')
    for rel, expected in frozen['reference_hashes'].items():
        if sha256(run / rel) != expected:
            raise RuntimeError(f'Reference snapshot changed: {rel}')
    if frozen['recipe'] != frozen['parent_recipe']:
        raise RuntimeError('Training recipe differs from A0/A4 reference')
    if any(frozen['recipe'].get(key) != value for key, value in SETTINGS.items()):
        raise RuntimeError('Explicit user parameters are not satisfied')
    if 'patience' in frozen['recipe'] or frozen['early_stopping']:
        raise RuntimeError('Early stopping must remain disabled')
    return frozen
