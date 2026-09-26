"""Contract for five fresh, three-seed UCRA experiments on full URPC2020."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.ucra_v2_common import SETTINGS, now, sha256, transfer_report, write_json

DATA = Path('/home/room305/ZZF/URPC2020/data.yaml')
MODELS = {
    'A0': 'yolov13.yaml',
    'A4': 'yolov13n-ucra-v2-a4-full.yaml',
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
    if path.parent != ROOT / 'runs' or not re.fullmatch(r'ucra_urpc2020_[A-Za-z0-9][A-Za-z0-9_-]{2,80}', path.name):
        raise ValueError(f'Unexpected run-root: {path}')
    return path


def verify_contract(run):
    frozen = json.loads((run / 'train/contract.json').read_text(encoding='utf-8'))
    if frozen['data'] != str(DATA) or frozen['stages'] != list(MODELS) or frozen['seeds'] != [0, 1, 2]:
        raise RuntimeError('Dataset/stage/seed contract mismatch')
    for rel, expected in frozen['hashes'].items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise RuntimeError(f'Contract file changed: {rel}')
    if sha256(DATA) != frozen['dataset_yaml_sha256']:
        raise RuntimeError('Dataset YAML changed after preflight')
    identity = json.loads((run / 'upstream_process.json').read_text(encoding='utf-8'))
    if identity != frozen['upstream_process']:
        raise RuntimeError('Upstream process identity changed')
    if any(frozen['recipe'].get(k) != v for k, v in SETTINGS.items()):
        raise RuntimeError('Requested training parameters changed')
    if 'patience' in frozen['recipe'] or frozen['early_stopping']:
        raise RuntimeError('Early stopping must be disabled')
    return frozen
