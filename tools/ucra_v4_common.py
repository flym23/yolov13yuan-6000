"""Frozen UCRA-v4/GIPR contract for the three-seed half-URPC2020 experiment."""
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

UPSTREAM_STATE = ROOT / 'runs/ucra_urpc2020_20260921_r1/state.json'
REFERENCE_RUNS = {
    'A0': ROOT / 'runs/ucra_v2_20260920_r1',
    'A4': ROOT / 'runs/ucra_v2_20260920_r1',
    'B2': ROOT / 'runs/ucra_v3_20260921_r1',
}
MODELS = {
    'C0': 'yolov13n-ucra-v4-c0-equivalence.yaml',
    'C1': 'yolov13n-ucra-v4-c1-gipr.yaml',
    'C2': 'yolov13n-ucra-v4-c2-conservative.yaml',
}


def model_path(stage):
    return ROOT / 'ultralytics/cfg/models/v13' / MODELS[stage]


def require_run_root(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('run-root must be absolute')
    path = path.resolve()
    if path.parent != ROOT / 'runs' or not re.fullmatch(r'ucra_v4_[A-Za-z0-9][A-Za-z0-9_-]{2,80}', path.name):
        raise ValueError(f'Unexpected run-root: {path}')
    return path


def verify_contract(run):
    frozen = json.loads((run / 'train/contract.json').read_text(encoding='utf-8'))
    if frozen['data'] != str(DATA) or frozen['stages'] != list(MODELS) or frozen['seeds'] != [0, 1, 2]:
        raise RuntimeError('Dataset, stage or seed contract mismatch')
    if frozen['c2_unconditional'] is not True or frozen['early_stopping'] is not False:
        raise RuntimeError('Experiment execution contract mismatch')
    if any(frozen['recipe'].get(k) != v for k, v in SETTINGS.items()) or 'patience' in frozen['recipe']:
        raise RuntimeError('Requested training parameters changed')
    if sha256(DATA) != frozen['dataset_yaml_sha256']:
        raise RuntimeError('Dataset YAML changed after preflight')
    for rel, expected in frozen['hashes'].items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise RuntimeError(f'Contract dependency changed: {rel}')
    for rel, expected in frozen['reference_hashes'].items():
        path = (run / rel).resolve()
        if not path.is_relative_to(run) or sha256(path) != expected:
            raise RuntimeError(f'Reference snapshot changed: {rel}')
    return frozen
