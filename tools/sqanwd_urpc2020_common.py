"""Frozen full-URPC2020 contract for the S2 -> S3 SQANWD experiment."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ucra_v2_common import SETTINGS, now, sha256, transfer_report, write_json

import ultralytics

if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f'External Ultralytics: {ultralytics.__file__}')

DATA = Path('/home/room305/ZZF/URPC2020/data.yaml')
REFERENCE_RUN = ROOT / 'runs/ucra_urpc2020_20260921_r1'
UPSTREAM_STATE = REFERENCE_RUN / 'state.json'
MODELS = {
    'S2': 'yolov13n-sqanwd.yaml',
    'S3': 'yolov13n-b1-sqanwd.yaml',
}
REFERENCE_STAGES = ('A0', 'B1')
SEEDS = (0, 1, 2)


def model_path(stage: str) -> Path:
    return ROOT / 'ultralytics/cfg/models/v13' / MODELS[stage]


def require_run_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('run-root must be absolute')
    path = path.resolve()
    if (path.parent != ROOT / 'runs'
            or not re.fullmatch(r'sqanwd_urpc2020_[A-Za-z0-9][A-Za-z0-9_-]{2,80}', path.name)):
        raise ValueError(f'Unexpected SQANWD run-root: {path}')
    return path


def verify_contract(run: Path) -> dict:
    run = require_run_root(str(run))
    frozen = json.loads((run / 'train/contract.json').read_text(encoding='utf-8'))
    if (frozen['data'] != str(DATA) or frozen['stages'] != list(MODELS)
            or frozen['seeds'] != list(SEEDS) or frozen['s3_unconditional'] is not True
            or frozen['full_dataset_only'] is not True):
        raise RuntimeError('Dataset/stage/seed contract mismatch')
    if (frozen['early_stopping'] is not False or 'patience' in frozen['recipe']
            or any(frozen['recipe'].get(key) != value for key, value in SETTINGS.items())):
        raise RuntimeError('Requested training settings changed')
    if sha256(DATA) != frozen['dataset_yaml_sha256']:
        raise RuntimeError('Dataset YAML changed')
    for relative, expected in frozen['source_hashes'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise RuntimeError(f'Source dependency changed: {relative}')
    for relative, expected in frozen['snapshot_hashes'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or sha256(path) != expected:
            raise RuntimeError(f'Experiment snapshot changed: {relative}')
    return frozen
