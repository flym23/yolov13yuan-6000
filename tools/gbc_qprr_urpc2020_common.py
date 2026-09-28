"""Frozen full-URPC2020 contract and metric helpers for GBC-QPRR."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ucra_v2_common import SETTINGS, transfer_report

import ultralytics

if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f"External Ultralytics: {ultralytics.__file__}")

DATA = Path('/home/room305/ZZF/URPC2020/data.yaml')
PRETRAINED = ROOT / 'yolov13n.pt'
BASELINE_RUN = ROOT / 'runs/ucra_urpc2020_20260921_r1'
QPRR_RUN = ROOT / 'runs/qprr_urpc2020_20260926_r1'
UPSTREAM_STATE = QPRR_RUN / 'state.json'
MODELS = {
    'E1': 'yolov13n-gbc-qprr-e1-instance.yaml',
    'E2': 'yolov13n-gbc-qprr-e2-full.yaml',
    'E3': 'yolov13n-a4-gbc-qprr-e3.yaml',
}
SEEDS = (0, 1, 2)
STAGE_ORDER = tuple(MODELS)
RUN_ID = '20260928_r1'

REFERENCE_STAGES = {
    'A0': (BASELINE_RUN, 'A0'),
    'A4': (BASELINE_RUN, 'A4'),
    'D1_QPRR': (QPRR_RUN, 'D1'),
    'D3_QPRR': (QPRR_RUN, 'D3'),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_path(stage: str) -> Path:
    return ROOT / 'ultralytics/cfg/models/v13' / MODELS[stage]


def require_run_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('run-root must be absolute')
    path = path.resolve()
    if (path.parent != ROOT / 'runs' or
            not re.fullmatch(r'gbc_qprr_urpc2020_[A-Za-z0-9][A-Za-z0-9_-]{2,80}', path.name)):
        raise ValueError(f'Unexpected GBC-QPRR run-root: {path}')
    return path


def verify_contract(run: Path) -> dict:
    run = require_run_root(run)
    frozen = load(run / 'train/contract.json')
    if (frozen.get('data') != str(DATA) or frozen.get('stages') != list(STAGE_ORDER)
            or frozen.get('seeds') != list(SEEDS) or frozen.get('unconditional') is not True
            or frozen.get('full_dataset_only') is not True):
        raise RuntimeError('Dataset/stage/seed contract mismatch')
    recipe = frozen.get('recipe', {})
    if (frozen.get('early_stopping') is not False or 'patience' in recipe
            or any(recipe.get(key) != value for key, value in SETTINGS.items())):
        raise RuntimeError('Requested training settings changed')
    if sha256(DATA) != frozen['dataset_yaml_sha256']:
        raise RuntimeError('Dataset YAML changed after preflight')
    if sha256(PRETRAINED) != frozen['pretrained_sha256']:
        raise RuntimeError('Pretrained weights changed after preflight')
    for relative, expected in frozen['source_hashes'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise RuntimeError(f'Source dependency changed: {relative}')
    for relative, expected in frozen['snapshot_hashes'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or sha256(path) != expected:
            raise RuntimeError(f'Experiment snapshot changed: {relative}')
    return frozen


def f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def statistics_for(rows: list[dict]) -> dict:
    if {row['seed'] for row in rows} != set(SEEDS):
        raise RuntimeError('Exactly three seed rows are required')
    metrics = [key for key in rows[0] if key not in ('stage', 'seed')]
    return {metric: dict(mean=statistics.mean(row[metric] for row in rows),
                         std=statistics.stdev(row[metric] for row in rows),
                         best=max(row[metric] for row in rows),
                         worst=min(row[metric] for row in rows)) for metric in metrics}


def enrich_reference(row: dict, alias: str) -> dict:
    result = dict(row)
    result['stage'] = alias
    result['F1'] = f1(float(result['P']), float(result['R']))
    return result


def summarize(run: Path, stage: str) -> tuple[list[dict], dict]:
    from tools.run_ucra_v2 import write_csv

    rows = []
    for seed in SEEDS:
        folder = run / 'test' / f'seed{seed}' / stage
        result = load(folder / 'summary_metrics.json')
        scale = load(folder / 'scale_ap_metrics.json')['metrics']
        train = load(run / 'train' / f'seed{seed}' / stage / 'train_complete.json')
        metrics = result['metrics']
        precision = float(metrics['metrics/precision(B)'])
        recall = float(metrics['metrics/recall(B)'])
        rows.append(dict(stage=stage, seed=seed, P=precision, R=recall,
            F1=f1(precision, recall), mAP50=float(metrics['metrics/mAP50(B)']),
            mAP50_95=float(metrics['metrics/mAP50-95(B)']), AP_S=float(scale['APS']) / 100,
            AP_M=float(scale['APM']) / 100, AP_L=float(scale['APL']) / 100,
            params=int(result['model']['parameters']), gflops=float(result['model']['gflops']),
            best_epoch=int(train['best_epoch'])))
    if not all(math.isfinite(value) for row in rows for key, value in row.items()
               if key not in ('stage',)):
        raise RuntimeError(f'Nonfinite metrics in {stage}')
    stats = statistics_for(rows)
    write_json(run / 'test' / f'{stage}_AllSeed_summary.json',
               dict(seeds=rows, statistics=stats, std_ddof=1))
    write_csv(run / 'test' / f'{stage}_AllSeed_summary.csv', rows)
    return rows, stats
