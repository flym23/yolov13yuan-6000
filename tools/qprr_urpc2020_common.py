"""Frozen full-URPC2020 contract and metrics for QPRR D1 -> D2 -> D3."""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ucra_v2_common import SETTINGS, now, sha256, transfer_report, write_json

import ultralytics

if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f'External Ultralytics: {ultralytics.__file__}')

DATA = Path('/home/room305/ZZF/URPC2020/data.yaml')
BASELINE_RUN = ROOT / 'runs/ucra_urpc2020_20260921_r1'
F1R_RUN = ROOT / 'runs/f1r_urpc2020_20260925_r1'
UPSTREAM_STATE = F1R_RUN / 'state.json'
MODELS = {
    'D1': 'yolov13n-a4-qprr-d1.yaml',
    'D2': 'yolov13n-b1-qprr-d2.yaml',
    'D3': 'yolov13n-qprr-d3-baseline.yaml',
}
REFERENCE_STAGES = {
    'A0': (BASELINE_RUN, 'A0'),
    'A4': (BASELINE_RUN, 'A4'),
    'B1': (BASELINE_RUN, 'B1'),
    'C1_F1R': (F1R_RUN, 'C1'),
    'C2_F1R': (F1R_RUN, 'C2'),
}
SEEDS = (0, 1, 2)


def model_path(stage: str) -> Path:
    return ROOT / 'ultralytics/cfg/models/v13' / MODELS[stage]


def require_run_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('run-root must be absolute')
    path = path.resolve()
    if (path.parent != ROOT / 'runs' or not
            re.fullmatch(r'qprr_urpc2020_[A-Za-z0-9][A-Za-z0-9_-]{2,80}', path.name)):
        raise ValueError(f'Unexpected QPRR run-root: {path}')
    return path


def verify_contract(run: Path) -> dict:
    run = require_run_root(str(run))
    frozen = json.loads((run / 'train/contract.json').read_text(encoding='utf-8'))
    if (frozen['data'] != str(DATA) or frozen['stages'] != list(MODELS)
            or frozen['seeds'] != list(SEEDS) or frozen['unconditional'] is not True
            or frozen['full_dataset_only'] is not True):
        raise RuntimeError('Dataset/stage/seed contract mismatch')
    if (frozen['early_stopping'] is not False or 'patience' in frozen['recipe']
            or any(frozen['recipe'].get(k) != v for k, v in SETTINGS.items())):
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


def f1(p: float, r: float) -> float:
    return 2.0 * p * r / (p + r) if p + r > 0 else 0.0


def statistics_for(rows: list[dict]) -> dict:
    if {row['seed'] for row in rows} != set(SEEDS):
        raise RuntimeError('Three seed rows required')
    keys = [key for key in rows[0] if key not in ('stage', 'seed')]
    return {key: dict(mean=statistics.mean(row[key] for row in rows),
                      std=statistics.stdev(row[key] for row in rows),
                      best=max(row[key] for row in rows),
                      worst=min(row[key] for row in rows)) for key in keys}


def enrich_reference(row: dict, stage: str) -> dict:
    result = dict(row)
    result['stage'] = stage
    result['F1'] = f1(result['P'], result['R'])
    return result


def summarize(run: Path, stage: str) -> tuple[list[dict], dict]:
    from tools.run_ucra_v2 import load, write_csv

    rows = []
    for seed in SEEDS:
        folder = run / 'test' / f'seed{seed}' / stage
        result = load(folder / 'summary_metrics.json')
        scale = load(folder / 'scale_ap_metrics.json')['metrics']
        train = load(run / 'train' / f'seed{seed}' / stage / 'train_complete.json')
        metrics = result['metrics']
        p, r = metrics['metrics/precision(B)'], metrics['metrics/recall(B)']
        rows.append(dict(stage=stage, seed=seed, P=p, R=r, F1=f1(p, r),
            mAP50=metrics['metrics/mAP50(B)'], mAP50_95=metrics['metrics/mAP50-95(B)'],
            AP_S=scale['APS']/100, AP_M=scale['APM']/100, AP_L=scale['APL']/100,
            params=result['model']['parameters'], gflops=result['model']['gflops'],
            best_epoch=train['best_epoch']))
    if not all(math.isfinite(value) for row in rows for key, value in row.items() if key != 'stage'):
        raise RuntimeError(f'Nonfinite {stage} metrics')
    stats = statistics_for(rows)
    write_json(run / 'test' / f'{stage}_AllSeed_summary.json',
               dict(seeds=rows, statistics=stats, std_ddof=1))
    write_csv(run / 'test' / f'{stage}_AllSeed_summary.csv', rows)
    return rows, stats
