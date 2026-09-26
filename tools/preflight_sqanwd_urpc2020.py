#!/usr/bin/env python3
"""Validate and freeze the full-URPC2020 SQANWD chain before launch."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.sqanwd_urpc2020_common import (
    DATA, MODELS, REFERENCE_RUN, REFERENCE_STAGES, ROOT, SEEDS, SETTINGS,
    UPSTREAM_STATE, model_path, now, require_run_root, sha256, transfer_report,
    write_json,
)


def source_files() -> list[Path]:
    selected = set((ROOT / 'ultralytics').rglob('*.py'))
    selected.update(ROOT / relative for relative in (
        'test.py', 'ultralytics/cfg/default.yaml',
        'ultralytics/cfg/models/v13/yolov13n-sqanwd.yaml',
        'ultralytics/cfg/models/v13/yolov13n-b1-sqanwd.yaml',
        'tools/ucra_v2_common.py', 'tools/run_ucra_v2.py',
        'tools/sqanwd_urpc2020_common.py', 'tools/preflight_sqanwd_urpc2020.py',
        'tools/train_sqanwd_urpc2020_worker.py', 'tools/run_sqanwd_urpc2020.py',
        'scripts/wait_sqanwd_urpc2020_after_ucra_urpc2020.sh',
        'scripts/run_sqanwd_urpc2020.sh', 'yolov13n.pt',
    ))
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    if run.exists():
        raise FileExistsError(f'Run ID already exists: {run}')
    if not DATA.is_file() or not (ROOT / 'yolov13n.pt').is_file():
        raise FileNotFoundError('Dataset YAML or yolov13n.pt missing')
    upstream = json.loads(UPSTREAM_STATE.read_text(encoding='utf-8'))
    if upstream.get('status') not in ('completed', 'failed', 'cancelled'):
        raise RuntimeError('Upstream state is not terminal')
    reference_contract_path = REFERENCE_RUN / 'train/contract.json'
    reference_contract = json.loads(reference_contract_path.read_text(encoding='utf-8'))
    if reference_contract.get('data') != str(DATA):
        raise RuntimeError('A0 reference used a different dataset')
    recipe = dict(reference_contract['recipe'])
    if 'patience' in recipe or any(recipe.get(key) != value for key, value in SETTINGS.items()):
        raise RuntimeError('A0 reference recipe conflicts with requested settings')

    import yaml
    from ultralytics import YOLO

    baseline_models = {
        'S2': ROOT / 'ultralytics/cfg/models/v13/yolov13.yaml',
        'S3': ROOT / 'ultralytics/cfg/models/v13/yolov13n-ucra-v3-b1-context-only.yaml',
    }
    transfers = {}
    for stage in MODELS:
        candidate = yaml.safe_load(model_path(stage).read_text(encoding='utf-8'))
        baseline = yaml.safe_load(baseline_models[stage].read_text(encoding='utf-8'))
        candidate.pop('sqanwd', None)
        baseline['nc'] = candidate['nc']
        if candidate != baseline:
            raise RuntimeError(f'{stage} architecture changed beyond SQANWD')
        model = YOLO(str(model_path(stage)))
        if model.model.yaml.get('sqanwd', {}).get('enabled') is not True:
            raise RuntimeError(f'{stage} SQANWD config did not reach the built model')
        transfers[stage] = transfer_report(model)
        del model

    files = source_files()
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise FileNotFoundError(f'Invalid source dependency: {path}')
    source_hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in files}

    train = run / 'train'
    references = run / 'test/references'
    snapshots = train / 'snapshots'
    references.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    snapshot_hashes = {}
    for stage in REFERENCE_STAGES:
        original = REFERENCE_RUN / 'test' / f'{stage}_AllSeed_summary.json'
        contents = json.loads(original.read_text(encoding='utf-8'))
        if {row.get('seed') for row in contents['seeds']} != set(SEEDS):
            raise RuntimeError(f'Incomplete {stage} reference')
        target = references / original.name
        shutil.copy2(original, target)
        snapshot_hashes[str(target.relative_to(run))] = sha256(target)
    for name, original in (
        ('data.yaml', DATA),
        ('S2.yaml', model_path('S2')),
        ('S3.yaml', model_path('S3')),
        ('A0_contract.json', reference_contract_path),
    ):
        target = snapshots / name
        shutil.copy2(original, target)
        snapshot_hashes[str(target.relative_to(run))] = sha256(target)

    contract = dict(
        data=str(DATA), dataset_yaml_sha256=sha256(DATA), stages=list(MODELS),
        seeds=list(SEEDS), recipe=recipe, early_stopping=False,
        full_dataset_only=True, s3_unconditional=True,
        upstream_state=str(UPSTREAM_STATE),
        reference_run=str(REFERENCE_RUN),
        source_hashes=source_hashes, snapshot_hashes=snapshot_hashes,
        pretrained_sha256=sha256(ROOT / 'yolov13n.pt'),
        created_at=now(),
    )
    write_json(train / 'contract.json', contract)
    write_json(train / 'preflight.json', dict(status='passed', run_root=str(run),
        data=str(DATA), upstream_status=upstream['status'],
        upstream_failure_reason=upstream.get('failure_reason'),
        transfers=transfers, checked_at=now()))
    print('SQANWD_PREFLIGHT_PASSED', run, json.dumps({key: value['loaded_tensors']
        for key, value in transfers.items()}), flush=True)


if __name__ == '__main__':
    main()
