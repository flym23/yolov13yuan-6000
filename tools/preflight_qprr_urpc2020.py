#!/usr/bin/env python3
"""Validate QPRR integration and freeze its full-URPC2020 chain contract."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.qprr_urpc2020_common import (
    BASELINE_RUN, DATA, F1R_RUN, MODELS, REFERENCE_STAGES, SEEDS, SETTINGS,
    UPSTREAM_STATE, model_path, now, require_run_root, sha256,
    transfer_report, write_json,
)


def source_files() -> list[Path]:
    selected = set((ROOT / 'ultralytics').rglob('*.py'))
    selected.update(ROOT / relative for relative in (
        'test.py', 'ultralytics/cfg/default.yaml',
        'ultralytics/cfg/models/v13/yolov13n-a4-qprr-d0-off.yaml',
        'ultralytics/cfg/models/v13/yolov13n-a4-qprr-d1.yaml',
        'ultralytics/cfg/models/v13/yolov13n-b1-qprr-d2.yaml',
        'ultralytics/cfg/models/v13/yolov13n-qprr-d3-baseline.yaml',
        'tools/ucra_v2_common.py', 'tools/run_ucra_v2.py',
        'tools/qprr_urpc2020_common.py', 'tools/preflight_qprr_urpc2020.py',
        'tools/train_qprr_urpc2020_worker.py', 'tools/run_qprr_urpc2020.py',
        'tests/verify_qprr_repo.py', 'tests/test_qprr.py',
        'scripts/wait_qprr_urpc2020_after_f1r.sh',
        'scripts/run_qprr_urpc2020.sh', 'yolov13n.pt',
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
    baseline_contract_path = BASELINE_RUN / 'train/contract.json'
    baseline_contract = json.loads(baseline_contract_path.read_text(encoding='utf-8'))
    if baseline_contract.get('data') != str(DATA):
        raise RuntimeError('Reference used another dataset')
    recipe = dict(baseline_contract['recipe'])
    if 'patience' in recipe or any(recipe.get(k) != v for k, v in SETTINGS.items()):
        raise RuntimeError('Reference recipe conflicts with requested settings')

    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    baseline_models = {
        'D0': ROOT / 'ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml',
        'D1': ROOT / 'ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml',
        'D2': ROOT / 'ultralytics/cfg/models/v13/yolov13n-ucra-v3-b1-context-only.yaml',
        'D3': ROOT / 'ultralytics/cfg/models/v13/yolov13.yaml',
    }
    transfers = {}
    for stage in baseline_models:
        path = (ROOT / 'ultralytics/cfg/models/v13/yolov13n-a4-qprr-d0-off.yaml'
                if stage == 'D0' else model_path(stage))
        candidate = yaml.safe_load(path.read_text(encoding='utf-8'))
        baseline = yaml.safe_load(baseline_models[stage].read_text(encoding='utf-8'))
        qprr_cfg = candidate.pop('qprr', None)
        if candidate != baseline or not isinstance(qprr_cfg, dict):
            raise RuntimeError(f'{stage} changes architecture beyond QPRR')
        if (qprr_cfg.get('enabled') is not True or
                (float(qprr_cfg['gain']) > 0.0) != (stage != 'D0')):
            raise RuntimeError(f'{stage} has unexpected QPRR gain')
        if stage == 'D0':
            continue
        model = YOLO(str(path))
        if type(model.model.model[-1]) is not Detect:
            raise RuntimeError(f'{stage} inference head changed')
        if model.model.yaml.get('sqanwd') or model.model.yaml.get('f1r'):
            raise RuntimeError(f'{stage} enables an unrelated method')
        transfers[stage] = transfer_report(model)
        del model

    subprocess.run([sys.executable, str(ROOT / 'tests/verify_qprr_repo.py')],
                   cwd='/tmp', check=True, timeout=300)
    files = source_files()
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise FileNotFoundError(f'Invalid source dependency: {path}')
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in files}

    train = run / 'train'
    references = run / 'test/references'
    snapshots = train / 'snapshots'
    references.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    snapshot_hashes = {}
    for alias, (source_run, source_stage) in REFERENCE_STAGES.items():
        original = source_run / 'test' / f'{source_stage}_AllSeed_summary.json'
        contents = json.loads(original.read_text(encoding='utf-8'))
        if {row.get('seed') for row in contents['seeds']} != set(SEEDS):
            raise RuntimeError(f'Incomplete {alias} reference')
        target = references / f'{alias}_AllSeed_summary.json'
        shutil.copy2(original, target)
        snapshot_hashes[str(target.relative_to(run))] = sha256(target)
    for name, original in (
        ('data.yaml', DATA), ('D0.yaml', ROOT / 'ultralytics/cfg/models/v13/yolov13n-a4-qprr-d0-off.yaml'),
        *((f'{stage}.yaml', model_path(stage)) for stage in MODELS),
        ('A0_contract.json', baseline_contract_path),
    ):
        target = snapshots / name
        shutil.copy2(original, target)
        snapshot_hashes[str(target.relative_to(run))] = sha256(target)
    contract = dict(data=str(DATA), dataset_yaml_sha256=sha256(DATA),
        stages=list(MODELS), seeds=list(SEEDS), recipe=recipe, early_stopping=False,
        full_dataset_only=True, unconditional=True, d0_equivalence_only=True,
        upstream_state=str(UPSTREAM_STATE),
        reference_runs={k: str(v[0]) for k, v in REFERENCE_STAGES.items()},
        source_hashes=source_hashes, snapshot_hashes=snapshot_hashes,
        pretrained_sha256=sha256(ROOT / 'yolov13n.pt'), created_at=now())
    write_json(train / 'contract.json', contract)
    write_json(train / 'preflight.json', dict(status='passed', run_root=str(run),
        data=str(DATA), upstream_status=upstream['status'],
        upstream_failure_reason=upstream.get('failure_reason'),
        transfers=transfers, checked_at=now()))
    print('QPRR_PREFLIGHT_PASSED', run,
          json.dumps({k: v['loaded_tensors'] for k, v in transfers.items()}), flush=True)


if __name__ == '__main__':
    main()
