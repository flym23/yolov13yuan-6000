#!/usr/bin/env python3
"""Validate F1R integration and freeze its full-URPC2020 training contract."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.f1r_urpc2020_common import (
    BASELINE_RUN, DATA, MODELS, REFERENCE_STAGES, SEEDS, SETTINGS,
    UPSTREAM_STATE, model_path, now, require_run_root, sha256,
    transfer_report, write_json,
)


def source_files() -> list[Path]:
    selected = set((ROOT / 'ultralytics').rglob('*.py'))
    selected.update(ROOT / relative for relative in (
        'test.py', 'ultralytics/cfg/default.yaml',
        'ultralytics/cfg/models/v13/yolov13n-a4-f1r-c0-off.yaml',
        'ultralytics/cfg/models/v13/yolov13n-a4-f1r-c1.yaml',
        'ultralytics/cfg/models/v13/yolov13n-f1r-c2-head-only.yaml',
        'tools/ucra_v2_common.py', 'tools/run_ucra_v2.py',
        'tools/f1r_urpc2020_common.py', 'tools/preflight_f1r_urpc2020.py',
        'tools/train_f1r_urpc2020_worker.py', 'tools/run_f1r_urpc2020.py',
        'tests/verify_f1r_repo.py', 'tests/test_f1_reconcile.py',
        'scripts/wait_f1r_urpc2020_after_sqanwd.sh',
        'scripts/run_f1r_urpc2020.sh', 'yolov13n.pt',
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
        raise RuntimeError('A0/A4 reference used another dataset')
    recipe = dict(baseline_contract['recipe'])
    if 'patience' in recipe or any(recipe.get(k) != v for k, v in SETTINGS.items()):
        raise RuntimeError('Reference recipe conflicts with requested settings')

    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.modules import F1ReconcileDetect

    baseline_models = {
        'C2': ROOT / 'ultralytics/cfg/models/v13/yolov13.yaml',
        'C1': ROOT / 'ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml',
    }
    transfers = {}
    for stage in MODELS:
        candidate = yaml.safe_load(model_path(stage).read_text(encoding='utf-8'))
        baseline = yaml.safe_load(baseline_models[stage].read_text(encoding='utf-8'))
        if 'sqanwd' in candidate:
            raise RuntimeError(f'{stage} unexpectedly enables SQANWD')
        baseline['nc'] = candidate['nc']
        if candidate['backbone'] != baseline['backbone'] or candidate['head'][:-1] != baseline['head'][:-1]:
            raise RuntimeError(f'{stage} changes architecture beyond the Detect head')
        if candidate['head'][-1][2] != 'F1ReconcileDetect':
            raise RuntimeError(f'{stage} does not use the F1R head')
        model = YOLO(str(model_path(stage)))
        if not isinstance(model.model.model[-1], F1ReconcileDetect):
            raise RuntimeError(f'{stage} built another head')
        transfers[stage] = transfer_report(model, stage)
        del model

    # The attached verifier checks C0/A4 bit-exact initialization and loss,
    # including C1 zero-gain equivalence, before any train worker is launched.
    subprocess.run([sys.executable, str(ROOT / 'tests/verify_f1r_repo.py')],
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
    for stage, source_run in REFERENCE_STAGES.items():
        original = source_run / 'test' / f'{stage}_AllSeed_summary.json'
        contents = json.loads(original.read_text(encoding='utf-8'))
        if {row.get('seed') for row in contents['seeds']} != set(SEEDS):
            raise RuntimeError(f'Incomplete {stage} reference')
        target = references / original.name
        shutil.copy2(original, target)
        snapshot_hashes[str(target.relative_to(run))] = sha256(target)
    for name, original in (
        ('data.yaml', DATA), ('C2.yaml', model_path('C2')),
        ('C1.yaml', model_path('C1')),
        ('C0.yaml', ROOT / 'ultralytics/cfg/models/v13/yolov13n-a4-f1r-c0-off.yaml'),
        ('A0_contract.json', baseline_contract_path),
    ):
        target = snapshots / name
        shutil.copy2(original, target)
        snapshot_hashes[str(target.relative_to(run))] = sha256(target)
    contract = dict(data=str(DATA), dataset_yaml_sha256=sha256(DATA),
        stages=list(MODELS), seeds=list(SEEDS), recipe=recipe, early_stopping=False,
        full_dataset_only=True, unconditional=True, c0_equivalence_only=True,
        upstream_state=str(UPSTREAM_STATE), reference_runs={k: str(v) for k, v in REFERENCE_STAGES.items()},
        source_hashes=source_hashes, snapshot_hashes=snapshot_hashes,
        pretrained_sha256=sha256(ROOT / 'yolov13n.pt'), created_at=now())
    write_json(train / 'contract.json', contract)
    write_json(train / 'preflight.json', dict(status='passed', run_root=str(run),
        data=str(DATA), upstream_status=upstream['status'],
        upstream_failure_reason=upstream.get('failure_reason'),
        transfers=transfers, checked_at=now()))
    print('F1R_PREFLIGHT_PASSED', run,
          json.dumps({k: v['loaded_tensors'] for k, v in transfers.items()}), flush=True)


if __name__ == '__main__':
    main()
