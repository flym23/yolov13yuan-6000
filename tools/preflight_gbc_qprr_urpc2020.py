#!/usr/bin/env python3
"""Validate GBC-QPRR sources, references, models, and freeze a run contract."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.gbc_qprr_urpc2020_common import (
    BASELINE_RUN, DATA, MODELS, PRETRAINED, QPRR_RUN, REFERENCE_STAGES,
    RUN_ID, SEEDS, SETTINGS, STAGE_ORDER, UPSTREAM_STATE, load, model_path,
    now, require_run_root, sha256, transfer_report, write_json,
)


def source_files() -> list[Path]:
    selected = set((ROOT / 'ultralytics').rglob('*.py'))
    selected.update(ROOT / relative for relative in (
        'train.py', 'test.py', 'ultralytics/cfg/default.yaml',
        'ultralytics/cfg/models/v13/yolov13.yaml',
        'ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml',
        'ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e0-off.yaml',
        *[f"ultralytics/cfg/models/v13/{name}" for name in MODELS.values()],
        'ultralytics/utils/gbc_qprr.py', 'tools/ucra_v2_common.py',
        'tools/run_ucra_v2.py', 'tools/gbc_qprr_urpc2020_common.py',
        'tools/preflight_gbc_qprr_urpc2020.py',
        'tools/train_gbc_qprr_urpc2020_worker.py',
        'tools/run_gbc_qprr_urpc2020.py',
        'tests/verify_gbc_qprr_repo.py', 'tests/test_gbc_qprr.py',
        'experiments/urpc2020_gbc_qprr.yaml',
        'experiments/GBC_QPRR_Codex_Plan_20260928.md',
        'scripts/wait_gbc_qprr_urpc2020_after_qprr.sh',
        'scripts/run_gbc_qprr_urpc2020.sh', 'yolov13n.pt',
    ))
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    if run.name != f'gbc_qprr_urpc2020_{RUN_ID}':
        raise ValueError(f'Unexpected immutable run_id: {run.name}')
    if run.exists():
        raise FileExistsError(f'Run ID already exists: {run}')
    if not DATA.is_file() or not PRETRAINED.is_file():
        raise FileNotFoundError('Full URPC2020 dataset YAML or yolov13n.pt is missing')
    upstream = load(UPSTREAM_STATE)
    if upstream.get('status') not in ('completed', 'failed', 'cancelled'):
        raise RuntimeError('QPRR upstream must reach a terminal state before launch')

    import yaml
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect

    if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
        raise ImportError(f'External Ultralytics: {ultralytics.__file__}')
    baseline_contract_path = BASELINE_RUN / 'train/contract.json'
    baseline_contract = load(baseline_contract_path)
    if baseline_contract.get('data') != str(DATA):
        raise RuntimeError('A0/A4 reference contract uses a different dataset')
    recipe = dict(baseline_contract['recipe'])
    if 'patience' in recipe or any(recipe.get(key) != value for key, value in SETTINGS.items()):
        raise RuntimeError('Reference training recipe conflicts with requested settings')

    baseline_models = {
        'E0': ROOT / 'ultralytics/cfg/models/v13/yolov13.yaml',
        'E1': ROOT / 'ultralytics/cfg/models/v13/yolov13.yaml',
        'E2': ROOT / 'ultralytics/cfg/models/v13/yolov13.yaml',
        'E3': ROOT / 'ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml',
    }
    stage_configs = {
        'E0': ROOT / 'ultralytics/cfg/models/v13/yolov13n-gbc-qprr-e0-off.yaml',
        **{stage: model_path(stage) for stage in MODELS},
    }
    expected_coverage = {'E0': 0.0, 'E1': 0.0, 'E2': 0.35, 'E3': 0.35}
    for stage, candidate_path in stage_configs.items():
        candidate = yaml.safe_load(candidate_path.read_text(encoding='utf-8'))
        baseline = yaml.safe_load(baseline_models[stage].read_text(encoding='utf-8'))
        gbc = candidate.pop('gbc_qprr', None)
        if candidate != baseline or not isinstance(gbc, dict):
            raise RuntimeError(f'{stage} changes inference architecture beyond its stated base')
        if gbc.get('enabled') is not True or float(gbc.get('gain', -1)) != (0.0 if stage == 'E0' else 0.08):
            raise RuntimeError(f'{stage} has an unexpected GBC-QPRR gain')
        if float(gbc.get('coverage_weight', -1)) != expected_coverage[stage]:
            raise RuntimeError(f'{stage} has unexpected coverage_weight')
        if candidate.get('qprr') or candidate.get('sqanwd'):
            raise RuntimeError(f'{stage} enables an unrelated loss method')

    transfers = {}
    for stage in STAGE_ORDER:
        model = YOLO(str(model_path(stage)))
        if type(model.model.model[-1]) is not Detect:
            raise RuntimeError(f'{stage} does not retain the original Detect inference head')
        if model.model.yaml.get('qprr') or model.model.yaml.get('sqanwd'):
            raise RuntimeError(f'{stage} contains an unrelated loss config')
        transfers[stage] = transfer_report(model)
        del model

    # D0 is a loss-only exact-off check; it must not be trained.
    subprocess.run([sys.executable, str(ROOT / 'tests/verify_gbc_qprr_repo.py')],
                   cwd='/tmp', check=True, timeout=600)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_gbc_qprr.py')],
                   cwd='/tmp', check=True, timeout=300,
                   env={**__import__('os').environ, 'PYTHONPATH': str(ROOT)})

    for alias, (source_run, source_stage) in REFERENCE_STAGES.items():
        source = source_run / 'test' / f'{source_stage}_AllSeed_summary.json'
        if not source.is_file():
            raise FileNotFoundError(f'Missing reference summary: {source}')
        if {row.get('seed') for row in load(source).get('seeds', [])} != set(SEEDS):
            raise RuntimeError(f'Reference {alias} does not contain all three seeds')

    selected_sources = source_files()
    for path in selected_sources:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise FileNotFoundError(f'Invalid source dependency: {path}')
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in selected_sources}

    run.mkdir(parents=True)
    train = run / 'train'
    references = run / 'test/references'
    snapshots = train / 'snapshots'
    references.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    snapshot_hashes = {}
    for alias, (source_run, source_stage) in REFERENCE_STAGES.items():
        source = source_run / 'test' / f'{source_stage}_AllSeed_summary.json'
        target = references / f'{alias}_AllSeed_summary.json'
        shutil.copy2(source, target)
        snapshot_hashes[str(target.relative_to(run))] = sha256(target)
    snapshot_sources = {
        'data.yaml': DATA,
        'A0.yaml': baseline_models['E0'],
        'A4.yaml': baseline_models['E3'],
        'E0.yaml': stage_configs['E0'],
        **{f'{stage}.yaml': model_path(stage) for stage in STAGE_ORDER},
        'A0_contract.json': baseline_contract_path,
    }
    for name, source in snapshot_sources.items():
        target = snapshots / name
        shutil.copy2(source, target)
        snapshot_hashes[str(target.relative_to(run))] = sha256(target)

    failure_reason = upstream.get('failure_reason') or upstream.get('failed_reason')
    contract = dict(
        run_id=RUN_ID, data=str(DATA), dataset_yaml_sha256=sha256(DATA),
        stages=list(STAGE_ORDER), seeds=list(SEEDS), recipe=recipe,
        early_stopping=False, full_dataset_only=True, unconditional=True,
        e0_equivalence_only=True, upstream_state=str(UPSTREAM_STATE),
        upstream_status=upstream['status'], upstream_failure_reason=failure_reason,
        reference_runs={alias: str(value[0]) for alias, value in REFERENCE_STAGES.items()},
        source_hashes=source_hashes, snapshot_hashes=snapshot_hashes,
        pretrained_sha256=sha256(PRETRAINED), created_at=now(),
    )
    write_json(train / 'contract.json', contract)
    write_json(train / 'preflight.json', dict(status='passed', run_root=str(run),
        run_id=RUN_ID, data=str(DATA), upstream_status=upstream['status'],
        upstream_failure_reason=failure_reason, transfers=transfers, checked_at=now()))
    write_json(run / 'state.json', dict(status='waiting', phase='preflight_passed',
        run_id=RUN_ID, run_root=str(run), data=str(DATA), stages=list(STAGE_ORDER),
        seeds=list(SEEDS), upstream_state=str(UPSTREAM_STATE),
        upstream_status=upstream['status'], upstream_failure_reason=failure_reason,
        created_at=now(), updated_at=now(), worker_pids={}, failure_reason=None,
        full_dataset_only=True, unconditional=True))
    print('GBC_QPRR_PREFLIGHT_PASSED', run,
          json.dumps({stage: report['loaded_tensors'] for stage, report in transfers.items()}),
          flush=True)


if __name__ == '__main__':
    main()
