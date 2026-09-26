#!/usr/bin/env python3
"""CPU-only full dataset and five-model audit; freeze the immutable training contract."""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ucra_urpc2020_common import ROOT, DATA, MODELS, SETTINGS, model_path, now, require_run_root, sha256, transfer_report, write_json
from tools.preflight_ucra_v2 import dataset_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    if (run / 'state.json').exists() or (run / 'train/contract.json').exists():
        raise FileExistsError('Existing run cannot be re-preflighted')
    upstream = json.loads((run / 'upstream_process.json').read_text(encoding='utf-8'))
    if upstream['pid'] != 3645793 or not isinstance(upstream['start_ticks'], int) or not upstream['boot_id']:
        raise ValueError('Invalid upstream identity')
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds, get_flops
    torch.set_num_threads(2)
    parent_path = ROOT / 'runs/ucra_v3_20260921_r1/train/contract.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    # Inherit recipe/code only; no model checkpoints or metrics from URPC2020half.
    dependencies = [rel for rel in parent['hashes'] if rel.startswith('ultralytics/') or rel in ('test.py', 'yolov13n.pt')]
    for rel in dependencies:
        if sha256(ROOT / rel) != parent['hashes'][rel]:
            raise RuntimeError(f'Previously validated model/evaluator dependency changed: {rel}')
    recipe = dict(parent['recipe'])
    recipe.pop('patience', None)
    recipe.update(SETTINGS)
    dependencies += ['tools/ucra_v2_common.py', 'tools/run_ucra_v2.py', 'tools/preflight_ucra_v2.py',
        'tools/ucra_urpc2020_common.py', 'tools/ucra_process_gate.py', 'tools/run_ucra_urpc2020.py',
        'tools/train_ucra_urpc2020_worker.py', 'tools/preflight_ucra_urpc2020.py',
        'scripts/run_ucra_urpc2020.sh', 'scripts/wait_ucra_urpc2020_after_pid3645793.sh',
        'tests/test_ucra_urpc2020_chain.py']
    dependencies += [model_path(s).relative_to(ROOT).as_posix() for s in MODELS]
    syntax = list((ROOT / 'ultralytics').rglob('*.py')) + [ROOT / p for p in dependencies if p.endswith('.py')]
    for path in set(syntax):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    audit = dataset_audit(DATA)
    for split in audit['splits'].values():
        if any(not Path(p).resolve().is_relative_to(DATA.parent) for p in split['paths']):
            raise RuntimeError('Dataset split resolves outside full URPC2020')
    write_json(run / 'train/dataset_audit.json', audit)
    print('FULL_DATASET_AUDIT', json.dumps(audit), flush=True)
    base = yaml.safe_load(model_path('A0').read_text(encoding='utf-8'))
    base_layers = base['backbone'] + base['head']
    reports, reference = {}, None
    image = torch.linspace(0, 1, 3 * 640 * 640).reshape(1, 3, 640, 640)
    for stage in MODELS:
        path = model_path(stage)
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        layers = cfg['backbone'] + cfg['head']
        if cfg['nc'] != 4 or len(layers) != 33 or layers[-1][0] != [23, 27, 31]:
            raise RuntimeError(f'Unexpected structure: {stage}')
        if any(layers[i] != base_layers[i] for i in range(33) if i not in (15, 19)):
            raise RuntimeError(f'Unexpected structural changes: {stage}')
        init_seeds(0, deterministic=True)
        model = YOLO(str(path), verbose=False)
        transfer = transfer_report(model)
        model.model.eval()
        with torch.no_grad():
            prediction = model.model(image)[0]
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f'Nonfinite output: {stage}')
        if reference is None:
            reference = prediction.clone()
        elif not torch.equal(prediction, reference):
            raise RuntimeError(f'Initial prediction differs from A0: {stage}')
        reports[stage] = dict(transfer=transfer, params=sum(p.numel() for p in model.model.parameters()),
            gflops=get_flops(model.model, imgsz=640), forward_640=True, exact_baseline_start=True)
        snapshot = run / 'train/model_snapshots' / path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        print('MODEL_PASS', stage, reports[stage]['params'], 'loaded', transfer['loaded_tensors'], flush=True)
        del model
    shutil.copy2(DATA, run / 'train/data.snapshot.yaml')
    write_json(run / 'train/recipe_source.json', dict(source=str(parent_path), sha256=sha256(parent_path), recipe=recipe))
    write_json(run / 'train/contract.json', dict(created_at=now(), data=str(DATA),
        dataset_yaml_sha256=sha256(DATA), recipe=recipe, early_stopping=False,
        inherited_patience_ignored=yaml.safe_load((ROOT / 'ultralytics/cfg/default.yaml').read_text())['patience'],
        stages=list(MODELS), seeds=[0, 1, 2], reference_stages=[], upstream_process=upstream,
        initialization='YOLO(model_yaml).load(yolov13n.pt)',
        hashes={rel: sha256(ROOT / rel) for rel in sorted(set(dependencies))}))
    write_json(run / 'train/preflight.json', dict(status='passed', at=now(), models=reports,
        syntax_files=len(set(syntax)), dataset=True, device='cpu', no_trial_training=True,
        previous_validated_model_code_unchanged=True))
    print('PREFLIGHT_PASSED', run, flush=True)


if __name__ == '__main__':
    main()
