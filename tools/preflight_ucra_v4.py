#!/usr/bin/env python3
"""Audit half-URPC2020 references, C0/C1/C2 invariants and transfer weights."""
from __future__ import annotations

import argparse
import ast
import json
import math
import runpy
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ucra_v4_common import ROOT, DATA, MODELS, REFERENCE_RUNS, SETTINGS, UPSTREAM_STATE, model_path, now, require_run_root, sha256, transfer_report, write_json
from tools.preflight_ucra_v2 import dataset_audit


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference_audit(run, audit):
    import yaml

    if load(UPSTREAM_STATE).get('status') not in ('completed', 'failed', 'cancelled'):
        raise RuntimeError('Upstream state is not terminal')
    v2 = REFERENCE_RUNS['A4']
    v3 = REFERENCE_RUNS['B2']
    for source in (v2, v3):
        if load(source / 'state.json')['status'] != 'completed':
            raise RuntimeError(f'Incomplete reference run: {source}')
        old_audit = load(source / 'train/dataset_audit.json')
        for split in ('train', 'val', 'test'):
            if audit['splits'][split]['content_sha256'] != old_audit['splits'][split]['content_sha256']:
                raise RuntimeError(f'Reference dataset changed: {source}/{split}')
    parent = load(v2 / 'train/contract.json')
    secondary = load(v3 / 'train/contract.json')
    if parent['recipe'] != secondary['recipe']:
        raise RuntimeError('A4 and B2 training recipes differ')
    if parent['dataset_yaml_sha256'] != sha256(DATA) or secondary['dataset_yaml_sha256'] != sha256(DATA):
        raise RuntimeError('Reference dataset YAML changed')
    if any(parent['recipe'].get(k) != v for k, v in SETTINGS.items()):
        raise RuntimeError('Parent recipe violates requested parameters')
    if 'patience' in parent['recipe'] or parent['early_stopping']:
        raise RuntimeError('Parent early stopping protocol differs')
    exceptions = {'ultralytics/nn/tasks.py', 'ultralytics/nn/modules/__init__.py'}
    for rel, expected in secondary['hashes'].items():
        if rel.startswith('ultralytics/') or rel in ('test.py', 'yolov13n.pt'):
            if rel not in exceptions and sha256(ROOT / rel) != expected:
                raise RuntimeError(f'Validated dependency changed: {rel}')
    refs = run / 'test/references'
    refs.mkdir(parents=True, exist_ok=True)
    records = []
    for stage, source_run in REFERENCE_RUNS.items():
        source = source_run / 'test' / f'{stage}_AllSeed_summary.json'
        summary = load(source)
        if len(summary['seeds']) != 3 or {r['seed'] for r in summary['seeds']} != {0, 1, 2}:
            raise RuntimeError(f'Incomplete reference summary: {stage}')
        for seed in range(3):
            train = source_run / 'train' / f'seed{seed}' / stage
            marker = load(train / 'train_complete.json')
            if marker['epochs'] != 180 or marker['settings'] != SETTINGS or marker['early_stopping']:
                raise RuntimeError(f'Invalid reference training: {stage}/seed{seed}')
            if sha256(train / 'weights/best.pt') != marker['weights_sha256']:
                raise RuntimeError(f'Reference checkpoint changed: {stage}/seed{seed}')
            args = yaml.safe_load((train / 'args.yaml').read_text(encoding='utf-8'))
            if any(args.get(k) != v for k, v in parent['recipe'].items()):
                raise RuntimeError(f'Effective reference recipe differs: {stage}/seed{seed}')
            raw = load(source_run / 'test' / f'seed{seed}' / stage / 'summary_metrics.json')['metrics']
            scale = load(source_run / 'test' / f'seed{seed}' / stage / 'scale_ap_metrics.json')['metrics']
            row = next(r for r in summary['seeds'] if r['seed'] == seed)
            for key, source_key in (('P', 'precision'), ('R', 'recall'), ('mAP50', 'mAP50'), ('mAP50_95', 'mAP50-95')):
                if not math.isclose(row[key], raw[f'metrics/{source_key}(B)'], rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f'Reference summary differs from raw metrics: {stage}/seed{seed}')
            for key, source_key in (('AP_S', 'APS'), ('AP_M', 'APM'), ('AP_L', 'APL')):
                if not math.isclose(row[key], scale[source_key] / 100, rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f'Reference scale summary differs: {stage}/seed{seed}')
        dest = refs / source.name
        shutil.copy2(source, dest)
        records.append(dict(stage=stage, source=str(source), snapshot=str(dest), sha256=sha256(dest)))
    write_json(refs / 'provenance.json', dict(sources=records, verified_at=now(),
        same_dataset=True, same_training_recipe=True, changed_registration_files=sorted(exceptions)))
    return parent['recipe']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    if (run / 'state.json').exists() or (run / 'train/contract.json').exists():
        raise FileExistsError('Existing run cannot be re-preflighted')
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.nn.tasks import torch_safe_load
    from ultralytics.utils.torch_utils import get_flops, init_seeds

    torch.set_num_threads(2)
    dependencies = ['test.py', 'yolov13n.pt', 'ultralytics/cfg/default.yaml',
        'tools/ucra_v2_common.py', 'tools/run_ucra_v2.py', 'tools/preflight_ucra_v2.py',
        'tools/ucra_v4_common.py', 'tools/train_ucra_v4_worker.py', 'tools/run_ucra_v4.py',
        'tools/preflight_ucra_v4.py', 'scripts/run_ucra_v4.sh',
        'scripts/wait_ucra_v4_after_ucra_urpc2020.sh',
        'experiments/urpc2020half_ucra_v4_gipr.yaml',
        'tests/test_ucra_v4_gipr.py', 'tests/verify_ucra_v4_repo.py',
        'tests/test_ucra_v4_chain.py']
    dependencies += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'ultralytics').rglob('*.py')]
    dependencies += [model_path(stage).relative_to(ROOT).as_posix() for stage in MODELS]
    dependencies.append((ROOT / 'ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml').relative_to(ROOT).as_posix())
    for path in set(ROOT / rel for rel in dependencies if rel.endswith('.py')):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    experiment = yaml.safe_load((ROOT / 'experiments/urpc2020half_ucra_v4_gipr.yaml').read_text(encoding='utf-8'))
    if ([stage for stage, entry in experiment['runs'].items() if entry['train']] != list(MODELS)
            or experiment['dataset_root'] != str(DATA.parent)):
        raise RuntimeError('Experiment manifest does not match user-directed full C0/C1/C2 run')
    audit = dataset_audit(DATA)
    write_json(run / 'train/dataset_audit.json', audit)
    print('DATASET_AUDIT', json.dumps(audit), flush=True)
    recipe = reference_audit(run, audit)
    print('REFERENCES_AND_DATASET_PASS', flush=True)
    functions = runpy.run_path(str(ROOT / 'tests/test_ucra_v4_gipr.py'))
    functions['round1_structure_and_math']()
    functions['round2_training_and_gradient_isolation']()
    runpy.run_path(str(ROOT / 'tests/verify_ucra_v4_repo.py'), run_name='__main__')
    base_path = ROOT / 'ultralytics/cfg/models/v13/yolov13n-ucra-v2-a4-full.yaml'
    base_cfg = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    base_layers = base_cfg['backbone'] + base_cfg['head']
    reports, baseline = {}, None
    sample = torch.linspace(0, 1, 3 * 640 * 640).reshape(1, 3, 640, 640)
    for stage, path in [('A4', base_path)] + [(stage, model_path(stage)) for stage in MODELS]:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        layers = cfg['backbone'] + cfg['head']
        if cfg['nc'] != 4 or len(layers) != 33 or layers[-1][0] != [23, 27, 31]:
            raise RuntimeError(f'Invalid model head: {stage}')
        if any(layers[i] != base_layers[i] for i in range(33) if i != 19):
            raise RuntimeError(f'Unexpected structure change: {stage}')
        init_seeds(0, deterministic=True)
        model = YOLO(str(path), verbose=False)
        transfer = transfer_report(model)
        model.model.eval()
        with torch.no_grad():
            prediction = model.model(sample)[0]
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f'Nonfinite prediction: {stage}')
        if baseline is None:
            baseline = prediction.clone()
        elif not torch.equal(prediction, baseline):
            raise RuntimeError(f'Zero-start prediction differs from A4: {stage}')
        started = time.perf_counter()
        with torch.no_grad():
            model.model(sample)
        cpu_forward_seconds = time.perf_counter() - started
        reports[stage] = dict(params=sum(p.numel() for p in model.model.parameters()),
            gflops=get_flops(model.model, imgsz=640), cpu_forward_640_seconds=cpu_forward_seconds,
            transfer=transfer, exact_A4_start=True, detect_sources=[23, 27, 31], layers=33)
        snapshot = run / 'train/model_snapshots' / path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
        print('MODEL_PASS', stage, reports[stage]['params'], reports[stage]['gflops'], flush=True)
        del model
    checkpoint = REFERENCE_RUNS['A4'] / 'train/seed0/A4/weights/best.pt'
    saved, _ = torch_safe_load(str(checkpoint))
    source = saved.get('ema') or saved['model']
    inherited = source.model[19].float().state_dict()
    init_seeds(0, deterministic=True)
    candidate = YOLO(str(model_path('C1')), verbose=False).model.model[19]
    missing, unexpected = candidate.load_state_dict(inherited, strict=False)
    if unexpected or any(not key.startswith(('precision_gain_raw', 'precision_offset_head.', 'precision_out.')) for key in missing):
        raise RuntimeError(f'A4 checkpoint incompatibility: missing={missing}, unexpected={unexpected}')
    if not all(torch.equal(candidate.state_dict()[key], value) for key, value in inherited.items()):
        raise RuntimeError('A4 inherited checkpoint parameters were not copied exactly')
    compatibility = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        missing_new_parameters=list(missing), unexpected_parameters=list(unexpected), inherited_exact=True)
    write_json(run / 'train/a4_checkpoint_compatibility.json', compatibility)
    shutil.copy2(DATA, run / 'train/data.snapshot.yaml')
    frozen = dict(created_at=now(), data=str(DATA), dataset_yaml_sha256=sha256(DATA),
        recipe=recipe, early_stopping=False, c2_unconditional=True, stages=list(MODELS),
        seeds=[0, 1, 2], references=list(REFERENCE_RUNS),
        initialization='YOLO(model_yaml).load(yolov13n.pt)',
        hashes={rel: sha256(ROOT / rel) for rel in sorted(set(dependencies))},
        reference_hashes={p.relative_to(run).as_posix(): sha256(p)
            for p in (run / 'test/references').glob('*.json')})
    write_json(run / 'train/contract.json', frozen)
    write_json(run / 'train/preflight.json', dict(status='passed', at=now(),
        syntax_files=len(set(rel for rel in dependencies if rel.endswith('.py'))),
        module_tests=True, repository_build_and_forward=True, models=reports,
        checkpoint_compatibility=compatibility, dataset_verified=True, no_trial_training=True))
    print('PREFLIGHT_PASSED', run, flush=True)


if __name__ == '__main__':
    main()
