#!/usr/bin/env python3
"""Fresh D1-small training and evaluation, seeds 0/1, 300 epochs, patience 30."""
from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ultralytics

if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f'External Ultralytics: {ultralytics.__file__}')

from tools.ucra_v2_common import now, sha256, write_json
from tools.run_ucra_v2 import load, stop_processes, write_csv

DATA = Path('/home/room305/ZZF/URPC2020/data.yaml')
PRETRAINED = Path('/home/room305/ZZF/yolov13s.pt')
MODEL = ROOT / 'ultralytics/cfg/models/v13/yolov13s-a4-qprr-d1.yaml'
HISTORY = ROOT / 'runs/qprr_urpc2020_20260926_r1'
UPSTREAM = ROOT / 'runs/baseline_d1_urpc2020_20260930_r1/state.json'
SEEDS = (0, 1)
SETTINGS = dict(epochs=300, patience=30, device=0, workers=2, amp=False,
                deterministic=True, plots=False, imgsz=640, batch=16)
TERMINAL = ('completed', 'failed', 'cancelled')
SHELLS = ('scripts/run_d1_dualseed_urpc2020.sh',
          'scripts/wait_d1_dualseed_urpc2020_after_baseline_d1.sh')


def require_run(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('run-root must be absolute')
    path = path.resolve()
    if path.parent != ROOT / 'runs' or not re.fullmatch(
            r'd1_dualseed_urpc2020_[A-Za-z0-9][A-Za-z0-9_-]{2,80}', path.name):
        raise ValueError(f'Unexpected immutable run-root: {path}')
    return path


def verify_contract(run):
    frozen = load(run / 'train/contract.json')
    if (frozen['run_id'] != run.name.removeprefix('d1_dualseed_urpc2020_')
            or frozen['seeds'] != list(SEEDS) or frozen['stages'] != ['D1']
            or frozen['scale'] != 's' or frozen['data'] != str(DATA)
            or frozen['pretrained'] != str(PRETRAINED)
            or frozen['early_stopping'] is not True
            or any(frozen['recipe'].get(k) != v for k, v in SETTINGS.items())):
        raise RuntimeError('Training contract changed')
    if sha256(DATA) != frozen['dataset_yaml_sha256'] or sha256(PRETRAINED) != frozen['pretrained_sha256']:
        raise RuntimeError('Dataset configuration or initialization weights changed')
    for base, key in ((ROOT, 'source_hashes'), (run, 'snapshot_hashes')):
        for relative, expected in frozen[key].items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base) or sha256(path) != expected:
                raise RuntimeError(f'Frozen dependency changed: {path}')
    return frozen


def load_pretrained(model):
    import torch
    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(str(PRETRAINED))
    source_model = (checkpoint.get('ema') or checkpoint['model']).float()
    if source_model.yaml.get('scale') != 's':
        raise RuntimeError('Specified initialization checkpoint is not scale s')
    source, target = source_model.state_dict(), model.model.state_dict()
    matched = [k for k in target if k in source and target[k].shape == source[k].shape]
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    mismatch = {k: [list(source[k].shape), list(target[k].shape)] for k in target
                if k in source and source[k].shape != target[k].shape}
    invalid = [k for k in missing if not k.startswith(('model.15.', 'model.19.'))]
    invalid += [k for k in mismatch if not k.startswith('model.32.cv3.')]
    invalid += [k for k in unexpected if not k.startswith('model.32.cv3.')]
    if invalid:
        raise RuntimeError(f'Unexpected pretrained incompatibility: {invalid}')
    model.load(str(PRETRAINED))
    loaded = model.model.state_dict()
    if not all(torch.equal(loaded[k], source[k]) for k in matched):
        raise RuntimeError('Pretrained tensors were not loaded exactly')
    return dict(pretrained=str(PRETRAINED), pretrained_sha256=sha256(PRETRAINED),
                scale='s', loaded_tensors=len(matched), target_tensors=len(target),
                missing_keys=missing, unexpected_keys=unexpected, shape_mismatch_keys=mismatch)


def completion_reason(epochs, best_epoch, max_epochs=300, patience=30):
    if not 1 <= best_epoch <= epochs <= max_epochs:
        raise RuntimeError('Invalid completed epoch range')
    if epochs == max_epochs:
        return 'max_epochs'
    if epochs - best_epoch >= patience:
        return 'early_stopping'
    raise RuntimeError('Training stopped before the configured limit or early stopping')


def preflight(run):
    import yaml
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.modules import Detect
    from ultralytics.utils import DEFAULT_CFG

    if run.exists():
        raise FileExistsError(f'Run ID already exists: {run}')
    upstream = load(UPSTREAM)
    if upstream.get('status') not in TERMINAL:
        raise RuntimeError('Previous baseline-D1 chain is not terminal')
    historical = load(HISTORY / 'train/contract.json')
    differences = [rel for rel, expected in historical['source_hashes'].items()
                   if not (ROOT / rel).is_file() or sha256(ROOT / rel) != expected]
    if differences:
        raise RuntimeError(f'Historical D1 source dependencies differ: {differences}')
    cfg = yaml.safe_load(MODEL.read_text(encoding='utf-8'))
    previous = yaml.safe_load((HISTORY / 'train/snapshots/D1.yaml').read_text(encoding='utf-8'))
    if cfg.pop('scale', None) != 's' or cfg != previous:
        raise RuntimeError('D1 configuration changed beyond the requested scale s')
    if historical['data'] != str(DATA) or sha256(DATA) != historical['dataset_yaml_sha256']:
        raise RuntimeError('Full URPC2020 dataset configuration changed')
    data = check_det_dataset(str(DATA), autodownload=False)
    for split in ('train', 'val', 'test'):
        values = data.get(split)
        if not values:
            raise RuntimeError(f'Dataset split missing: {split}')
        for value in values if isinstance(values, list) else [values]:
            path = Path(value).resolve()
            if not path.exists() or not path.is_relative_to(DATA.parent):
                raise RuntimeError(f'Unexpected URPC2020 split: {path}')
    recipe = dict(historical['recipe'])
    recipe.update(SETTINGS, pretrained=str(PRETRAINED))
    if recipe.get('time') is not None or recipe.get('resume'):
        raise RuntimeError('Time limit and checkpoint resume are not permitted')
    model = YOLO(str(MODEL))
    if model.model.yaml.get('scale') != 's' or type(model.model.model[-1]) is not Detect:
        raise RuntimeError('Expected D1-small model with Detect head')
    transfer = load_pretrained(model)
    model.model.args = DEFAULT_CFG
    criterion = model.model.init_criterion()
    if not criterion.use_qprr or getattr(criterion, 'use_gbc_qprr', False):
        raise RuntimeError('Expected historical D1 QPRR loss')
    parameters = sum(p.numel() for p in model.model.parameters())
    del criterion, model
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_qprr.py'),
                    str(ROOT / 'tests/test_d1_dualseed_chain.py')], cwd='/tmp', check=True, timeout=180)
    for relative in SHELLS:
        subprocess.run(['bash', '-n', str(ROOT / relative)], check=True)
    files = set(historical['source_hashes']) | {str(MODEL.relative_to(ROOT)),
        str(Path(__file__).relative_to(ROOT)), 'tests/test_d1_dualseed_chain.py', *SHELLS}
    files.update(str(p.relative_to(ROOT)) for p in (ROOT / 'ultralytics').rglob('*.py'))
    for relative in files:
        if relative.endswith('.py'):
            ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'), filename=relative)
    snapshots = run / 'train/snapshots'
    snapshots.mkdir(parents=True)
    (run / 'test').mkdir()
    for source, name in ((MODEL, MODEL.name), (DATA, 'data.yaml'),
                         (HISTORY / 'train/contract.json', 'historical_D1_contract.json')):
        shutil.copy2(source, snapshots / name)
    write_json(run / 'train/contract.json', dict(
        run_id=run.name.removeprefix('d1_dualseed_urpc2020_'), stages=['D1'], seeds=list(SEEDS),
        data=str(DATA), scale='s', pretrained=str(PRETRAINED), early_stopping=True, recipe=recipe,
        dataset_yaml_sha256=sha256(DATA), pretrained_sha256=sha256(PRETRAINED),
        model_snapshot=str((snapshots / MODEL.name).relative_to(run)),
        source_hashes={relative: sha256(ROOT / relative) for relative in sorted(files)},
        snapshot_hashes={str(p.relative_to(run)): sha256(p) for p in snapshots.iterdir()},
        upstream_state=str(UPSTREAM), created_at=now()))
    audit = dict(historical_source_count=len(historical['source_hashes']),
                 historical_sources_identical=True, D1_only_change='scale n -> s',
                 parameters=parameters, transfer=transfer, syntax_checked=len(files))
    write_json(run / 'train/D1_code_audit.json', audit)
    write_json(run / 'train/preflight.json', dict(status='passed', checked_at=now(), audit=audit,
        upstream_status=upstream['status'], upstream_failure_reason=upstream.get('failure_reason')))
    verify_contract(run)
    print('D1_DUALSEED_PREFLIGHT_PASSED', json.dumps(audit), flush=True)


def worker(run, seed, mode):
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    frozen = verify_contract(run)
    output = run / mode / f'seed{seed}'
    train = run / 'train' / f'seed{seed}'
    if output.exists():
        raise FileExistsError(f'Refusing duplicate output: {output}')
    print(f'LOCAL_ULTRALYTICS={ultralytics.__file__}; scale=s; seed={seed}; mode={mode}', flush=True)
    if mode == 'test':
        import test as evaluator
        best = train / 'weights/best.pt'
        if not best.is_file():
            raise FileNotFoundError(best)
        sys.argv = [str(ROOT / 'test.py'), '--weights', str(best), '--data', str(DATA),
                    '--project', str(output.parent), '--name', output.name, '--device', '0',
                    '--batch', '16', '--imgsz', '640', '--workers', '2', '--no-plots']
        evaluator.main()
        for name in ('summary_metrics.json', 'scale_ap_metrics.json'):
            if not (output / name).is_file():
                raise FileNotFoundError(output / name)
        return
    init_seeds(seed, deterministic=True)
    model = YOLO(str(run / frozen['model_snapshot']))
    write_json(run / 'train' / f'seed{seed}.pretrained_load.json', load_pretrained(model))

    def check_recipe(trainer):
        actual = yaml.safe_load((output / 'args.yaml').read_text(encoding='utf-8'))
        if (any(actual.get(k) != v for k, v in frozen['recipe'].items())
                or actual['seed'] != seed or actual['resume'] is not False
                or trainer.stopper.patience != 30 or trainer.model.yaml.get('scale') != 's'):
            raise RuntimeError('Effective training recipe/scale/early stopping differs')
        print('EFFECTIVE_RECIPE_VERIFIED; SCALE=s; PATIENCE=30; RESUME=False', flush=True)

    def record_best(trainer):
        if trainer.best_fitness == trainer.fitness:
            write_json(output / 'best_epoch.json', dict(epoch=trainer.epoch + 1, fitness=float(trainer.fitness)))

    model.add_callback('on_train_start', check_recipe)
    model.add_callback('on_model_save', record_best)
    options = dict(frozen['recipe'], data=str(DATA), seed=seed, project=str(output.parent),
                   name=output.name, exist_ok=False, resume=False)
    started = time.perf_counter()
    model.train(**options)
    epochs = model.trainer.epoch + 1
    reason = completion_reason(epochs, model.trainer.stopper.best_epoch)
    best = output / 'weights/best.pt'
    if not best.is_file():
        raise FileNotFoundError(best)
    with (output / 'results.csv').open(encoding='utf-8') as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if ([int(row['epoch']) for row in rows] != list(range(1, epochs + 1))
            or not all(math.isfinite(v) for row in rows for v in row.values())):
        raise RuntimeError('Incomplete or nonfinite training results')
    write_json(output / 'train_complete.json', dict(stage='D1', seed=seed, epochs=epochs,
        max_epochs=300, patience=30, stop_reason=reason, early_stopping=True, scale='s',
        best_epoch=load(output / 'best_epoch.json')['epoch'], weights_sha256=sha256(best),
        settings=SETTINGS, seconds=time.perf_counter() - started, completed_at=now()))


def run_batch(run, mode, update):
    children = []
    try:
        pids = {}
        for seed in SEEDS:
            if (run / mode / f'seed{seed}').exists():
                raise FileExistsError(f'Refusing duplicate {mode}/seed{seed}')
            stream = (run / mode / f'seed{seed}.worker.log').open('a', encoding='utf-8')
            try:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    'worker', '--run-root', str(run), '--seed', str(seed), '--mode', mode],
                    cwd='/tmp', stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
            except BaseException:
                stream.close()
                raise
            children.append((process, stream))
            pids[str(seed)] = process.pid
            (run / f'{mode}_seed{seed}.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
            update('running', phase=mode, worker_pids=dict(pids))
        pending = {p.pid: p for p, _ in children}
        while pending:
            for pid, process in list(pending.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'{mode} worker {pid} exited {code}')
                    del pending[pid]
            if pending:
                time.sleep(2)
    except BaseException:
        stop_processes(children)
        raise
    finally:
        for _, stream in children:
            stream.close()
    update('running', worker_pids={})


def summarize(run):
    rows = []
    for seed in SEEDS:
        result = load(run / 'test' / f'seed{seed}/summary_metrics.json')
        scale = load(run / 'test' / f'seed{seed}/scale_ap_metrics.json')['metrics']
        train = load(run / 'train' / f'seed{seed}/train_complete.json')
        metrics = result['metrics']
        p, r = metrics['metrics/precision(B)'], metrics['metrics/recall(B)']
        rows.append(dict(stage='D1', seed=seed, P=p, R=r, F1=2*p*r/(p+r) if p+r else 0.,
            mAP50=metrics['metrics/mAP50(B)'], mAP50_95=metrics['metrics/mAP50-95(B)'],
            AP_S=scale['APS']/100, AP_M=scale['APM']/100, AP_L=scale['APL']/100,
            params=result['model']['parameters'], gflops=result['model']['gflops'],
            best_epoch=train['best_epoch'], epochs=train['epochs'], stop_reason=train['stop_reason']))
    keys = [k for k in rows[0] if k not in ('stage', 'seed', 'stop_reason')]
    if not all(math.isfinite(row[k]) for row in rows for k in keys):
        raise RuntimeError('Nonfinite summary metrics')
    stats = {k: dict(mean=statistics.mean(row[k] for row in rows),
                    std=statistics.stdev(row[k] for row in rows),
                    best=max(row[k] for row in rows), worst=min(row[k] for row in rows)) for k in keys}
    write_csv(run / 'test/AllSeed_summary.csv', rows)
    write_json(run / 'test/AllSeed_summary.json', dict(data=str(DATA), scale='s',
        pretrained=str(PRETRAINED), max_epochs=300, patience=30, seeds=rows,
        statistics={'D1': stats}, completed_stages=['D1'], std_ddof=1))


def manage(run, wait):
    verify_contract(run)
    if load(run / 'train/preflight.json').get('status') != 'passed':
        raise RuntimeError('Preflight did not pass')
    with (run / 'train/chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = run / 'state.json'
        old = load(path) if path.exists() else {}
        if old and not (not wait and old.get('status') == 'waiting'
                        and old.get('launcher_pid') == os.getpid()):
            raise RuntimeError('Run ID already started; explicit recovery required')
        state = dict(old, run_id=run.name.removeprefix('d1_dualseed_urpc2020_'),
                     run_root=str(run), data=str(DATA), stage='D1', scale='s', seeds=list(SEEDS),
                     worker_pids={}, launcher_pid=os.getpid(), upstream_state=str(UPSTREAM),
                     failure_reason=None, created_at=old.get('created_at', now()))

        def update(status, **extra):
            state.update(status=status, updated_at=now(), **extra)
            write_json(path, state)
            print(now(), json.dumps(state), flush=True)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'received signal {signum}')

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        (run / 'launcher.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
        try:
            update('waiting', phase='waiting_for_upstream')
            print('WAIT_BEGIN', str(UPSTREAM), now(), flush=True)
            while True:
                try:
                    upstream = load(UPSTREAM)
                except (OSError, ValueError, TypeError):
                    upstream = {}
                if isinstance(upstream, dict) and upstream.get('status') in TERMINAL:
                    break
                if not wait:
                    raise RuntimeError('Upstream is not terminal')
                time.sleep(30)
            update('waiting', phase='upstream_terminal', upstream_status=upstream['status'],
                   upstream_failure_reason=upstream.get('failure_reason'))
            print('UPSTREAM_TERMINAL', upstream['status'], upstream.get('failure_reason'), flush=True)
            if wait:
                print('CHAIN_START', now(), flush=True)
                fcntl.flock(lock, fcntl.LOCK_UN)
                os.execv('/bin/bash', ['bash', str(ROOT / SHELLS[0]), str(run),
                                     str(run / 'train/launcher.log')])
            update('running', started_at=now())
            for mode in ('train', 'test'):
                verify_contract(run)
                run_batch(run, mode, update)
            update('running', phase='summary')
            summarize(run)
            update('completed', phase='completed', completed_at=now(), worker_pids={})
        except BaseException as error:
            update('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                   failure_reason=f'{type(error).__name__}: {error}', worker_pids={}, stopped_at=now())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'wait', 'run', 'worker'))
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--mode', choices=('train', 'test'))
    args = parser.parse_args()
    run = require_run(args.run_root)
    if args.action == 'preflight':
        preflight(run)
    elif args.action == 'worker':
        if args.seed is None or args.mode is None:
            parser.error('worker requires seed and mode')
        worker(run, args.seed, args.mode)
    else:
        manage(run, wait=args.action == 'wait')


if __name__ == '__main__':
    main()
