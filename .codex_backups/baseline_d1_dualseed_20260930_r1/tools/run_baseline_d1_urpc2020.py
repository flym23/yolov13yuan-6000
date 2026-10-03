#!/usr/bin/env python3
"""Reproduce original YOLOv13 A0 then historical QPRR D1, three seeds per stage."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ultralytics

if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f'External Ultralytics: {ultralytics.__file__}')

from tools.qprr_urpc2020_common import DATA, SEEDS, SETTINGS, summarize
from tools.ucra_v2_common import now, sha256, transfer_report, write_json
from tools.run_ucra_v2 import load, stop_processes, write_csv
from tools.train_qprr_urpc2020_worker import no_early_stop

RUN_ID = None
RUN = None
HISTORY = ROOT / 'runs/qprr_urpc2020_20260926_r1'
BASELINE_HISTORY = ROOT / 'runs/ucra_urpc2020_20260921_r1'
UPSTREAM = HISTORY / 'state.json'
MODELS = {'A0': 'yolov13.yaml', 'D1': 'yolov13n-a4-qprr-d1.yaml'}
SHELLS = ('scripts/run_baseline_d1_urpc2020.sh',
          'scripts/wait_baseline_d1_urpc2020_after_qprr.sh')
TERMINAL = ('completed', 'failed', 'cancelled')


def model_path(stage):
    return ROOT / 'ultralytics/cfg/models/v13' / MODELS[stage]


def verify_contract(run):
    if run != RUN:
        raise ValueError(f'Unexpected immutable run-root: {run}')
    contract = load(run / 'train/contract.json')
    if (contract['run_id'] != RUN_ID or contract['stages'] != list(MODELS)
            or contract['seeds'] != list(SEEDS) or contract['data'] != str(DATA)
            or contract['early_stopping'] is not False or 'patience' in contract['recipe']
            or any(contract['recipe'].get(k) != v for k, v in SETTINGS.items())):
        raise RuntimeError('Training contract changed')
    if sha256(DATA) != contract['dataset_yaml_sha256']:
        raise RuntimeError('Dataset YAML changed')
    for rel, expected in contract['source_hashes'].items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise RuntimeError(f'Source changed: {rel}')
    for rel, expected in contract['snapshot_hashes'].items():
        path = (run / rel).resolve()
        if not path.is_relative_to(run) or sha256(path) != expected:
            raise RuntimeError(f'Snapshot changed: {rel}')
    return contract


def preflight(run):
    import ast
    import yaml
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.modules import Detect
    from ultralytics.utils import DEFAULT_CFG

    if run.exists():
        raise FileExistsError(f'Run ID already exists: {run}')
    upstream = load(UPSTREAM)
    if upstream.get('status') not in TERMINAL:
        raise RuntimeError('QPRR upstream is not terminal')
    historical = load(HISTORY / 'train/contract.json')
    if load(HISTORY / 'state.json').get('status') != 'completed':
        raise RuntimeError('Historical QPRR run is incomplete')
    mismatches = [rel for rel, expected in historical['source_hashes'].items()
                  if not (ROOT / rel).is_file() or sha256(ROOT / rel) != expected]
    if mismatches:
        raise RuntimeError(f'Historical QPRR dependencies differ: {mismatches}')
    a0_history = load(BASELINE_HISTORY / 'train/contract.json')
    a0_rel = str(model_path('A0').relative_to(ROOT))
    if sha256(model_path('A0')) != a0_history['hashes'][a0_rel]:
        raise RuntimeError('Original A0 architecture changed')
    if sha256(model_path('D1')) != sha256(HISTORY / 'train/snapshots/D1.yaml'):
        raise RuntimeError('D1 differs from its historical YAML snapshot')
    recipe = historical['recipe']
    if (historical['data'] != str(DATA) or 'patience' in recipe
            or any(recipe.get(k) != v for k, v in SETTINGS.items())
            or historical['dataset_yaml_sha256'] != sha256(DATA)):
        raise RuntimeError('Historical dataset/recipe differs')
    if sha256(ROOT / 'yolov13n.pt') != historical['pretrained_sha256']:
        raise RuntimeError('Initialization weights changed')
    data = check_det_dataset(str(DATA), autodownload=False)
    for split in ('train', 'val', 'test'):
        paths = data.get(split)
        if not paths:
            raise RuntimeError(f'Dataset split missing: {split}')
        for value in paths if isinstance(paths, list) else [paths]:
            path = Path(value).resolve()
            if not path.is_relative_to(DATA.parent) or not path.exists():
                raise RuntimeError(f'Dataset split outside full URPC2020: {path}')
    old_args = {}
    for seed in SEEDS:
        p = HISTORY / 'train' / f'seed{seed}/D1/args.yaml'
        a = yaml.safe_load(p.read_text(encoding='utf-8'))
        if a['seed'] != seed or a['data'] != str(DATA):
            raise RuntimeError(f'Historical seed/data mismatch: {seed}')
        if any(a.get(k) != v for k, v in recipe.items()):
            raise RuntimeError(f'Historical effective recipe differs: {seed}')
        old_args[seed] = p
    transfers = {}
    for stage in MODELS:
        model = YOLO(str(model_path(stage)))
        cfg = model.model.yaml
        if type(model.model.model[-1]) is not Detect or cfg.get('gbc_qprr') or cfg.get('sqanwd'):
            raise RuntimeError(f'Unexpected model/loss: {stage}')
        if stage == 'A0' and cfg.get('qprr'):
            raise RuntimeError('Baseline must have no QPRR')
        if stage == 'D1' and (not cfg.get('qprr', {}).get('enabled') or cfg['qprr']['gain'] <= 0):
            raise RuntimeError('D1 QPRR is not enabled')
        transfers[stage] = transfer_report(model)
        model.model.args = DEFAULT_CFG
        criterion = model.model.init_criterion()
        if criterion.use_qprr != (stage == 'D1') or getattr(criterion, 'use_gbc_qprr', False):
            raise RuntimeError(f'Wrong criterion: {stage}')
        del criterion, model
    subprocess.run([sys.executable, str(ROOT / 'tests/verify_qprr_repo.py')],
                   cwd='/tmp', check=True, timeout=300)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_qprr.py'),
                    str(ROOT / 'tests/test_baseline_d1_chain.py')],
                   cwd='/tmp', check=True, timeout=180)
    for shell in SHELLS:
        subprocess.run(['bash', '-n', str(ROOT / shell)], check=True)
    files = set(historical['source_hashes']) | {a0_rel, str(Path(__file__).relative_to(ROOT)),
                                             'tests/test_baseline_d1_chain.py', *SHELLS}
    files.update(str(p.relative_to(ROOT)) for p in (ROOT / 'ultralytics').rglob('*.py'))
    for rel in files:
        if rel.endswith('.py'):
            ast.parse((ROOT / rel).read_text(encoding='utf-8-sig'), filename=rel)
    source_hashes = {rel: sha256(ROOT / rel) for rel in sorted(files)}
    snapshots = run / 'train/snapshots'
    snapshots.mkdir(parents=True)
    (run / 'test').mkdir()
    originals = {'data.yaml': DATA, 'historical_QPRR_contract.json': HISTORY / 'train/contract.json',
                 'historical_A0_contract.json': BASELINE_HISTORY / 'train/contract.json'}
    originals.update({f'{stage}.yaml': model_path(stage) for stage in MODELS})
    originals.update({f'D1_seed{seed}_historical_args.yaml': p for seed, p in old_args.items()})
    for name, original in originals.items():
        shutil.copy2(original, snapshots / name)
    snapshot_hashes = {str(p.relative_to(run)): sha256(p) for p in snapshots.iterdir()}
    audit = dict(checked_at=now(), historical_run=str(HISTORY),
                 historical_source_count=len(historical['source_hashes']),
                 all_historical_sources_identical=True, mismatches=[],
                 original_A0_yaml_identical=True, D1_yaml_identical=True,
                 pretrained_identical=True, dataset_yaml_identical=True,
                 historical_recipe_identical=True, old_seeds=list(SEEDS),
                 loss_sha256=sha256(ROOT / 'ultralytics/utils/loss.py'),
                 note='Source and recipe identity do not guarantee bit-identical training metrics.')
    write_json(run / 'train/D1_code_audit.json', audit)
    write_json(run / 'train/contract.json', dict(run_id=RUN_ID, data=str(DATA),
        stages=list(MODELS), seeds=list(SEEDS), recipe=recipe, early_stopping=False,
        dataset_yaml_sha256=sha256(DATA), source_hashes=source_hashes,
        snapshot_hashes=snapshot_hashes, upstream_state=str(UPSTREAM), created_at=now()))
    write_json(run / 'train/preflight.json', dict(status='passed', transfers=transfers,
        upstream_status=upstream['status'], upstream_failure_reason=upstream.get('failure_reason'),
        checked_at=now(), audit=audit))
    verify_contract(run)
    print('BASELINE_D1_PREFLIGHT_PASSED', run, json.dumps(audit), flush=True)


def worker(run, stage, seed, mode):
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    frozen = verify_contract(run)
    output = run / mode / f'seed{seed}' / stage
    train = run / 'train' / f'seed{seed}' / stage
    if output.exists():
        raise FileExistsError(f'Refusing duplicate output: {output}')
    print(f'LOCAL_ULTRALYTICS={ultralytics.__file__}; stage={stage}; seed={seed}; mode={mode}', flush=True)
    if mode == 'test':
        best = train / 'weights/best.pt'
        if not best.is_file():
            raise FileNotFoundError(best)
        import test as evaluator
        sys.argv = [str(ROOT / 'test.py'), '--weights', str(best), '--data', str(DATA),
                    '--project', str(output.parent), '--name', stage, '--device', '0',
                    '--batch', '16', '--imgsz', '640', '--workers', '2', '--no-plots']
        evaluator.main()
        for name in ('summary_metrics.json', 'scale_ap_metrics.json'):
            if not (output / name).is_file():
                raise FileNotFoundError(output / name)
        return
    init_seeds(seed, deterministic=True)
    model = YOLO(str(model_path(stage)))
    write_json(output.parent / f'{stage}.pretrained_load.json', transfer_report(model))
    model.add_callback('on_train_start', no_early_stop)

    def check_effective_recipe(trainer):
        actual = yaml.safe_load((output / 'args.yaml').read_text(encoding='utf-8'))
        if any(actual.get(k) != v for k, v in frozen['recipe'].items()):
            raise RuntimeError('Effective training recipe differs from history')
        if not math.isinf(trainer.stopper.patience):
            raise RuntimeError('Early stopping must be disabled')
        if stage == 'D1':
            expected = yaml.safe_load((run / 'train/snapshots' /
                                      f'D1_seed{seed}_historical_args.yaml').read_text(encoding='utf-8'))
            ignored = {'project', 'save_dir'}
            differences = {k: [expected.get(k), actual.get(k)] for k in set(expected) | set(actual)
                           if k not in ignored and expected.get(k) != actual.get(k)}
            write_json(output / 'historical_args_check.json',
                       dict(differences=differences, allowed_output_fields=sorted(ignored), checked_at=now()))
            if differences:
                raise RuntimeError(f'D1 effective args differ: {differences}')
        print('EFFECTIVE_RECIPE_VERIFIED; EARLY_STOPPING_DISABLED', flush=True)

    def record_best_epoch(trainer):
        if trainer.best_fitness == trainer.fitness:
            write_json(output / 'best_epoch.json', dict(epoch=trainer.epoch + 1, fitness=float(trainer.fitness)))

    model.add_callback('on_train_start', check_effective_recipe)
    model.add_callback('on_model_save', record_best_epoch)
    options = dict(frozen['recipe'])
    options.update(SETTINGS, data=str(DATA), seed=seed, project=str(output.parent),
                   name=stage, exist_ok=False, resume=False)
    started = time.perf_counter()
    model.train(**options)
    best = output / 'weights/best.pt'
    if model.trainer.epoch + 1 != SETTINGS['epochs'] or not best.is_file():
        raise RuntimeError('Training did not complete 180 epochs with a checkpoint')
    with (output / 'results.csv').open(encoding='utf-8') as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if ([int(row['epoch']) for row in rows] != list(range(1, 181))
            or not all(math.isfinite(v) for row in rows for v in row.values())):
        raise RuntimeError('Incomplete or nonfinite training results')
    write_json(output / 'train_complete.json', dict(stage=stage, seed=seed, data=str(DATA),
        epochs=len(rows), settings=SETTINGS, early_stopping=False,
        best_epoch=load(output / 'best_epoch.json')['epoch'], weights_sha256=sha256(best),
        seconds=time.perf_counter() - started, completed_at=now()))


def run_batch(run, stage, mode, update):
    children = []
    try:
        pids = {}
        for seed in SEEDS:
            if (run / mode / f'seed{seed}' / stage).exists():
                raise FileExistsError(f'Refusing duplicate {stage}/{mode}/seed{seed}')
            stream = (run / mode / f'{stage}_seed{seed}.worker.log').open('a', encoding='utf-8')
            try:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    'worker', '--run-root', str(run), '--stage', stage, '--seed', str(seed), '--mode', mode],
                    cwd='/tmp', stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
            except BaseException:
                stream.close()
                raise
            children.append((process, stream))
            pids[str(seed)] = process.pid
            (run / f'{mode}_{stage}_seed{seed}.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
            update('running', stage=stage, phase=mode, worker_pids=dict(pids))
        pending = {p.pid: p for p, _ in children}
        while pending:
            for pid, process in list(pending.items()):
                code = process.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f'{stage}/{mode} worker {pid} exited {code}')
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


def manage(run, wait):
    verify_contract(run)
    if load(run / 'train/preflight.json').get('status') != 'passed':
        raise RuntimeError('Preflight did not pass')
    with (run / 'train/chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = run / 'state.json'
        old = load(state_path) if state_path.exists() else {}
        # Only the same PID may hand off from the wait script; no implicit restart.
        if old and not (not wait and old.get('status') == 'waiting'
                        and old.get('launcher_pid') == os.getpid()):
            raise RuntimeError('Run ID already started; manual recovery required')
        state = dict(old, run_id=RUN_ID, run_root=str(run), data=str(DATA),
                     stages=list(MODELS), seeds=list(SEEDS), worker_pids={},
                     launcher_pid=os.getpid(), upstream_state=str(UPSTREAM),
                     failure_reason=None, created_at=old.get('created_at', now()))

        def update(status, **extra):
            state.update(status=status, updated_at=now(), **extra)
            write_json(state_path, state)
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
            update('running', phase='start', started_at=now())
            rows, stats = [], {}
            for stage in MODELS:
                verify_contract(run)
                for mode in ('train', 'test'):
                    run_batch(run, stage, mode, update)
                stage_rows, stage_stats = summarize(run, stage)
                rows.extend(stage_rows)
                stats[stage] = stage_stats
                write_csv(run / 'test/AllSeed_summary.csv', rows)
                write_json(run / 'test/AllSeed_summary.json', dict(data=str(DATA), seeds=rows,
                    statistics=stats, completed_stages=list(stats), std_ddof=1))
                update('running', stage=stage, phase='summarized')
            update('completed', phase='completed', worker_pids={}, completed_at=now())
        except BaseException as error:
            update('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                   failure_reason=f'{type(error).__name__}: {error}', worker_pids={}, stopped_at=now())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'wait', 'run', 'worker'))
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--stage', choices=MODELS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--mode', choices=('train', 'test'))
    args = parser.parse_args()
    global RUN_ID, RUN
    run = Path(args.run_root)
    if not run.is_absolute():
        raise ValueError('run-root must be absolute')
    run = run.resolve()
    prefix = 'baseline_d1_urpc2020_'
    if (run.parent != ROOT / 'runs' or not run.name.startswith(prefix)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,80}', run.name[len(prefix):])):
        raise ValueError('run-root must contain a valid immutable baseline-D1 run ID')
    RUN_ID = run.name[len(prefix):]
    RUN = run
    if args.action == 'preflight':
        preflight(run)
    elif args.action == 'worker':
        if args.stage is None or args.seed is None or args.mode is None:
            parser.error('worker requires stage, seed and mode')
        worker(run, args.stage, args.seed, args.mode)
    else:
        manage(run, wait=args.action == 'wait')


if __name__ == '__main__':
    main()
