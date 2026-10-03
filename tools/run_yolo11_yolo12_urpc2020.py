#!/usr/bin/env python3
"""Fresh YOLO11n -> YOLO12n chain: three seeds, 180 full epochs, fail-fast supervision."""
from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ultralytics

if not Path(ultralytics.__file__).resolve().is_relative_to(ROOT):
    raise ImportError(f'External Ultralytics: {ultralytics.__file__}')

PREFIX = 'yolo11_yolo12_urpc2020_'
DATA = Path('/home/room305/ZZF/URPC2020/data.yaml')
UPSTREAM = ROOT / 'runs/d1_dualseed150_urpc2020_20261002_r1/state.json'
STAGES = ('YOLO11', 'YOLO12')
SEEDS = (0, 1, 2)
WEIGHTS = {stage: Path(f'/home/room305/ZZF/yolo{version}n.pt')
           for stage, version in zip(STAGES, (11, 12))}
SETTINGS = dict(epochs=180, device=0, workers=2, amp=False, deterministic=True,
                plots=False, imgsz=640, batch=16)
TERMINAL = ('completed', 'failed', 'cancelled')
EVALUATOR = ROOT / 'tools/d1_dualseed_legacy_test.py'
SHELLS = ('scripts/run_yolo11_yolo12_urpc2020.sh',
          'scripts/wait_yolo11_yolo12_urpc2020_after_d1_dualseed150.sh')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def require_run(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('run-root must be absolute')
    path = path.resolve()
    if path.parent != ROOT / 'runs' or not re.fullmatch(PREFIX + r'[A-Za-z0-9][A-Za-z0-9_-]{2,80}', path.name):
        raise ValueError(f'Unexpected immutable run-root: {path}')
    return path


def upstream_terminal():
    try:
        value = load(UPSTREAM)
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) and value.get('status') in TERMINAL else None


def verify_contract(run):
    frozen = load(run / 'train/contract.json')
    if (frozen['run_id'] != run.name.removeprefix(PREFIX) or frozen['stages'] != list(STAGES)
            or frozen['seeds'] != list(SEEDS) or frozen['data'] != str(DATA)
            or frozen['settings'] != SETTINGS or frozen['scale'] != 'n'
            or frozen['early_stopping'] is not False or 'patience' in frozen['settings']):
        raise RuntimeError('Training contract changed')
    if sha256(DATA) != frozen['dataset_yaml_sha256']:
        raise RuntimeError('Dataset configuration changed')
    for stage in STAGES:
        if (frozen['models'][stage]['pretrained'] != str(WEIGHTS[stage])
                or sha256(WEIGHTS[stage]) != frozen['models'][stage]['pretrained_sha256']):
            raise RuntimeError(f'{stage} initialization weights changed')
    for base, key in ((ROOT, 'source_hashes'), (run, 'snapshot_hashes')):
        for relative, expected in frozen[key].items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base) or sha256(path) != expected:
                raise RuntimeError(f'Frozen dependency changed: {path}')
    return frozen


def import_evaluator(path):
    spec = importlib.util.spec_from_file_location('baseline_comparison_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'main', None)):
        raise RuntimeError('Evaluator has no main entrypoint')
    return module


def verify_transfer(target, source, allow_class_resize=False):
    import torch
    source, target = source.state_dict(), target.state_dict()
    matched = [k for k in target if k in source and source[k].shape == target[k].shape]
    different = sorted(set(source) ^ set(target))
    different += [k for k in source.keys() & target.keys() if source[k].shape != target[k].shape]
    head = f'model.{max(int(k.split(".")[1]) for k in target if k.startswith("model."))}.cv3.'
    invalid = [k for k in different if not (allow_class_resize and k.startswith(head))]
    if invalid or not matched:
        raise RuntimeError(f'Unexpected architecture/weight mismatch: {invalid}')
    if not all(torch.equal(target[k], source[k]) for k in matched):
        raise RuntimeError('Pretrained tensors were not loaded exactly')
    return dict(loaded_tensors=len(matched), target_tensors=len(target), class_resize_keys=sorted(different))


def baseline_criterion(model):
    from ultralytics.nn.modules import Detect
    from ultralytics.utils import DEFAULT_CFG
    model.args = DEFAULT_CFG
    criterion = model.init_criterion()
    if (type(model.model[-1]) is not Detect or criterion.use_qprr or criterion.use_sqanwd
            or criterion.has_quality_branch or getattr(criterion, 'use_gbc_qprr', False)):
        raise RuntimeError('Expected original Detect and standard detection loss')


def preflight(run):
    import yaml
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT
    if run.exists():
        raise FileExistsError(f'Run ID already exists: {run}')
    upstream = upstream_terminal()
    if not upstream:
        raise RuntimeError('Previous D1 chain must be terminal before deployment')
    data = check_det_dataset(str(DATA), autodownload=False)
    for split in ('train', 'val', 'test'):
        values = data.get(split)
        if not values:
            raise RuntimeError(f'Missing dataset split: {split}')
        for value in values if isinstance(values, list) else [values]:
            path = Path(value).resolve()
            if not path.exists() or not path.is_relative_to(DATA.parent):
                raise RuntimeError(f'Unexpected URPC2020 split: {path}')
    import_evaluator(EVALUATOR)
    audits, configs = {}, {}
    for stage in STAGES:
        if stage == 'YOLO12':
            from tools.yolo12_checkpoint_compat import install, verify_attention
            install(WEIGHTS[stage])
            verify_attention()
        source = YOLO(str(WEIGHTS[stage]))
        cfg = json.loads(json.dumps(source.model.yaml))
        if cfg.get('scale') != 'n' or cfg.get('nc') != 80:
            raise RuntimeError(f'{stage} checkpoint must be original nano COCO model')
        if set(cfg) - {'nc', 'scales', 'backbone', 'head', 'scale', 'yaml_file', 'ch'}:
            raise RuntimeError(f'Unexpected custom model configuration: {stage}')
        modules = {layer[2] for layer in cfg['backbone'] + cfg['head']}
        if (stage == 'YOLO11' and not {'C2PSA', 'SPPF'} <= modules
                or stage == 'YOLO12' and ('A2C2f' not in modules or 'C2PSA' in modules)):
            raise RuntimeError(f'Incorrect model family: {stage}')
        configs[stage] = {k: v for k, v in cfg.items() if k != 'yaml_file'}
        rebuilt = DetectionModel(cfg, verbose=False)
        rebuilt.load(source.model)
        exact = verify_transfer(rebuilt, source.model)
        baseline_criterion(rebuilt)
        import torch
        source.model.eval()
        rebuilt.eval()
        with torch.no_grad():
            image = torch.randn(1, 3, 128, 128)
            actual, expected = rebuilt(image)[0], source.model(image)[0]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            forward_max_abs_difference = float((actual - expected).abs().max())
        with tempfile.TemporaryDirectory(prefix='yolo_baseline_model_', dir='/tmp') as directory:
            path = Path(directory) / f'yolo{11 if stage == "YOLO11" else 12}n-urpc2020.yaml'
            path.write_text(yaml.safe_dump(configs[stage], sort_keys=False), encoding='utf-8')
            from_yaml = YOLO(str(path)).load(str(WEIGHTS[stage]))
            verify_transfer(from_yaml.model, source.model)
            baseline_criterion(from_yaml.model)
            del from_yaml
        target = DetectionModel(cfg, nc=data['nc'], verbose=False)
        target.load(source.model)
        transfer = verify_transfer(target, source.model, allow_class_resize=True)
        baseline_criterion(target)
        audits[stage] = dict(exact_checkpoint_architecture=exact, dataset_transfer=transfer,
            parameters=sum(p.numel() for p in target.parameters()), nc=data['nc'],
            forward_max_abs_difference=forward_max_abs_difference)
        del target, rebuilt, source
    for relative in SHELLS:
        subprocess.run(['bash', '-n', str(ROOT / relative)], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_yolo11_yolo12_chain.py')],
                   cwd='/tmp', check=True, timeout=180)
    files = {str(p.relative_to(ROOT)) for p in (ROOT / 'ultralytics').rglob('*.py')}
    files.update({str(Path(__file__).relative_to(ROOT)), str(EVALUATOR.relative_to(ROOT)),
                  'tools/yolo12_checkpoint_compat.py', 'ultralytics/cfg/default.yaml',
                  'tests/test_yolo11_yolo12_chain.py', *SHELLS})
    for relative in files:
        if relative.endswith('.py'):
            ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'), filename=relative)
    snapshots = run / 'train/snapshots'
    snapshots.mkdir(parents=True)
    (run / 'test').mkdir()
    models = {}
    for stage, cfg in configs.items():
        name = f'yolo{11 if stage == "YOLO11" else 12}n-urpc2020.yaml'
        (snapshots / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        models[stage] = dict(model_snapshot=str((snapshots / name).relative_to(run)),
            pretrained=str(WEIGHTS[stage]), pretrained_sha256=sha256(WEIGHTS[stage]))
        for mode in ('train', 'test'):
            (run / mode / stage).mkdir()
    shutil.copy2(DATA, snapshots / 'data.yaml')
    shutil.copy2(EVALUATOR, snapshots / 'evaluation.py')
    write_json(snapshots / 'default_training_arguments.json', DEFAULT_CFG_DICT)
    write_json(run / 'train/contract.json', dict(run_id=run.name.removeprefix(PREFIX),
        stages=list(STAGES), seeds=list(SEEDS), data=str(DATA), settings=SETTINGS, scale='n',
        early_stopping=False, models=models, dataset_yaml_sha256=sha256(DATA),
        source_hashes={rel: sha256(ROOT / rel) for rel in sorted(files)},
        snapshot_hashes={str(p.relative_to(run)): sha256(p) for p in snapshots.iterdir()},
        upstream_state=str(UPSTREAM), created_at=now()))
    write_json(run / 'train/model_audit.json', audits)
    write_json(run / 'train/preflight.json', dict(status='passed', checked_at=now(),
        imported_ultralytics=str(Path(ultralytics.__file__).resolve()), syntax_checked=len(files),
        upstream_status=upstream['status'], upstream_failure_reason=upstream.get('failure_reason'), models=audits))
    verify_contract(run)
    print('YOLO11_YOLO12_PREFLIGHT_PASSED', json.dumps(audits), flush=True)


def disable_early_stopping(trainer):
    trainer.stopper.patience = math.inf
    trainer.stopper.possible_stop = False
    print('EARLY_STOPPING_DISABLED: complete all 180 epochs', flush=True)


def validate_completion(rows):
    if ([int(row['epoch']) for row in rows] != list(range(1, SETTINGS['epochs'] + 1))
            or not all(math.isfinite(float(value)) for row in rows for value in row.values())):
        raise RuntimeError('Must complete all 180 epochs with finite results')


def worker(run, stage, seed, mode):
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    frozen = verify_contract(run)
    if stage == 'YOLO12':
        from tools.yolo12_checkpoint_compat import install
        install(WEIGHTS[stage])
    output = run / mode / stage / f'seed{seed}'
    train = run / 'train' / stage / f'seed{seed}'
    if output.exists():
        raise FileExistsError(f'Refusing duplicate output: {output}')
    print(f'LOCAL_ULTRALYTICS={ultralytics.__file__}; stage={stage}; seed={seed}; mode={mode}', flush=True)
    if mode == 'test':
        best = train / 'weights/best.pt'
        complete = load(train / 'train_complete.json')
        if complete['epochs'] != 180 or sha256(best) != complete['weights_sha256']:
            raise RuntimeError('Training completion or checkpoint hash invalid')
        evaluator = run / 'train/snapshots/evaluation.py'
        sys.argv = [str(evaluator), '--weights', str(best), '--data', str(DATA),
                    '--project', str(output.parent), '--name', output.name, '--device', '0',
                    '--batch', '16', '--imgsz', '640', '--workers', '2', '--no-plots']
        import_evaluator(evaluator).main()
        for name in ('summary_metrics.json', 'scale_ap_metrics.json'):
            if not (output / name).is_file():
                raise FileNotFoundError(output / name)
        return
    init_seeds(seed, deterministic=True)
    model = YOLO(str(run / frozen['models'][stage]['model_snapshot'])).load(str(WEIGHTS[stage]))
    source = YOLO(str(WEIGHTS[stage])).model.float()
    transfer = verify_transfer(model.model, source)
    write_json(run / 'train' / stage / f'seed{seed}.pretrained_load.json', transfer)

    def check_recipe(trainer):
        actual = yaml.safe_load((output / 'args.yaml').read_text(encoding='utf-8'))
        if (any(actual.get(k) != v for k, v in SETTINGS.items()) or actual['seed'] != seed
                or actual['resume'] is not False or actual['pretrained'] != str(WEIGHTS[stage])
                or not math.isinf(trainer.stopper.patience) or trainer.model.yaml.get('scale') != 'n'):
            raise RuntimeError('Effective training recipe/scale/early stopping differs')
        transfer = verify_transfer(trainer.model, source.to(trainer.device), allow_class_resize=True)
        write_json(output / 'effective_training.json', dict(settings=SETTINGS, seed=seed,
                   stage=stage, scale='n', early_stopping=False, resume=False, transfer=transfer))
        print('EFFECTIVE_RECIPE_VERIFIED; SCALE=n; EPOCHS=180; EARLY_STOPPING_DISABLED; RESUME=False', flush=True)

    def record_best(trainer):
        if trainer.best_fitness == trainer.fitness:
            write_json(output / 'best_epoch.json', dict(epoch=trainer.epoch + 1, fitness=float(trainer.fitness)))

    model.add_callback('on_train_start', disable_early_stopping)
    model.add_callback('on_train_start', check_recipe)
    model.add_callback('on_model_save', record_best)
    started = time.perf_counter()
    model.train(**SETTINGS, data=str(DATA), pretrained=str(WEIGHTS[stage]), seed=seed,
                project=str(output.parent), name=output.name, exist_ok=False, resume=False, time=None)
    if model.trainer.epoch + 1 != SETTINGS['epochs']:
        raise RuntimeError('Training stopped before 180 epochs')
    best = output / 'weights/best.pt'
    if not best.is_file():
        raise FileNotFoundError(best)
    with (output / 'results.csv').open(encoding='utf-8') as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    validate_completion(rows)
    write_json(output / 'train_complete.json', dict(stage=stage, seed=seed, epochs=180,
        early_stopping=False, best_epoch=load(output / 'best_epoch.json')['epoch'],
        weights_sha256=sha256(best), settings=SETTINGS, seconds=time.perf_counter() - started, completed_at=now()))


def stop_processes(children):
    for process, _ in children:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    for process, _ in children:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_batch(run, stage, mode, update):
    children = []
    try:
        pids = {}
        for seed in SEEDS:
            if (run / mode / stage / f'seed{seed}').exists():
                raise FileExistsError(f'Refusing duplicate {stage}/{mode}/seed{seed}')
            stream = (run / mode / stage / f'seed{seed}.worker.log').open('a', encoding='utf-8')
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
            (run / f'{stage}_{mode}_seed{seed}.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
            update('running', phase=mode, stage=stage, worker_pids=dict(pids))
        pending = {p.pid: p for p, _ in children}
        while pending:
            for pid, process in list(pending.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'{stage} {mode} worker {pid} exited {code}')
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


def summarize(run, stage):
    rows = []
    for seed in SEEDS:
        result = load(run / 'test' / stage / f'seed{seed}/summary_metrics.json')
        scale = load(run / 'test' / stage / f'seed{seed}/scale_ap_metrics.json')['metrics']
        train = load(run / 'train' / stage / f'seed{seed}/train_complete.json')
        metrics = result['metrics']
        p, r = metrics['metrics/precision(B)'], metrics['metrics/recall(B)']
        rows.append(dict(stage=stage, seed=seed, P=p, R=r, F1=2*p*r/(p+r) if p+r else 0.,
            mAP50=metrics['metrics/mAP50(B)'], mAP50_95=metrics['metrics/mAP50-95(B)'],
            AP_S=scale['APS']/100, AP_M=scale['APM']/100, AP_L=scale['APL']/100,
            params=result['model']['parameters'], gflops=result['model']['gflops'],
            best_epoch=train['best_epoch'], epochs=train['epochs']))
    keys = [k for k in rows[0] if k not in ('stage', 'seed')]
    if any(row['epochs'] != 180 for row in rows) or not all(math.isfinite(row[k]) for row in rows for k in keys):
        raise RuntimeError('Invalid summary training completion or metrics')
    stats = {k: dict(mean=statistics.mean(row[k] for row in rows), std=statistics.stdev(row[k] for row in rows),
                    max=max(row[k] for row in rows), min=min(row[k] for row in rows)) for k in keys}
    write_csv(run / 'test' / stage / 'AllSeed_summary.csv', rows)
    write_json(run / 'test' / stage / 'AllSeed_summary.json', dict(stage=stage, data=str(DATA), scale='n',
        pretrained=str(WEIGHTS[stage]), settings=SETTINGS, early_stopping=False, seeds=rows, statistics=stats, std_ddof=1))
    return rows, stats


def execute_stages(run, update):
    rows, statistics_by_stage = [], {}
    for stage in STAGES:
        for mode in ('train', 'test'):
            verify_contract(run)
            run_batch(run, stage, mode, update)
        update('running', stage=stage, phase='summary')
        stage_rows, stats = summarize(run, stage)
        rows.extend(stage_rows)
        statistics_by_stage[stage] = stats
        update('running', completed_stages=list(statistics_by_stage))
    write_csv(run / 'test/AllSeed_summary.csv', rows)
    write_json(run / 'test/AllSeed_summary.json', dict(data=str(DATA), scale='n', settings=SETTINGS,
        early_stopping=False, seeds=rows, statistics=statistics_by_stage, completed_stages=list(STAGES), std_ddof=1))


def manage(run, wait):
    verify_contract(run)
    if load(run / 'train/preflight.json').get('status') != 'passed':
        raise RuntimeError('Preflight did not pass')
    with (run / 'train/chain.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = run / 'state.json'
        old = load(path) if path.exists() else {}
        if old and not (not wait and old.get('status') == 'waiting' and old.get('launcher_pid') == os.getpid()):
            raise RuntimeError('Run ID already started; explicit recovery required')
        state = dict(old, run_id=run.name.removeprefix(PREFIX), run_root=str(run), data=str(DATA),
            stages=list(STAGES), scale='n', seeds=list(SEEDS), settings=SETTINGS, early_stopping=False,
            worker_pids={}, launcher_pid=os.getpid(), upstream_state=str(UPSTREAM), failure_reason=None,
            created_at=old.get('created_at', now()))

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
            upstream = upstream_terminal()
            while not upstream:
                if not wait:
                    raise RuntimeError('Upstream is not terminal')
                time.sleep(30)
                upstream = upstream_terminal()
            update('waiting', phase='upstream_terminal', upstream_status=upstream['status'],
                   upstream_failure_reason=upstream.get('failure_reason'))
            print('UPSTREAM_TERMINAL', upstream['status'], upstream.get('failure_reason'), flush=True)
            if wait:
                print('CHAIN_START', now(), flush=True)
                fcntl.flock(lock, fcntl.LOCK_UN)
                os.execv('/bin/bash', ['bash', str(ROOT / SHELLS[0]), str(run), str(run / 'train/launcher.log')])
            update('running', started_at=now(), completed_stages=[])
            execute_stages(run, update)
            update('completed', phase='completed', completed_at=now(), worker_pids={})
        except BaseException as error:
            update('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                   failure_reason=f'{type(error).__name__}: {error}', worker_pids={}, stopped_at=now())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'wait', 'run', 'worker'))
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--mode', choices=('train', 'test'))
    args = parser.parse_args()
    run = require_run(args.run_root)
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
