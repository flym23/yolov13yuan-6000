#!/usr/bin/env python3
"""Resume only interrupted UCRA-v4 C2 seeds, then rejoin the frozen chain."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_ucra_v2 import load, stop_processes
from tools.run_ucra_v4 import completed_worker
from tools.train_ucra_v4_worker import no_early_stop
from tools.ucra_v4_common import DATA, SETTINGS, now, require_run_root, sha256, verify_contract, write_json

SEEDS = (0, 1, 2)
BACKUP_FILES = ('results.csv', 'args.yaml', 'best_epoch.json', 'weights/last.pt', 'weights/best.pt')


def train_folder(run: Path, seed: int) -> Path:
    return run / 'train' / f'seed{seed}' / 'C2'


def check_seed(run: Path, seed: int) -> dict:
    import torch

    folder = train_folder(run, seed)
    if (folder / 'train_complete.json').exists():
        raise RuntimeError(f'C2 seed {seed} is already complete')
    weights = folder / 'weights/last.pt'
    with (folder / 'results.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    args = checkpoint.get('train_args') or {}
    expected = dict(epochs=180, batch=16, imgsz=640, device=0, workers=2,
                    amp=False, deterministic=True, plots=False, seed=seed,
                    data=str(DATA), project=str(folder.parent), name='C2')
    bad = {key: (args.get(key), value) for key, value in expected.items() if args.get(key) != value}
    if bad or args.get('save_dir') != str(folder):
        raise RuntimeError(f'C2 seed {seed} checkpoint arguments differ: {bad}; save_dir={args.get("save_dir")}')
    if checkpoint.get('optimizer') is None or checkpoint.get('ema') is None:
        raise RuntimeError(f'C2 seed {seed} checkpoint lacks optimizer or EMA')
    completed = int(checkpoint.get('epoch', -2)) + 1
    if completed != len(rows) or not 0 < completed < SETTINGS['epochs']:
        raise RuntimeError(f'C2 seed {seed} checkpoint/CSV mismatch: {completed} vs {len(rows)}')
    return dict(seed=seed, completed_epochs=completed, checkpoint=str(weights),
                checkpoint_sha256=sha256(weights))


def backup(run: Path, checkpoints: list[dict], state_path: Path) -> Path:
    destination = ROOT / '.codex_backups' / ('ucra_v4_c2_resume_' + time.strftime('%Y%m%d_%H%M%S'))
    if destination.exists():
        raise FileExistsError(destination)
    manifest = []
    for seed in (item['seed'] for item in checkpoints):
        folder = train_folder(run, seed)
        for relative in BACKUP_FILES:
            source = folder / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            target = destination / 'train' / f'seed{seed}' / 'C2' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha256(source)
            if sha256(target) != digest:
                raise RuntimeError(f'Backup checksum differs: {source}')
            manifest.append(dict(source=str(source), backup=str(target), size=source.stat().st_size,
                                 sha256=digest))
    state_copy = destination / 'state_before_resume.json'
    shutil.copy2(state_path, state_copy)
    write_json(destination / 'manifest.json', dict(run_root=str(run), checkpoints=checkpoints,
                                                   files=manifest, state_sha256=sha256(state_copy)))
    return destination


def worker(run: Path, seed: int) -> None:
    from ultralytics import YOLO

    verify_contract(run)
    folder = train_folder(run, seed)
    model = YOLO(str(folder / 'weights/last.pt'))
    model.add_callback('on_train_start', no_early_stop)

    def record_best_epoch(trainer):
        if trainer.best_fitness == trainer.fitness:
            write_json(folder / 'best_epoch.json', dict(epoch=trainer.epoch + 1,
                                                       fitness=float(trainer.fitness)))

    model.add_callback('on_model_save', record_best_epoch)
    started = perf_counter()
    print(f'RESUME_C2 seed={seed} checkpoint={folder / "weights/last.pt"}', flush=True)
    model.train(resume=True, batch=16, imgsz=640, device=0)
    if model.trainer.save_dir.resolve() != folder.resolve():
        raise RuntimeError(f'Unexpected resume output: {model.trainer.save_dir}')
    if model.trainer.epoch + 1 != SETTINGS['epochs']:
        raise RuntimeError(f'C2 seed {seed} stopped at epoch {model.trainer.epoch + 1}')
    with (folder / 'results.csv').open(encoding='utf-8') as stream:
        rows = [{key.strip(): float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if len(rows) != 180 or not all(math.isfinite(value) for row in rows for value in row.values()):
        raise RuntimeError(f'Invalid C2 seed {seed} training results')
    if [int(row['epoch']) for row in rows] != list(range(1, 181)):
        raise RuntimeError(f'C2 seed {seed} has missing or duplicate epochs')
    best = folder / 'weights/best.pt'
    if not best.is_file():
        raise FileNotFoundError(best)
    best_epoch = load(folder / 'best_epoch.json')['epoch']
    write_json(folder / 'train_complete.json', dict(stage='C2', seed=seed, data=str(DATA),
        epochs=len(rows), best_epoch=best_epoch, settings=SETTINGS, early_stopping=False,
        seconds=perf_counter() - started, completed_at=now(), weights_sha256=sha256(best)))
    print(f'RESUME_C2_COMPLETE seed={seed} epochs={len(rows)} best_epoch={best_epoch}', flush=True)


def manager(run: Path) -> None:
    lock = (run / 'chain.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = run / 'state.json'
    state = load(state_path)
    children = []

    def update(status: str, **extra) -> None:
        state.update(status=status, updated_at=now(), **extra)
        write_json(state_path, state)
        print(now(), json.dumps({key: state.get(key) for key in
              ('status', 'stage', 'phase', 'worker_pids', 'failure_reason')}), flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if (state.get('status') != 'failed' or state.get('stage') != 'C2'
                or state.get('phase') not in ('train', 'train_resume')):
            raise RuntimeError('Expected failed C2 training state')
        for pid in [state.get('launcher_pid'), *state.get('worker_pids', {}).values()]:
            if pid and Path(f'/proc/{pid}').exists():
                raise RuntimeError(f'Original process is still present: {pid}')
        verify_contract(run)
        if load(run / 'train/preflight.json')['status'] != 'passed':
            raise RuntimeError('Original preflight did not pass')
        for stage in ('C0', 'C1'):
            for seed in SEEDS:
                for mode in ('train', 'test'):
                    if not completed_worker(run, stage, seed, mode):
                        raise RuntimeError(f'Incomplete previous stage: {stage} seed{seed} {mode}')
        pending_seeds = [seed for seed in SEEDS if not completed_worker(run, 'C2', seed, 'train')]
        checkpoints = [check_seed(run, seed) for seed in pending_seeds]
        backup_path = backup(run, checkpoints, state_path) if checkpoints else None
        print('RECOVERY_BACKUP', backup_path, flush=True)
        print('RECOVERY_CHECKPOINTS', json.dumps(checkpoints), flush=True)
        state.update(previous_failure_reason=state.get('failure_reason'),
                     recovery_backup=str(backup_path) if backup_path else state.get('recovery_backup'),
                     recovery_checkpoints=checkpoints,
                     launcher_pid=os.getpid(), failure_reason=None, worker_pids={})
        (run / 'launcher.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
        update('running', stage='C2', phase='train_resume')
        for seed in pending_seeds:
            log = run / 'train' / f'C2_seed{seed}.resume.worker.log'
            stream = log.open('a', encoding='utf-8')
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-root', str(run),
                       '--seed', str(seed)]
            try:
                process = subprocess.Popen(command, cwd='/tmp', stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                stream.close()
                raise
            children.append((process, stream))
            state['worker_pids'][str(seed)] = process.pid
            (run / f'train_seed{seed}.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
            update('running')
        pending = {process.pid: process for process, _ in children}
        while pending:
            for pid, process in list(pending.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'C2 resume worker {pid} exited {code}')
                    del pending[pid]
            if pending:
                time.sleep(2)
        for _, stream in children:
            stream.close()
        children = []
        for seed in SEEDS:
            if not completed_worker(run, 'C2', seed, 'train'):
                raise RuntimeError(f'C2 seed {seed} completion marker invalid')
        update('running', phase='train_resumed', worker_pids={})
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        os.execv(sys.executable, [sys.executable, '-u', str(ROOT / 'tools/run_ucra_v4.py'),
                                   '--run-root', str(run), '--resume'])
    except BaseException as error:
        stop_processes(children)
        for _, stream in children:
            stream.close()
        update('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
               failure_reason=f'{type(error).__name__}: {error}', worker_pids={})
        raise
    finally:
        if not lock.closed:
            lock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    arguments = parser.parse_args()
    run = require_run_root(arguments.run_root)
    if arguments.seed is None:
        manager(run)
    else:
        worker(run, arguments.seed)


if __name__ == '__main__':
    main()
