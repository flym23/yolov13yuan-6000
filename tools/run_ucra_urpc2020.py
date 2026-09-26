#!/usr/bin/env python3
"""PID exit -> A0 -> A4 -> B1 -> B2 -> B3; three seeds per stage, fail fast."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ucra_urpc2020_common import ROOT, DATA, MODELS, SETTINGS, now, require_run_root, sha256, write_json, verify_contract
from tools.ucra_process_gate import is_original_running, wait_for_exit
from tools.run_ucra_v2 import load, write_csv, summarize, stop_processes


def completed_worker(run, stage, seed, mode):
    folder = run / mode / f'seed{seed}' / stage
    marker = folder / ('train_complete.json' if mode == 'train' else 'summary_metrics.json')
    if not marker.is_file():
        return False
    if mode == 'test':
        if not (folder / 'scale_ap_metrics.json').is_file():
            raise RuntimeError(f'Incomplete test output: {folder}')
        return True
    result = load(marker)
    if (result.get('epochs') != 180 or result.get('settings') != SETTINGS
            or result.get('early_stopping') is not False or result.get('data') != str(DATA)
            or result.get('stage') != stage or result.get('seed') != seed):
        raise RuntimeError(f'Invalid completed training: {folder}')
    if sha256(folder / 'weights/best.pt') != result['weights_sha256']:
        raise RuntimeError(f'Completed checkpoint changed: {folder}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'chain.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = run / 'state.json'
    old = load(state_path) if state_path.exists() else {}
    if old and not args.resume:
        raise RuntimeError('Existing run_id: explicit --resume required')
    if old.get('status') == 'completed':
        raise RuntimeError('This run already completed')
    # Never start a replacement batch while an orphaned old worker still exists.
    for pid in old.get('worker_pids', {}).values():
        if Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'Previous worker {pid} still exists; recovery needs review')
    state = dict(old)
    state.update(run_id=run.name.removeprefix('ucra_urpc2020_'), run_root=str(run),
        data=str(DATA), stages=list(MODELS), seeds=[0, 1, 2], launcher_pid=os.getpid(), worker_pids={},
        created_at=old.get('created_at', now()), failure_reason=None, upstream_state=None,
        trigger='exit of the original PID 3645793 (explicit user instruction)',
        upstream_status=old.get('upstream_status', 'running'), upstream_failure_reason=None)
    (run / 'launcher.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
    children = []

    def update(status, **extra):
        state.update(status=status, updated_at=now(), **extra)
        write_json(state_path, state)
        print(now(), json.dumps({k: state.get(k) for k in ('status', 'stage', 'phase', 'worker_pids', 'failure_reason')}), flush=True)

    def cancelled(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')

    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        if load(run / 'train/preflight.json')['status'] != 'passed':
            raise RuntimeError('Preflight not passed')
        contract = verify_contract(run)
        upstream = contract['upstream_process']
        state['upstream_process'] = upstream
        if args.wait:
            update('waiting', phase='waiting_for_process_exit')
            print('WAIT_BEGIN', json.dumps(upstream), flush=True)
            observation = wait_for_exit(upstream, lambda text: print(now(), text, flush=True))
            state.update(upstream_status='exited', upstream_exit_observation=observation,
                upstream_exit_observed_at=now(), upstream_exit_code=None,
                upstream_exit_note='Not our child process; exit code and success/failure are unavailable.')
            print('UPSTREAM_EXIT', json.dumps(state['upstream_process']), observation, flush=True)
            update('waiting', phase='handoff_to_run_script')
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            os.execv('/bin/bash', ['bash', str(ROOT / 'scripts/run_ucra_urpc2020.sh'),
                     str(run), str(run / 'train/launcher.log'), '--resume'])
        if is_original_running(upstream):
            raise RuntimeError('Upstream process still running; use the wait script')
        state.update(upstream_status='exited', upstream_exit_code=None,
                     upstream_exit_observed_at=state.get('upstream_exit_observed_at', now()))
        update('running', started_at=state.get('started_at', now()))
        all_rows, mean_rows = [], []
        # A0 and A4 are freshly trained here, never imported from the half dataset.
        for stage in MODELS:
            for mode in ('train', 'test'):
                children = []
                update('running', stage=stage, phase=mode, worker_pids={})
                for seed in range(3):
                    folder = run / mode / f'seed{seed}' / stage
                    if args.resume and completed_worker(run, stage, seed, mode):
                        continue
                    if folder.exists():
                        raise RuntimeError(f'Incomplete output needs explicit review: {folder}')
                    log = run / mode / f'{stage}_seed{seed}.worker.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    stream = log.open('a', encoding='utf-8')
                    command = [sys.executable, '-u', str(ROOT / 'tools/train_ucra_urpc2020_worker.py'),
                        '--run-root', str(run), '--stage', stage, '--seed', str(seed), '--mode', mode]
                    try:
                        process = subprocess.Popen(command, cwd='/tmp', stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        stream.close()
                        raise
                    children.append((process, stream))
                    state['worker_pids'][str(seed)] = process.pid
                    (run / f'{mode}_seed{seed}.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
                    update('running')
                pending = {process.pid: process for process, _ in children}
                while pending:
                    for pid, process in list(pending.items()):
                        code = process.poll()
                        if code is not None:
                            if code != 0:
                                raise RuntimeError(f'{stage}/{mode} worker {pid} failed: exit {code}')
                            del pending[pid]
                    if pending:
                        time.sleep(2)
                for _, stream in children:
                    stream.close()
                children = []
                update('running', worker_pids={})
            rows, stats = summarize(run, stage)
            all_rows.extend(rows)
            mean_rows.extend(dict(stage=stage, metric=key, **value) for key, value in stats.items())
            write_csv(run / 'test/summary.csv', all_rows)
            write_csv(run / 'test/summary_mean_std.csv', mean_rows)
            write_json(run / 'test/AllSeed_summary.json', dict(data=str(DATA), run_root=str(run),
                trained_stages=list(MODELS)[:list(MODELS).index(stage) + 1], reference_stages=[],
                seeds=all_rows, statistics=mean_rows, std_ddof=1, updated_at=now()))
            update('running', phase='summarized')
        update('completed', phase='completed', completed_at=now(), worker_pids={})
    except BaseException as error:
        stop_processes(children)
        for _, stream in children:
            stream.close()
        update('cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
               failure_reason=f'{type(error).__name__}: {error}', worker_pids={})
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
