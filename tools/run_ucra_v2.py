#!/usr/bin/env python3
"""Recoverable, fail-fast A0 -> A3 -> A4 chain with three managed seeds per batch."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ucra_v2_common import ROOT, MODELS, now, require_run_root, write_json

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def summarize(run, stage):
    rows = []
    for seed in range(3):
        folder = run / 'test' / f'seed{seed}' / stage
        result = load(folder / 'summary_metrics.json')
        scale = load(folder / 'scale_ap_metrics.json')['metrics']
        train = load(run / 'train' / f'seed{seed}' / stage / 'train_complete.json')
        metrics = result['metrics']
        rows.append(dict(stage=stage, seed=seed, P=metrics['metrics/precision(B)'],
            R=metrics['metrics/recall(B)'], mAP50=metrics['metrics/mAP50(B)'],
            mAP50_95=metrics['metrics/mAP50-95(B)'], AP_S=scale['APS']/100,
            AP_M=scale['APM']/100, AP_L=scale['APL']/100, params=result['model']['parameters'],
            gflops=result['model']['gflops'], best_epoch=train['best_epoch']))
    assert all(math.isfinite(v) for row in rows for k,v in row.items() if k != 'stage')
    stats = {key: dict(mean=statistics.mean(row[key] for row in rows),
                       std=statistics.stdev(row[key] for row in rows),
                       best=max(row[key] for row in rows), worst=min(row[key] for row in rows))
             for key in rows[0] if key not in ('stage','seed')}
    write_json(run / 'test' / f'{stage}_AllSeed_summary.json', dict(seeds=rows, statistics=stats, std_ddof=1))
    write_csv(run / 'test' / f'{stage}_AllSeed_summary.csv', rows)
    return rows, stats

def stop_processes(children):
    # Each worker has its own session, including DataLoader descendants.
    for process, _ in children:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    for process, _ in children:
        try:
            process.wait(timeout=max(0.1, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--upstream-state', type=Path)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'chain.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.upstream_state and not args.upstream_state.is_absolute():
        raise ValueError('upstream state must be absolute')
    if load(run / 'train/preflight.json')['status'] != 'passed':
        raise RuntimeError('Preflight not passed')
    state_path = run / 'state.json'
    old = load(state_path) if state_path.exists() else {}
    if old and not args.resume and not (old.get('status') == 'waiting' and old.get('launcher_pid') == os.getpid()):
        raise RuntimeError('Existing run_id: explicit --resume required')
    if old.get('status') == 'completed':
        raise RuntimeError('This run already completed')
    state = dict(old)
    state.update(run_id=run.name.removeprefix('ucra_v2_'), run_root=str(run),
                 created_at=old.get('created_at',now()), launcher_pid=os.getpid(), worker_pids={},
                 upstream_state=str(args.upstream_state) if args.upstream_state else old.get('upstream_state'),
                 upstream_status=old.get('upstream_status'), upstream_failure_reason=old.get('upstream_failure_reason'),
                 failure_reason=None)
    (run / 'launcher.pid').write_text(str(os.getpid())+'\n', encoding='utf-8')
    children = []
    def update(status, **extra):
        state.update(status=status, updated_at=now(), **extra)
        write_json(state_path, state)
        print(now(), json.dumps({k: state.get(k) for k in ('status','stage','phase','worker_pids','failure_reason')}), flush=True)
    def cancelled(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        if args.upstream_state:
            update('waiting')
            print('WAIT_BEGIN', str(args.upstream_state), flush=True)
            while True:
                try:
                    upstream = load(args.upstream_state)
                except (OSError, ValueError):
                    upstream = {}
                if upstream.get('status') in ('completed','failed','cancelled'):
                    state['upstream_status'] = upstream['status']
                    state['upstream_failure_reason'] = upstream.get('failure_reason') or upstream.get('failed_reason')
                    print('UPSTREAM_TERMINAL', json.dumps(upstream), flush=True)
                    break
                if not args.wait:
                    raise RuntimeError('Upstream not terminal; use wait script')
                time.sleep(30)
            update('waiting')
        if args.wait:
            # Preserve PID across wait -> bash run script -> manager. The replacement reacquires the lock.
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            os.execv('/bin/bash', ['bash', str(ROOT/'scripts/run_ucra_v2.sh'), str(run), '--resume'])
        update('running', started_at=state.get('started_at', now()))
        all_rows, mean_rows = [], []
        for stage in MODELS:
            for mode in ('train','test'):
                children = []
                state['worker_pids'] = {}
                update('running', stage=stage, phase=mode)
                for seed in range(3):
                    folder = run / mode / f'seed{seed}' / stage
                    marker = folder / ('train_complete.json' if mode == 'train' else 'summary_metrics.json')
                    if args.resume and marker.is_file():
                        if mode == 'test' and not (folder/'scale_ap_metrics.json').is_file():
                            raise RuntimeError(f'Incomplete test output: {folder}')
                        if mode == 'train' and not (folder/'weights/best.pt').is_file():
                            raise RuntimeError(f'Missing completed weights: {folder}')
                        continue
                    if folder.exists():
                        raise RuntimeError(f'Incomplete output needs explicit review before recovery: {folder}')
                    log = run / mode / f'{stage}_seed{seed}.worker.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    stream = log.open('a', encoding='utf-8')
                    command = [sys.executable, '-u', str(ROOT/'tools/train_ucra_v2_worker.py'),
                               '--run-root', str(run), '--stage', stage, '--seed', str(seed), '--mode', mode]
                    try:
                        process = subprocess.Popen(command, cwd='/tmp', stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        stream.close()
                        raise
                    children.append((process, stream))
                    state['worker_pids'][str(seed)] = process.pid
                    (run/f'{mode}_seed{seed}.pid').write_text(str(process.pid)+'\n', encoding='utf-8')
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
                state['worker_pids'] = {}
            rows, stats = summarize(run, stage)
            all_rows.extend(rows)
            mean_rows.extend(dict(stage=stage, metric=metric, **values) for metric, values in stats.items())
            write_csv(run/'test/summary.csv', all_rows)
            write_csv(run/'test/summary_mean_std.csv', mean_rows)
            update('running', phase='summarized')
        comparisons = {}
        for base in ('A0','A3'):
            comparisons[f'A4_vs_{base}'] = {}
            for metric in ('P','R','mAP50','mAP50_95','AP_S','AP_M','AP_L'):
                delta = [next(r[metric] for r in all_rows if r['stage']=='A4' and r['seed']==s)
                         - next(r[metric] for r in all_rows if r['stage']==base and r['seed']==s) for s in range(3)]
                comparisons[f'A4_vs_{base}'][metric] = dict(per_seed_delta=delta, mean_delta=statistics.mean(delta),
                                                         positive_seeds=sum(d>0 for d in delta))
        write_json(run/'test/AllSeed_summary.json', dict(seeds=all_rows, statistics=mean_rows,
                   comparisons=comparisons, std_ddof=1, phase2_executed=False))
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
