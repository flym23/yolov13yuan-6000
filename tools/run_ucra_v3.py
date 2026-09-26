#!/usr/bin/env python3
"""Recoverable, fail-fast B1 -> B2 -> B3 chain with three managed seeds per batch."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ucra_v3_common import ROOT, MODELS, REFERENCE_RUN, SETTINGS, now, require_run_root, sha256, write_json, verify_contract
from tools.run_ucra_v2 import load, write_csv, summarize, stop_processes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--upstream-state', type=Path, default=REFERENCE_RUN / 'state.json')
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
    verify_contract(run)
    state_path = run / 'state.json'
    old = load(state_path) if state_path.exists() else {}
    if old and not args.resume and not (old.get('status') == 'waiting' and old.get('launcher_pid') == os.getpid()):
        raise RuntimeError('Existing run_id: explicit --resume required')
    if old.get('status') == 'completed':
        raise RuntimeError('This run already completed')
    state = dict(old)
    state.update(run_id=run.name.removeprefix('ucra_v3_'), run_root=str(run),
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
                except (OSError, ValueError, TypeError):
                    upstream = {}
                if not isinstance(upstream, dict):
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
            os.execv('/bin/bash', ['bash', str(ROOT/'scripts/run_ucra_v3.sh'), str(run), str(run/'train/launcher.log'), '--resume', '--upstream-state', str(args.upstream_state)])
        update('running', started_at=state.get('started_at', now()))
        all_rows, mean_rows = [], []
        for reference in ('A0', 'A4'):
            snapshot = load(run/'test/references'/f'{reference}_AllSeed_summary.json')
            all_rows.extend(snapshot['seeds'])
            mean_rows.extend(dict(stage=reference, metric=key, **value) for key,value in snapshot['statistics'].items())
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
                        if mode == 'train':
                            completed = load(marker)
                            if completed.get('epochs') != 180 or completed.get('settings') != SETTINGS or completed.get('early_stopping'):
                                raise RuntimeError(f'Invalid completion marker: {folder}')
                            if sha256(folder/'weights/best.pt') != completed['weights_sha256']:
                                raise RuntimeError(f'Completed checkpoint changed: {folder}')
                        continue
                    if folder.exists():
                        raise RuntimeError(f'Incomplete output needs explicit review before recovery: {folder}')
                    log = run / mode / f'{stage}_seed{seed}.worker.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    stream = log.open('a', encoding='utf-8')
                    command = [sys.executable, '-u', str(ROOT/'tools/train_ucra_v3_worker.py'),
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
        finalize(run, all_rows, mean_rows)
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

def finalize(run, rows, statistics_rows):
    metrics = ('P', 'R', 'mAP50', 'mAP50_95', 'AP_S', 'AP_M', 'AP_L')
    def values(stage, metric):
        selected = [row for row in rows if row['stage'] == stage]
        if len(selected) != 3 or {row['seed'] for row in selected} != {0, 1, 2}:
            raise RuntimeError(f'Invalid three-seed summary for {stage}')
        return [next(row[metric] for row in selected if row['seed'] == seed) for seed in range(3)]
    comparisons = {}
    pairs = [(candidate, reference) for candidate in MODELS for reference in ('A0', 'A4')]
    pairs += [('B2', 'B1'), ('B3', 'B2')]
    for candidate, reference in pairs:
        comparisons[f'{candidate}_vs_{reference}'] = {}
        for metric in metrics:
            delta = [a-b for a,b in zip(values(candidate, metric), values(reference, metric))]
            comparisons[f'{candidate}_vs_{reference}'][metric] = dict(
                per_seed_delta=delta, mean_delta=statistics.mean(delta),
                std_delta=statistics.stdev(delta), positive_seeds=sum(v > 0 for v in delta),
                negative_seeds=sum(v < 0 for v in delta), median_delta=statistics.median(delta))
    acceptance = {}
    for candidate in MODELS:
        contrast = comparisons[f'{candidate}_vs_A4']
        primary_std = statistics.stdev(values(candidate, 'mAP50_95'))
        checks = dict(
            mean_primary_improved=contrast['mAP50_95']['mean_delta'] > 0,
            primary_positive_at_least_two_seeds=contrast['mAP50_95']['positive_seeds'] >= 2,
            primary_std_within_0_0025=primary_std <= 0.0025,
            mean_small_ap_improved=contrast['AP_S']['mean_delta'] > 0,
            small_ap_not_single_seed=contrast['AP_S']['positive_seeds'] >= 2,
            precision_drop_within_0_5pp=contrast['P']['mean_delta'] >= -0.005,
            recall_drop_within_0_5pp=contrast['R']['mean_delta'] >= -0.005,
            no_persistent_medium_decline=not (contrast['AP_M']['mean_delta'] < 0 and contrast['AP_M']['negative_seeds'] >= 2),
            no_persistent_large_decline=not (contrast['AP_L']['mean_delta'] < 0 and contrast['AP_L']['negative_seeds'] >= 2))
        acceptance[candidate] = dict(checks=checks, accepted=all(checks.values()), primary_std=primary_std)
    accepted = [stage for stage, result in acceptance.items() if result['accepted']]
    preferred = max(accepted, key=lambda s: statistics.mean(values(s, 'mAP50_95'))) if accepted else None
    seed0 = [row for row in rows if row['seed'] == 0]
    write_csv(run/'test/seed0_comparison.csv', seed0)
    write_json(run/'test/decision.json', dict(acceptance=acceptance, preferred_structure=preferred,
        all_candidates_ran_three_seeds_by_user_request=True, phase2_executed=False,
        operational_definitions=dict(small_ap_not_single_seed='positive AP_S delta in at least 2/3 seeds',
            persistent_decline='negative mean delta and at least 2/3 negative per-seed deltas')))
    write_json(run/'test/AllSeed_summary.json', dict(seeds=rows, statistics=statistics_rows,
        comparisons=comparisons, acceptance=acceptance, preferred_structure=preferred,
        reference_stages=['A0', 'A4'], trained_stages=list(MODELS), std_ddof=1, phase2_executed=False))

if __name__ == '__main__':
    main()
