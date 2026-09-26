#!/usr/bin/env python3
"""Recoverable C0 -> C1 -> C2 chain, three managed seeds at every stage."""
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
from tools.ucra_v4_common import ROOT, DATA, MODELS, SETTINGS, UPSTREAM_STATE, now, require_run_root, sha256, write_json, verify_contract
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
    if (result.get('stage') != stage or result.get('seed') != seed
            or result.get('data') != str(DATA) or result.get('epochs') != 180
            or result.get('settings') != SETTINGS or result.get('early_stopping') is not False):
        raise RuntimeError(f'Invalid completed training: {folder}')
    if sha256(folder / 'weights/best.pt') != result['weights_sha256']:
        raise RuntimeError(f'Completed checkpoint changed: {folder}')
    return True


def finalize(run, rows, statistics_rows):
    metrics = ('P', 'R', 'mAP50', 'mAP50_95', 'AP_S', 'AP_M', 'AP_L')

    def values(stage, metric):
        selected = {row['seed']: row[metric] for row in rows if row['stage'] == stage}
        if set(selected) != {0, 1, 2}:
            raise RuntimeError(f'Incomplete three-seed comparison for {stage}')
        return [selected[seed] for seed in range(3)]

    comparisons = {}
    for candidate in MODELS:
        for reference in ('A4', 'B2'):
            comparisons[f'{candidate}_vs_{reference}'] = {}
            for metric in metrics:
                delta = [a - b for a, b in zip(values(candidate, metric), values(reference, metric))]
                comparisons[f'{candidate}_vs_{reference}'][metric] = dict(
                    per_seed_delta=delta, mean_delta=statistics.mean(delta),
                    std_delta=statistics.stdev(delta), positive_seeds=sum(x > 0 for x in delta),
                    negative_seeds=sum(x < 0 for x in delta))
    for metric in metrics:
        delta = [a - b for a, b in zip(values('C2', metric), values('C1', metric))]
        comparisons['C2_vs_C1'] = comparisons.get('C2_vs_C1', {})
        comparisons['C2_vs_C1'][metric] = dict(per_seed_delta=delta, mean_delta=statistics.mean(delta),
            std_delta=statistics.stdev(delta), positive_seeds=sum(x > 0 for x in delta),
            negative_seeds=sum(x < 0 for x in delta))
    acceptance = {}
    for candidate in ('C1', 'C2'):
        contrast = comparisons[f'{candidate}_vs_A4']
        checks = dict(mean_mAP50_95_gt_A4=contrast['mAP50_95']['mean_delta'] > 0,
            primary_positive_at_least_two_seeds=contrast['mAP50_95']['positive_seeds'] >= 2,
            AP_S_mean_drop_at_most_0_20pp=contrast['AP_S']['mean_delta'] >= -0.002,
            recall_mean_drop_at_most_0_30pp=contrast['R']['mean_delta'] >= -0.003,
            no_persistent_AP_M_decline=not (contrast['AP_M']['mean_delta'] < 0 and contrast['AP_M']['negative_seeds'] >= 2),
            no_persistent_AP_L_decline=not (contrast['AP_L']['mean_delta'] < 0 and contrast['AP_L']['negative_seeds'] >= 2),
            precision_mean_or_stability_improved=(contrast['P']['mean_delta'] > 0 or
                statistics.stdev(values(candidate, 'P')) < statistics.stdev(values('A4', 'P'))))
        acceptance[candidate] = dict(checks=checks, accepted=all(checks.values()))
    accepted = [stage for stage in ('C1', 'C2') if acceptance[stage]['accepted']]
    preferred = max(accepted, key=lambda stage: statistics.mean(values(stage, 'mAP50_95'))) if accepted else None
    write_csv(run / 'test/seed0_comparison.csv', [row for row in rows if row['seed'] == 0])
    decision = dict(acceptance=acceptance, preferred_structure=preferred,
        c0_c1_c2_all_ran_three_seeds_by_user_request=True, c2_unconditional=True,
        phase2_executed=False)
    write_json(run / 'test/decision.json', decision)
    write_json(run / 'test/AllSeed_summary.json', dict(data=str(DATA), seeds=rows,
        statistics=statistics_rows, comparisons=comparisons, **decision,
        reference_stages=['A0', 'A4', 'B2'], trained_stages=list(MODELS), std_ddof=1))


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
    for pid in old.get('worker_pids', {}).values():
        if Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'Previous worker {pid} still exists; recovery needs review')
    state = dict(old)
    state.update(run_id=run.name.removeprefix('ucra_v4_'), run_root=str(run),
        data=str(DATA), stages=list(MODELS), seeds=[0, 1, 2],
        created_at=old.get('created_at', now()), launcher_pid=os.getpid(), worker_pids={},
        upstream_state=str(UPSTREAM_STATE), upstream_status=old.get('upstream_status'),
        upstream_failure_reason=old.get('upstream_failure_reason'), failure_reason=None)
    (run / 'launcher.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
    children = []

    def update(status, **extra):
        state.update(status=status, updated_at=now(), **extra)
        write_json(state_path, state)
        print(now(), json.dumps({key: state.get(key) for key in
            ('status', 'stage', 'phase', 'worker_pids', 'failure_reason')}), flush=True)

    def cancelled(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')

    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        if load(run / 'train/preflight.json')['status'] != 'passed':
            raise RuntimeError('Preflight not passed')
        verify_contract(run)
        update('waiting', phase='waiting_for_upstream')
        print('WAIT_BEGIN', UPSTREAM_STATE, flush=True)
        while True:
            try:
                upstream = load(UPSTREAM_STATE)
            except (OSError, ValueError, TypeError):
                upstream = {}
            if isinstance(upstream, dict) and upstream.get('status') in ('completed', 'failed', 'cancelled'):
                state['upstream_status'] = upstream['status']
                state['upstream_failure_reason'] = upstream.get('failure_reason') or upstream.get('failed_reason')
                print('UPSTREAM_TERMINAL', state['upstream_status'], state['upstream_failure_reason'], flush=True)
                break
            if not args.wait:
                raise RuntimeError('Upstream not terminal; use wait script')
            time.sleep(30)
        update('waiting', phase='upstream_terminal')
        if args.wait:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            os.execv('/bin/bash', ['bash', str(ROOT / 'scripts/run_ucra_v4.sh'),
                str(run), str(run / 'train/launcher.log'), '--resume'])
        update('running', started_at=state.get('started_at', now()))
        all_rows, mean_rows = [], []
        for reference in ('A0', 'A4', 'B2'):
            snapshot = load(run / 'test/references' / f'{reference}_AllSeed_summary.json')
            all_rows.extend(snapshot['seeds'])
            mean_rows.extend(dict(stage=reference, metric=key, **value)
                             for key, value in snapshot['statistics'].items())
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
                    command = [sys.executable, '-u', str(ROOT / 'tools/train_ucra_v4_worker.py'),
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
            mean_rows.extend(dict(stage=stage, metric=metric, **values) for metric, values in stats.items())
            write_csv(run / 'test/summary.csv', all_rows)
            write_csv(run / 'test/summary_mean_std.csv', mean_rows)
            write_json(run / 'test/AllSeed_summary.json', dict(data=str(DATA), seeds=all_rows,
                statistics=mean_rows, reference_stages=['A0', 'A4', 'B2'],
                trained_stages=list(MODELS)[:list(MODELS).index(stage) + 1], std_ddof=1))
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


if __name__ == '__main__':
    main()
