#!/usr/bin/env python3
"""C2 -> C1 full-URPC2020 chain, three managed seeds at each stage."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_ucra_v2 import load, stop_processes, write_csv
from tools.f1r_urpc2020_common import (
    DATA, MODELS, REFERENCE_STAGES, SEEDS, SETTINGS, UPSTREAM_STATE,
    enrich_reference, now, require_run_root, sha256, statistics_for,
    summarize, verify_contract, write_json,
)


def completed_worker(run: Path, stage: str, seed: int, mode: str) -> bool:
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
            or result.get('data') != str(DATA) or result.get('epochs') != SETTINGS['epochs']
            or result.get('settings') != SETTINGS or result.get('early_stopping') is not False):
        raise RuntimeError(f'Invalid completed training: {folder}')
    if sha256(folder / 'weights/best.pt') != result['weights_sha256']:
        raise RuntimeError(f'Completed checkpoint changed: {folder}')
    return True


def finalize(run: Path, rows: list[dict], statistics_rows: list[dict]) -> None:
    metrics = ('P', 'R', 'F1', 'mAP50', 'mAP50_95', 'AP_S', 'AP_M', 'AP_L')

    def values(stage: str, metric: str) -> list[float]:
        selected = {row['seed']: row[metric] for row in rows if row['stage'] == stage}
        if set(selected) != set(SEEDS):
            raise RuntimeError(f'Incomplete three-seed comparison: {stage}')
        return [selected[seed] for seed in SEEDS]

    comparisons = {}
    for candidate, reference in (('C2', 'A0'), ('C1', 'A0'), ('C1', 'A4'), ('C1', 'C2')):
        comparison = {}
        for metric in metrics:
            deltas = [left - right for left, right in
                zip(values(candidate, metric), values(reference, metric))]
            comparison[metric] = dict(per_seed_delta=deltas,
                mean_delta=statistics.mean(deltas), std_delta=statistics.stdev(deltas),
                positive_seeds=sum(value > 0 for value in deltas),
                negative_seeds=sum(value < 0 for value in deltas))
        comparisons[f'{candidate}_vs_{reference}'] = comparison

    c1_a4 = comparisons['C1_vs_A4']
    c1_a0 = comparisons['C1_vs_A0']
    checks = dict(mean_F1_gt_A4=c1_a4['F1']['mean_delta'] > 0,
        F1_positive_vs_A4_at_least_two_seeds=c1_a4['F1']['positive_seeds'] >= 2,
        mean_R_gt_A4=c1_a4['R']['mean_delta'] > 0,
        P_drop_vs_A4_at_most_0_25pp=c1_a4['P']['mean_delta'] >= -0.0025,
        AP_S_mean_not_below_A0=c1_a0['AP_S']['mean_delta'] >= 0,
        mAP50_95_drop_vs_A0_at_most_0_20pp=c1_a0['mAP50_95']['mean_delta'] >= -0.002)
    decision = dict(full_dataset_only=True, unconditional=True,
                    acceptance={'C1': dict(checks=checks, accepted=all(checks.values()))},
                    c0_equivalence_only=True)
    write_csv(run / 'test/seed0_comparison.csv', [row for row in rows if row['seed'] == 0])
    write_json(run / 'test/decision.json', decision)
    write_json(run / 'test/AllSeed_summary.json', dict(data=str(DATA), seeds=rows,
        statistics=statistics_rows, comparisons=comparisons, **decision,
        reference_stages=list(REFERENCE_STAGES), trained_stages=list(MODELS), std_ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = require_run_root(args.run_root)
    if not run.is_dir():
        raise FileNotFoundError(f'Preflight run directory missing: {run}')
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
    state.update(run_id=run.name.removeprefix('f1r_urpc2020_'), run_root=str(run),
        data=str(DATA), stages=list(MODELS), seeds=list(SEEDS),
        created_at=old.get('created_at', now()), launcher_pid=os.getpid(), worker_pids={},
        upstream_state=str(UPSTREAM_STATE), upstream_status=old.get('upstream_status'),
        upstream_failure_reason=old.get('upstream_failure_reason'), failure_reason=None,
        full_dataset_only=True, unconditional=True)
    (run / 'launcher.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
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
        if load(run / 'train/preflight.json')['status'] != 'passed':
            raise RuntimeError('Preflight did not pass')
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
                print('UPSTREAM_TERMINAL', state['upstream_status'],
                      state['upstream_failure_reason'], flush=True)
                break
            if not args.wait:
                raise RuntimeError('Upstream state is not terminal; use wait script')
            time.sleep(30)
        update('waiting', phase='upstream_terminal')
        if args.wait:
            print('CHAIN_START', now(), str(run), flush=True)
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            os.execv('/bin/bash', ['bash', str(ROOT / 'scripts/run_f1r_urpc2020.sh'),
                str(run), str(run / 'train/launcher.log'), '--resume'])

        update('running', started_at=state.get('started_at', now()))
        all_rows, mean_rows = [], []
        for reference in REFERENCE_STAGES:
            snapshot = load(run / 'test/references' / f'{reference}_AllSeed_summary.json')
            reference_rows = [enrich_reference(row) for row in snapshot['seeds']]
            all_rows.extend(reference_rows)
            mean_rows.extend(dict(stage=reference, metric=metric, **values)
                             for metric, values in statistics_for(reference_rows).items())
        for stage in MODELS:
            for mode in ('train', 'test'):
                children = []
                update('running', stage=stage, phase=mode, worker_pids={})
                for seed in SEEDS:
                    folder = run / mode / f'seed{seed}' / stage
                    if args.resume and completed_worker(run, stage, seed, mode):
                        continue
                    if folder.exists():
                        raise RuntimeError(f'Incomplete output needs explicit review: {folder}')
                    log = run / mode / f'{stage}_seed{seed}.worker.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    stream = log.open('a', encoding='utf-8')
                    command = [sys.executable, '-u', str(ROOT / 'tools/train_f1r_urpc2020_worker.py'),
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
                            if code:
                                raise RuntimeError(f'{stage}/{mode} worker {pid} failed: exit {code}')
                            del pending[pid]
                    if pending:
                        time.sleep(2)
                for _, stream in children:
                    stream.close()
                children = []
                update('running', worker_pids={})
            rows, summary = summarize(run, stage)
            all_rows.extend(rows)
            mean_rows.extend(dict(stage=stage, metric=metric, **values)
                             for metric, values in summary.items())
            write_csv(run / 'test/summary.csv', all_rows)
            write_csv(run / 'test/summary_mean_std.csv', mean_rows)
            write_json(run / 'test/AllSeed_summary.json', dict(data=str(DATA),
                seeds=all_rows, statistics=mean_rows, reference_stages=list(REFERENCE_STAGES),
                trained_stages=list(MODELS)[:list(MODELS).index(stage) + 1],
                full_dataset_only=True, unconditional=True, std_ddof=1))
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
