"""Exercise three-seed C0/C1/C2 scheduling and sibling cleanup without GPU training."""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('fcntl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_ucra_v4 as chain


def prepare(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    upstream = tmp_path / 'upstream.json'
    chain.write_json(upstream, {'status': 'completed', 'failure_reason': None})
    chain.write_json(run / 'train/preflight.json', {'status': 'passed'})
    monkeypatch.setattr(chain, 'UPSTREAM_STATE', upstream)
    monkeypatch.setattr(chain, 'require_run_root', lambda _: run)
    monkeypatch.setattr(chain, 'verify_contract', lambda _: {})
    monkeypatch.setattr(sys, 'argv', ['run_ucra_v4', '--run-root', str(run)])
    baseline = dict(P=.8, R=.75, mAP50=.8, mAP50_95=.46, AP_S=.14, AP_M=.38, AP_L=.49)
    for stage in ('A0', 'A4', 'B2'):
        chain.write_json(run / 'test/references' / f'{stage}_AllSeed_summary.json',
            dict(seeds=[dict(baseline, stage=stage, seed=seed) for seed in range(3)],
                 statistics={'P': dict(mean=.8, std=0, best=.8, worst=.8)}))
    return run, baseline


def test_failure_stops_all_three_siblings(tmp_path, monkeypatch):
    run, _ = prepare(tmp_path, monkeypatch)
    real_popen, children = subprocess.Popen, []

    def launch(command, **kwargs):
        seed = int(command[command.index('--seed') + 1])
        script = 'raise SystemExit(7)' if seed == 0 else 'import time;time.sleep(120)'
        process = real_popen([sys.executable, '-c', script], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(chain.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match='C0/train.*exit 7'):
        chain.main()
    assert len(children) == 3 and all(p.poll() is not None for p in children)
    state = chain.load(run / 'state.json')
    assert state['status'] == 'failed' and state['worker_pids'] == {}


def test_all_stages_launch_three_seeds_before_waiting(tmp_path, monkeypatch):
    run, baseline = prepare(tmp_path, monkeypatch)
    launched = []

    class SuccessfulWorker:
        def __init__(self, command, **kwargs):
            stage = command[command.index('--stage') + 1]
            mode = command[command.index('--mode') + 1]
            seed = int(command[command.index('--seed') + 1])
            launched.append((stage, mode, seed))
            self.pid = 900000 + len(launched)

        def poll(self):
            assert len(launched) % 3 == 0
            return 0

    def summarize(run, stage):
        assert launched[-3:] == [(stage, 'test', seed) for seed in range(3)]
        values = dict(baseline, mAP50_95=baseline['mAP50_95'] + .005,
                      AP_S=baseline['AP_S'] + .005, P=baseline['P'] + .005)
        return ([dict(values, stage=stage, seed=seed) for seed in range(3)],
                {'P': dict(mean=values['P'], std=0, best=values['P'], worst=values['P'])})

    monkeypatch.setattr(chain.subprocess, 'Popen', SuccessfulWorker)
    monkeypatch.setattr(chain, 'summarize', summarize)
    chain.main()
    assert launched == [(stage, mode, seed) for stage in ('C0', 'C1', 'C2')
                        for mode in ('train', 'test') for seed in range(3)]
    summary = chain.load(run / 'test/AllSeed_summary.json')
    assert len(summary['seeds']) == 18
    assert summary['trained_stages'] == ['C0', 'C1', 'C2']
    assert summary['c2_unconditional'] is True
    assert chain.load(run / 'state.json')['status'] == 'completed'


def test_half_dataset_completion_marker_required(tmp_path):
    folder = tmp_path / 'train/seed0/C0'
    chain.write_json(folder / 'train_complete.json', dict(stage='C0', seed=0,
        data='/home/room305/ZZF/URPC2020/data.yaml', epochs=180,
        settings=chain.SETTINGS, early_stopping=False))
    with pytest.raises(RuntimeError, match='Invalid completed training'):
        chain.completed_worker(tmp_path, 'C0', 0, 'train')
