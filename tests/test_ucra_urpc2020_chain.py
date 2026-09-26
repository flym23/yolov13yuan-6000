"""Verify the process gate and batch scheduling without running GPU training."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip('fcntl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_ucra_urpc2020 as chain
from tools import ucra_process_gate as gate


def prepare(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    chain.write_json(run / 'train/preflight.json', {'status': 'passed'})
    identity = dict(pid=3645793, start_ticks=1, boot_id='test-only')
    monkeypatch.setattr(chain, 'require_run_root', lambda _: run)
    monkeypatch.setattr(chain, 'verify_contract', lambda _: {'upstream_process': identity})
    monkeypatch.setattr(chain, 'is_original_running', lambda _: False)
    monkeypatch.setattr(sys, 'argv', ['chain', '--run-root', str(run)])
    return run


def test_gate_waits_for_real_exit():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.8)'])
    try:
        identity = gate.process_identity(process.pid)
        started = time.monotonic()
        assert gate.is_original_running(identity)
        assert gate.wait_for_exit(identity) == 'original_process_exit_observed'
        assert time.monotonic() - started >= 0.3
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_gate_does_not_wait_on_reused_pid():
    identity = gate.process_identity(os.getpid())
    identity['start_ticks'] -= 1
    assert not gate.is_original_running(identity)
    assert gate.wait_for_exit(identity) == 'original_process_already_exited'


def test_gate_rechecks_identity_after_pidfd_open(monkeypatch):
    monkeypatch.setattr(gate, 'is_original_running', lambda _: next(states))
    states = iter((True, False))
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(gate, 'open_pidfd', lambda _: read_fd)
    try:
        assert gate.wait_for_exit({'pid': 42}) == 'original_process_exited_during_open'
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        os.close(write_fd)


def test_live_upstream_blocks_direct_run(tmp_path, monkeypatch):
    run = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(chain, 'is_original_running', lambda _: True)
    with pytest.raises(RuntimeError, match='still running'):
        chain.main()
    state = chain.load(run / 'state.json')
    assert state['status'] == 'failed' and state['worker_pids'] == {}


def test_failure_stops_real_sibling_processes(tmp_path, monkeypatch):
    run = prepare(tmp_path, monkeypatch)
    real_popen, children = subprocess.Popen, []

    def launch(command, **kwargs):
        seed = int(command[command.index('--seed') + 1])
        script = 'raise SystemExit(7)' if seed == 0 else 'import time;time.sleep(120)'
        process = real_popen([sys.executable, '-c', script], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(chain.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match='exit 7'):
        chain.main()
    assert len(children) == 3 and all(p.poll() is not None for p in children)
    state = chain.load(run / 'state.json')
    assert state['status'] == 'failed' and state['worker_pids'] == {}


def test_all_five_stages_train_three_fresh_seeds_in_order(tmp_path, monkeypatch):
    run = prepare(tmp_path, monkeypatch)
    launched, polled = [], []

    class SuccessfulWorker:
        def __init__(self, command, **kwargs):
            stage = command[command.index('--stage') + 1]
            mode = command[command.index('--mode') + 1]
            seed = int(command[command.index('--seed') + 1])
            launched.append((stage, mode, seed))
            self.pid = 900000 + len(launched)

        def poll(self):
            # The manager must launch all three before waiting for any of them.
            assert len(launched) % 3 == 0
            polled.append(self.pid)
            return 0

    def summarize(run, stage):
        assert launched[-3:] == [(stage, 'test', s) for s in range(3)]
        return [dict(stage=stage, seed=s, P=.8) for s in range(3)], {'P': dict(mean=.8, std=0, best=.8, worst=.8)}

    monkeypatch.setattr(chain.subprocess, 'Popen', SuccessfulWorker)
    monkeypatch.setattr(chain, 'summarize', summarize)
    chain.main()
    assert launched == [(stage, mode, seed) for stage in ('A0', 'A4', 'B1', 'B2', 'B3')
                        for mode in ('train', 'test') for seed in range(3)]
    assert len(polled) == 30
    result = chain.load(run / 'test/AllSeed_summary.json')
    assert len(result['seeds']) == 15 and result['reference_stages'] == []
    assert chain.load(run / 'state.json')['status'] == 'completed'


def test_half_dataset_completion_cannot_be_resumed(tmp_path):
    folder = tmp_path / 'train/seed0/A0'
    chain.write_json(folder / 'train_complete.json', dict(epochs=180, settings=chain.SETTINGS,
        early_stopping=False, data='/home/room305/ZZF/URPC2020half/data.yaml', stage='A0', seed=0))
    with pytest.raises(RuntimeError, match='Invalid completed training'):
        chain.completed_worker(tmp_path, 'A0', 0, 'train')
