"""Exercise real process cleanup without starting training or using a GPU."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_baseline_d1_urpc2020 as chain

REAL_POPEN = subprocess.Popen


def launch_stub(monkeypatch, fail_seed=None):
    processes = []

    def launch(command, **kwargs):
        seed = int(command[command.index('--seed') + 1])
        code = ('import time; time.sleep(0.1); raise SystemExit(7)' if seed == fail_seed
                else 'import time; time.sleep(30)')
        process = REAL_POPEN([sys.executable, '-c', code], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(chain.subprocess, 'Popen', launch)
    return processes


def test_failed_worker_stops_all_siblings(tmp_path, monkeypatch):
    (tmp_path / 'train').mkdir()
    processes = launch_stub(monkeypatch, fail_seed=1)
    with pytest.raises(RuntimeError, match='exited 7'):
        chain.run_batch(tmp_path, 'A0', 'train', lambda *args, **kwargs: None)
    assert len(processes) == 3
    assert all(p.poll() is not None for p in processes)


def test_cancellation_during_launch_stops_started_workers(tmp_path, monkeypatch):
    (tmp_path / 'train').mkdir()
    processes = launch_stub(monkeypatch)

    def cancel(status, **extra):
        if len(extra.get('worker_pids', {})) == 2:
            raise KeyboardInterrupt('user cancelled')

    with pytest.raises(KeyboardInterrupt):
        chain.run_batch(tmp_path, 'A0', 'train', cancel)
    assert len(processes) == 2
    assert all(p.poll() is not None for p in processes)
