"""Validate early stopping and fail-fast process cleanup without GPU training."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_d1_dualseed_urpc2020 as chain

REAL_POPEN = subprocess.Popen


def test_historical_evaluator_import_does_not_execute_validation(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['worker', '--seed', '0', '--mode', 'test'])
    evaluator = chain.import_evaluator(chain.EVALUATOR)
    assert callable(evaluator.main)


@pytest.mark.parametrize('epochs,best,reason', [(300, 280, 'max_epochs'), (120, 90, 'early_stopping')])
def test_valid_completion(epochs, best, reason):
    assert chain.completion_reason(epochs, best) == reason


@pytest.mark.parametrize('epochs,best', [(120, 100), (301, 290), (0, 0)])
def test_rejects_premature_or_invalid_completion(epochs, best):
    with pytest.raises(RuntimeError):
        chain.completion_reason(epochs, best)


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


def test_failed_worker_stops_sibling(tmp_path, monkeypatch):
    (tmp_path / 'train').mkdir()
    processes = launch_stub(monkeypatch, fail_seed=1)
    with pytest.raises(RuntimeError, match='exited 7'):
        chain.run_batch(tmp_path, 'train', lambda *args, **kwargs: None)
    assert len(processes) == 2
    assert all(p.poll() is not None for p in processes)


def test_cancellation_stops_started_workers(tmp_path, monkeypatch):
    (tmp_path / 'train').mkdir()
    processes = launch_stub(monkeypatch)

    def cancel(status, **extra):
        if len(extra.get('worker_pids', {})) == 2:
            raise KeyboardInterrupt('user cancelled')

    with pytest.raises(KeyboardInterrupt):
        chain.run_batch(tmp_path, 'train', cancel)
    assert len(processes) == 2
    assert all(p.poll() is not None for p in processes)
