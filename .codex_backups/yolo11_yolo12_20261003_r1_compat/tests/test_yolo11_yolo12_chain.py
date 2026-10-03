"""Check real sibling-process cleanup, terminal rules, and sequential model execution."""
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_yolo11_yolo12_urpc2020 as chain

REAL_POPEN = subprocess.Popen


def test_early_stopping_disabled():
    trainer = SimpleNamespace(stopper=SimpleNamespace(patience=40, possible_stop=True))
    chain.disable_early_stopping(trainer)
    assert math.isinf(trainer.stopper.patience) and not trainer.stopper.possible_stop
    assert 'patience' not in chain.SETTINGS


@pytest.mark.parametrize('count', [179, 180, 181])
def test_completed_epochs(count):
    rows = [dict(epoch=epoch, loss=1.) for epoch in range(1, count + 1)]
    if count == 180:
        chain.validate_completion(rows)
    else:
        with pytest.raises(RuntimeError):
            chain.validate_completion(rows)


@pytest.mark.parametrize('status', ['completed', 'failed', 'cancelled', 'running', 'invalid', None])
def test_upstream_terminal_status(tmp_path, monkeypatch, status):
    path = tmp_path / 'state.json'
    path.write_text(json.dumps(dict(status=status, failure_reason='example')), encoding='utf-8')
    monkeypatch.setattr(chain, 'UPSTREAM', path)
    value = chain.upstream_terminal()
    assert bool(value) == (status in chain.TERMINAL)
    if value:
        assert value['failure_reason'] == 'example'


@pytest.mark.parametrize('content', [None, 'broken json', '[]'])
def test_missing_invalid_state_keeps_waiting(tmp_path, monkeypatch, content):
    path = tmp_path / 'state.json'
    if content is not None:
        path.write_text(content, encoding='utf-8')
    monkeypatch.setattr(chain, 'UPSTREAM', path)
    assert chain.upstream_terminal() is None


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


def test_failure_stops_all_three_workers(tmp_path, monkeypatch):
    (tmp_path / 'train/YOLO11').mkdir(parents=True)
    processes = launch_stub(monkeypatch, fail_seed=1)
    with pytest.raises(RuntimeError, match='exited 7'):
        chain.run_batch(tmp_path, 'YOLO11', 'train', lambda *args, **kwargs: None)
    assert len(processes) == 3 and all(p.poll() is not None for p in processes)


def test_cancellation_stops_started_workers(tmp_path, monkeypatch):
    (tmp_path / 'train/YOLO11').mkdir(parents=True)
    processes = launch_stub(monkeypatch)

    def cancel(status, **extra):
        if len(extra.get('worker_pids', {})) == 3:
            raise KeyboardInterrupt('cancelled')

    with pytest.raises(KeyboardInterrupt):
        chain.run_batch(tmp_path, 'YOLO11', 'train', cancel)
    assert len(processes) == 3 and all(p.poll() is not None for p in processes)


def test_yolo12_never_starts_after_yolo11_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(chain, 'verify_contract', lambda run: None)

    def fail(run, stage, mode, update):
        calls.append((stage, mode))
        raise RuntimeError('train failed')

    monkeypatch.setattr(chain, 'run_batch', fail)
    with pytest.raises(RuntimeError):
        chain.execute_stages(tmp_path, lambda *args, **kwargs: None)
    assert calls == [('YOLO11', 'train')]


def test_duplicate_output_rejected_before_launch(tmp_path, monkeypatch):
    (tmp_path / 'train/YOLO11/seed0').mkdir(parents=True)
    processes = launch_stub(monkeypatch)
    with pytest.raises(FileExistsError):
        chain.run_batch(tmp_path, 'YOLO11', 'train', lambda *args, **kwargs: None)
    assert not processes
