"""Read-only Linux process exit gate, bound to boot ID and process start ticks."""
from __future__ import annotations

import errno
import ctypes
import os
import select
import time
from pathlib import Path


def process_identity(pid):
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text(encoding='utf-8')
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = text[text.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf-8').strip())


def is_original_running(expected):
    current = process_identity(expected['pid'])
    return (current is not None and current['state'] not in ('Z', 'X')
            and all(current[k] == expected[k] for k in ('pid', 'start_ticks', 'boot_id')))


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Some Conda Python builds omit os.pidfd_open although the host supports it.
    library = ctypes.CDLL(None, use_errno=True)
    function = getattr(library, 'pidfd_open', None)
    if function is None:
        raise OSError(errno.ENOSYS, 'libc.pidfd_open unavailable')
    function.argtypes = (ctypes.c_int, ctypes.c_uint)
    function.restype = ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    try:
        os.set_inheritable(descriptor, False)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def wait_for_exit(expected, announce=print):
    # pidfd becomes readable on exit, including exit before the parent reaps it.
    # Rechecking identity after opening prevents accidentally waiting on a reused PID.
    if not is_original_running(expected):
        return 'original_process_already_exited'
    descriptor = None
    try:
        try:
            descriptor = open_pidfd(expected['pid'])
        except ProcessLookupError:
            return 'original_process_exited_during_open'
        except OSError as error:
            if error.errno not in (errno.ENOSYS, errno.EINVAL, errno.EPERM, errno.EACCES):
                raise
            announce(f'PIDFD_UNAVAILABLE {error}; identity polling every 1 second')
        if not is_original_running(expected):
            return 'original_process_exited_during_open'
        if descriptor is not None:
            announce('WAIT_METHOD pidfd: exit event triggers immediately')
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            while True:
                events = poller.poll(30000)
                if events or not is_original_running(expected):
                    return 'original_process_exit_observed'
        announce('WAIT_METHOD identity polling every 1 second')
        while is_original_running(expected):
            time.sleep(1)
        return 'original_process_exit_observed'
    finally:
        if descriptor is not None:
            os.close(descriptor)
