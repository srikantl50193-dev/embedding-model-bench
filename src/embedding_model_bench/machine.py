"""What the machine is doing: memory, competing apps, and who holds the heavy-step lock."""

import fcntl
import os
import resource
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import psutil

from embedding_model_bench.models import MachineState

GIB = 1024**3
MIB = 1024**2
LOW_MEMORY_GB = 4.0
COMPETING_APPS = {"Google Chrome": "Chrome", "Microsoft Teams": "Teams"}


class HeavyStepBusyError(RuntimeError):
    """Another benchmark, training or generation run is in progress."""


def machine_state() -> MachineState:
    """Report memory and competing apps. Timings taken under these conditions are provisional."""
    names = set()
    for process in psutil.process_iter(["name"]):
        name = process.info["name"] or ""
        names.update(label for app, label in COMPETING_APPS.items() if name.startswith(app))
    available = psutil.virtual_memory().available / GIB
    return MachineState(
        available_gb=round(available, 2),
        swap_used_gb=round(psutil.swap_memory().used / GIB, 2),
        competing_apps=sorted(names),
        timings_provisional=bool(names) or available < LOW_MEMORY_GB,
    )


def peak_rss_mb() -> float:
    """Return this process's peak resident memory. macOS reports bytes, and Linux kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / MIB if sys.platform == "darwin" else peak / 1024


def mps_allocated_mb() -> float:
    """Return what the MPS driver holds. Models on the GPU don't all show up in RSS."""
    import torch  # noqa: PLC0415

    return torch.mps.driver_allocated_memory() / MIB if torch.backends.mps.is_available() else 0.0


def _try_lock(fd: int) -> bool:
    """Try to take an exclusive lock without waiting, and say whether it was taken."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def lock_holder(lock_path: Path) -> int | None:
    """Return the process ID holding the lock, or None if nobody holds it.

    The lock is the operating system's, so a process that died, even without cleaning up,
    doesn't hold it. A recorded ID that is unreadable is reported as 0.
    """
    if not lock_path.exists():
        return None
    fd = os.open(lock_path, os.O_RDONLY)
    try:
        if _try_lock(fd):
            return None
        text = lock_path.read_text(encoding="utf-8").strip()
        return int(text) if text.isdigit() else 0
    finally:
        os.close(fd)


@contextmanager
def heavy_step(lock_path: Path) -> Generator[None]:
    """Hold the lock that keeps Ollama generation apart from benchmarks and training.

    The file stays in place; holding it is what counts, and the kernel releases it when this
    process ends however it ends.

    Raises:
        HeavyStepBusyError: If another process holds the lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _try_lock(fd):
            holder = lock_holder(lock_path)
            raise HeavyStepBusyError(f"Process {holder} is running a heavy step. Wait for it.")
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)  # Closing releases the lock.
