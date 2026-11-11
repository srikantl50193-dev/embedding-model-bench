import os
import subprocess
import sys
from pathlib import Path

import pytest

from embedding_model_bench import machine
from embedding_model_bench.machine import HeavyStepBusyError, heavy_step, lock_holder


def test_the_lock_is_released_after_use_and_after_an_error(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    with heavy_step(lock):
        assert lock_holder(lock) == os.getpid()
    assert lock_holder(lock) is None

    with pytest.raises(ValueError, match="boom"), heavy_step(lock):
        raise ValueError("boom")
    assert lock_holder(lock) is None


def test_a_held_lock_blocks_a_second_step(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    busy = pytest.raises(HeavyStepBusyError, match=str(os.getpid()))
    with heavy_step(lock), busy, heavy_step(lock):
        pass


def test_a_file_left_by_a_dead_process_does_not_block(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    lock.write_text("99999", encoding="utf-8")  # Recorded, but nobody holds the lock.
    assert lock_holder(lock) is None
    with heavy_step(lock):
        assert lock_holder(lock) == os.getpid()


def test_a_lock_held_by_another_process_blocks(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    script = (
        "import fcntl, os, sys, time; "
        f"fd = os.open({str(lock)!r}, os.O_RDWR | os.O_CREAT); "
        "fcntl.flock(fd, fcntl.LOCK_EX); os.write(fd, b'4242'); "
        "print('held', flush=True); time.sleep(30)"
    )
    holder = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        assert lock_holder(lock) == 4242
        with pytest.raises(HeavyStepBusyError, match="4242"), heavy_step(lock):
            pass
    finally:
        holder.kill()
        holder.wait()
    assert lock_holder(lock) is None


class FakeProcess:
    def __init__(self, name: str) -> None:
        self.info = {"name": name}


def test_competing_apps_make_timings_provisional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        machine.psutil,
        "process_iter",
        lambda _fields: [FakeProcess("Google Chrome Helper"), FakeProcess("python")],
    )
    state = machine.machine_state()
    assert state.competing_apps == ["Chrome"]
    assert state.timings_provisional
