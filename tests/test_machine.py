import os
from pathlib import Path

import pytest

from embedding_model_bench import machine
from embedding_model_bench.machine import HeavyStepBusyError, heavy_step, lock_holder


def test_the_lock_is_released_after_use_and_after_an_error(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    with heavy_step(lock):
        assert lock_holder(lock) == os.getpid()
    assert not lock.exists()

    with pytest.raises(ValueError, match="boom"), heavy_step(lock):
        raise ValueError("boom")
    assert not lock.exists()


def test_a_live_holder_blocks_a_second_step(tmp_path: Path) -> None:
    lock = tmp_path / "heavy.lock"
    with heavy_step(lock), pytest.raises(HeavyStepBusyError), heavy_step(lock):
        pass


def test_a_dead_holder_does_not_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "heavy.lock"
    lock.write_text("99999", encoding="utf-8")
    monkeypatch.setattr(machine.psutil, "pid_exists", lambda _pid: False)
    assert lock_holder(lock) is None
    with heavy_step(lock):
        pass


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
