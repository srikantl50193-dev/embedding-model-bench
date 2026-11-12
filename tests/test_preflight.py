import socket
from collections.abc import Iterator

import httpx
import pytest

from embedding_model_bench import preflight
from embedding_model_bench.config import Settings
from embedding_model_bench.preflight import port_is_listening, run_preflight


@pytest.fixture
def no_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "port_is_listening", lambda _port: False)


@pytest.fixture
def listening_port() -> Iterator[int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield sock.getsockname()[1]


def fake_ollama(monkeypatch: pytest.MonkeyPatch, models: list[str]) -> None:
    def get(url: str, **_options: object) -> httpx.Response:
        body = {"models": [{"name": name} for name in models]}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(preflight.httpx, "get", get)


def by_name(step: preflight.HeavyStep) -> dict[str, bool]:
    return {check.name: check.ok for check in run_preflight(step, Settings())}


def test_port_is_listening_detects_a_real_socket(listening_port: int) -> None:
    assert port_is_listening(listening_port)


def test_running_fine_print_servers_fail_the_check(
    monkeypatch: pytest.MonkeyPatch, listening_port: int
) -> None:
    fake_ollama(monkeypatch, [])
    settings = Settings(fine_print_api_port=listening_port)
    checks = {check.name: check.ok for check in run_preflight("benchmark", settings)}
    assert checks["Fine Print API"] is False


@pytest.mark.usefixtures("no_servers")
def test_benchmark_stops_when_ollama_holds_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ollama(monkeypatch, ["gemma4:e4b"])
    assert by_name("benchmark")["Ollama idle"] is False


@pytest.mark.usefixtures("no_servers")
def test_benchmark_passes_when_ollama_is_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ollama(monkeypatch, [])
    assert all(by_name("benchmark").values())


@pytest.mark.usefixtures("no_servers")
def test_benchmark_passes_when_ollama_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_options: object) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(preflight.httpx, "get", refuse)
    assert all(by_name("benchmark").values())


@pytest.mark.usefixtures("no_servers")
def test_generation_needs_the_model_and_a_running_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ollama(monkeypatch, ["llama3.2:latest"])
    assert by_name("generation")["Ollama model"] is False

    fake_ollama(monkeypatch, ["gemma4:e4b"])
    assert by_name("generation")["Ollama model"] is True

    def refuse(*_args: object, **_options: object) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(preflight.httpx, "get", refuse)
    assert by_name("generation")["Ollama model"] is False
