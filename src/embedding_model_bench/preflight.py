"""Checks that must pass before a step that uses a lot of memory.

These report what they find and never stop anything, since the servers belong to the user.
"""

import socket
from typing import Literal

import httpx
import psutil

from embedding_model_bench.config import Settings
from embedding_model_bench.models import FrozenModel

# A plain alias rather than a `type` statement, which Typer can't read as a CLI argument.
HeavyStep = Literal["benchmark", "generation"]

GIB = 1024**3


class Check(FrozenModel):
    """The result of one check."""

    name: str
    ok: bool
    detail: str


def port_is_listening(port: int) -> bool:
    """Say whether something on this machine accepts connections on a port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _ollama_names(settings: Settings, endpoint: str) -> list[str] | None:
    """Return the model names from an Ollama endpoint, or None if Ollama isn't reachable."""
    try:
        response = httpx.get(f"{settings.ollama_url}{endpoint}", timeout=settings.ollama_timeout)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return [model["name"] for model in response.json().get("models", [])]


def _server_check(name: str, port: int) -> Check:
    if port_is_listening(port):
        return Check(name=name, ok=False, detail=f"running on port {port}; please stop it")
    return Check(name=name, ok=True, detail=f"nothing on port {port}")


def run_preflight(step: HeavyStep, settings: Settings) -> list[Check]:
    """Check that a heavy step can run without competing for memory.

    Args:
        step: `benchmark` also covers training, and needs Ollama to hold no model in memory.
            `generation` needs the Ollama model to be available.
        settings: The ports and Ollama address.

    Returns:
        One result per check.
    """
    checks = [
        _server_check("Fine Print API", settings.fine_print_api_port),
        _server_check("Fine Print UI", settings.fine_print_ui_port),
    ]
    if step == "benchmark":
        loaded = _ollama_names(settings, "/api/ps") or []
        detail = f"loaded: {', '.join(loaded)}" if loaded else "no model in memory"
        checks.append(Check(name="Ollama idle", ok=not loaded, detail=detail))
    else:
        pulled = _ollama_names(settings, "/api/tags")
        if pulled is None:
            checks.append(Check(name="Ollama model", ok=False, detail="Ollama isn't running"))
        else:
            ok = settings.ollama_model in pulled
            detail = f"{settings.ollama_model} {'available' if ok else 'not pulled'}"
            checks.append(Check(name="Ollama model", ok=ok, detail=detail))
    available = psutil.virtual_memory().available / GIB
    checks.append(Check(name="Memory", ok=True, detail=f"{available:.1f} GB available"))
    return checks
