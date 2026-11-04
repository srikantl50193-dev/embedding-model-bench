from pathlib import Path

import pytest
from typer.testing import CliRunner

from embedding_model_bench import cli, preflight
from embedding_model_bench.config import Settings

runner = CliRunner()


def use_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)


def test_data_fails_clearly_when_fine_print_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_settings(
        monkeypatch, Settings(fine_print_dir=tmp_path / "nowhere", data_dir=tmp_path / "data")
    )
    result = runner.invoke(cli.app, ["data"])
    assert result.exit_code == 1
    assert "was not found" in result.output


def test_data_verify_fails_without_a_manifest(
    monkeypatch: pytest.MonkeyPatch, fine_print_tree: Path, tmp_path: Path
) -> None:
    use_settings(monkeypatch, Settings(fine_print_dir=fine_print_tree, data_dir=tmp_path / "data"))
    result = runner.invoke(cli.app, ["data-verify"])
    assert result.exit_code == 1
    assert "emb-bench data" in result.output


def test_preflight_exit_code_follows_the_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch, Settings())
    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda _step, _settings: [preflight.Check(name="Fine Print API", ok=False, detail="busy")],
    )
    result = runner.invoke(cli.app, ["preflight", "benchmark"])
    assert result.exit_code == 1
    assert "[STOP] Fine Print API: busy" in result.output

    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda _step, _settings: [preflight.Check(name="Memory", ok=True, detail="fine")],
    )
    assert runner.invoke(cli.app, ["preflight", "benchmark"]).exit_code == 0
