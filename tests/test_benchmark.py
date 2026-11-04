from pathlib import Path

import pytest

from conftest_fakes import CHUNKS, INPUTS, QUESTIONS, SPEC, FakeEncoder
from embedding_model_bench.benchmark import (
    load_tuning_questions,
    run_benchmark,
    summary_table,
)
from embedding_model_bench.compare import noise_note
from embedding_model_bench.models import BenchmarkResult, ModelSpec


def run(tmp_path: Path) -> BenchmarkResult:
    return run_benchmark(SPEC, INPUTS, tmp_path, lambda _spec, _device: FakeEncoder(), ["cpu"])


def test_end_to_end_with_a_fake_model(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result.tuning.num_queries == 2
    assert result.tuning.first_relevant_ranks == [1, 1]
    assert result.scifact.num_queries == 1
    assert result.scifact.recall_at_k[1] == 1
    assert result.num_chunks == len(CHUNKS)
    assert result.timings[0].device == "cpu"
    assert result.timings[0].num_query_samples == 10
    assert result.resources.parameters == 1_000_000
    assert result.resources.weights_mb == 4
    assert result.resources.mps_allocated_mb == 0


def test_quality_scores_are_the_same_when_read_back_from_the_cache(tmp_path: Path) -> None:
    first = run(tmp_path)
    second = run(tmp_path)
    assert first.tuning == second.tuning
    assert first.scifact == second.scifact


def test_every_device_gets_its_own_timing(tmp_path: Path) -> None:
    devices = []

    def factory(_spec: ModelSpec, device: str) -> FakeEncoder:
        devices.append(device)
        return FakeEncoder()

    result = run_benchmark(SPEC, INPUTS, tmp_path, factory, ["cpu", "other"])
    assert [t.device for t in result.timings] == ["cpu", "other"]
    assert devices == ["cpu", "other"]


def test_tuning_questions_load_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(QUESTIONS[0].model_dump_json() + "\n\n", encoding="utf-8")
    assert load_tuning_questions(path) == [QUESTIONS[0]]


def test_summary_table_reports_the_question_count_and_noise(tmp_path: Path) -> None:
    table = summary_table([run(tmp_path)])
    assert "| fake |" in table
    assert "n=2" in table


@pytest.mark.parametrize(("count", "expected"), [(31, "3.2 points"), (20, "5.0 points")])
def test_noise_note_states_what_one_question_is_worth(count: int, expected: str) -> None:
    assert expected in noise_note(count)
