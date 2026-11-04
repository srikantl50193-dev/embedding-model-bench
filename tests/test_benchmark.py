from pathlib import Path

import pytest
from fine_print.models import Chunk

from conftest_fakes import FakeEncoder
from embedding_model_bench.benchmark import (
    BenchmarkInputs,
    load_tuning_questions,
    noise_note,
    run_benchmark,
    summary_table,
)
from embedding_model_bench.models import BenchmarkResult, EvalQuestion, ModelSpec, SciFact

SPEC = ModelSpec(key="fake", repo="org/fake", revision="abc", dim=64, query_prefix="q: ")


def make_chunk(chunk_id: str, text: str, doc_type: str = "terms", section: str = "Misc") -> Chunk:
    return Chunk(
        text=text,
        company="acme",
        doc_type=doc_type,
        section=section,
        chunk_id=chunk_id,
        strategy="section",
        start_char=0,
        end_char=len(text),
    )


CHUNKS = [
    make_chunk("a0", "we do not provide refunds", section="Refunds"),
    make_chunk("a1", "we keep data as long as necessary", "privacy", "Retention"),
    *[make_chunk(f"f{i}", f"unrelated filler {i}") for i in range(12)],
]
QUESTIONS = [
    EvalQuestion(
        question="provide refunds", company="acme", doc_type="terms", keywords=["refunds"]
    ),
    EvalQuestion(question="keep data", company="acme", doc_type="privacy", keywords=["as long as"]),
]
SCIFACT = SciFact(
    corpus={str(i): f"science document {i} topic{i}" for i in range(12)},
    queries={"q1": "topic3"},
    qrels={"q1": {"3": 1}},
)


INPUTS = BenchmarkInputs(CHUNKS, QUESTIONS, SCIFACT, "commit", "fingerprint")


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
