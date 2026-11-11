from pathlib import Path

import pytest

from conftest_fakes import CHUNKS, QUESTIONS, SPEC, FakeEncoder
from embedding_model_bench import holdout
from embedding_model_bench.benchmark import score_vectors, tuning_relevant
from embedding_model_bench.embedding import Encoder
from embedding_model_bench.holdout import (
    BASELINE,
    CONFIGS,
    HoldoutConfig,
    HoldoutResult,
    embed_all,
    holdout_table,
    score_config,
    score_set,
)
from embedding_model_bench.matryoshka import truncate
from embedding_model_bench.quantization import Scheme
from embedding_model_bench.registry import BGE_BASE, BGE_SMALL


def never() -> Encoder:
    raise AssertionError("scoring must not load a model")


def fake_config(name: str, dim: int = 64, scheme: Scheme = Scheme.FLOAT32) -> HoldoutConfig:
    return HoldoutConfig(name=name, key=SPEC.key, dim=dim, scheme=scheme)


def test_the_configurations_match_the_frozen_note() -> None:
    names = [c.name for c in CONFIGS]
    assert names[-1] == BASELINE.name
    assert len(names) == len(set(names)) == 5
    assert {(c.key, c.dim, c.scheme, c.fine_tuned) for c in CONFIGS} == {
        (BGE_BASE.key, 768, Scheme.FLOAT32, False),
        (BGE_BASE.key, 768, Scheme.BINARY_RESCORE_FLOAT, False),
        (BGE_BASE.key, 256, Scheme.FLOAT32, False),
        (BGE_SMALL.key, 384, Scheme.FLOAT32, True),
        (BGE_SMALL.key, 384, Scheme.FLOAT32, False),
    }


@pytest.mark.parametrize(
    ("dim", "scheme"),
    [(64, Scheme.FLOAT32), (32, Scheme.FLOAT32), (64, Scheme.BINARY_RESCORE_FLOAT)],
)
def test_scoring_reads_only_the_cache_after_embedding(
    tmp_path: Path, dim: int, scheme: Scheme
) -> None:
    embed_all([SPEC], [[q.question for q in QUESTIONS]], CHUNKS, tmp_path, lambda _s: FakeEncoder())
    relevant = tuning_relevant(CHUNKS, QUESTIONS)
    scores = score_config(
        fake_config("x", dim, scheme), SPEC, QUESTIONS, CHUNKS, relevant, tmp_path, never
    )
    assert scores.num_queries == len(QUESTIONS)
    if scheme == Scheme.FLOAT32:
        encoder = FakeEncoder()
        queries = encoder.encode([SPEC.query_prefix + q.question for q in QUESTIONS])
        chunks = encoder.encode([c.embedding_text() for c in CHUNKS])
        if dim < SPEC.dim:
            queries, chunks = truncate(queries, dim), truncate(chunks, dim)
        assert scores == score_vectors(queries, chunks, relevant)


def test_every_configuration_is_compared_with_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = [fake_config("a"), fake_config("b", 32), fake_config("base")]
    monkeypatch.setattr(holdout, "CONFIGS", configs)
    monkeypatch.setattr(holdout, "BASELINE", configs[-1])
    embed_all([SPEC], [[q.question for q in QUESTIONS]], CHUNKS, tmp_path, lambda _s: FakeEncoder())
    held = score_set(
        "Fresh",
        "note",
        "sha",
        QUESTIONS,
        CHUNKS,
        {c.name: SPEC for c in configs},
        cache_dir=tmp_path,
        load=never,
    )
    assert [r.config.name for r in held.rows] == ["a", "b", "base"]
    assert all((r.vs_baseline.wins, r.vs_baseline.losses) == (0, 0) for r in held.rows)
    table = holdout_table(HoldoutResult(fine_print_commit="c", corpus_fingerprint="f", sets=[held]))
    assert "## Fresh" in table
    assert f"2/{len(QUESTIONS)}" in table
    assert f"n={len(QUESTIONS)}" in table


def test_two_question_sets_are_each_found_by_scoring(tmp_path: Path) -> None:
    first, second = QUESTIONS[:1], QUESTIONS[1:]
    sets = [[q.question for q in first], [q.question for q in second]]
    embed_all([SPEC], sets, CHUNKS, tmp_path, lambda _s: FakeEncoder())
    for questions in (first, second):
        relevant = tuning_relevant(CHUNKS, questions)
        score_config(fake_config("x"), SPEC, questions, CHUNKS, relevant, tmp_path, never)
