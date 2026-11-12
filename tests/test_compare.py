import pytest

from embedding_model_bench.compare import (
    apply_selection_rule,
    comparison_table,
    hits_at,
    paired_comparison,
    sign_test_p,
)
from embedding_model_bench.models import (
    BenchmarkResult,
    MachineState,
    ModelSpec,
    Resources,
    RetrievalScores,
    Timing,
)


def scores(ranks: list[int | None]) -> RetrievalScores:
    return RetrievalScores(
        num_queries=len(ranks),
        num_without_relevant=0,
        hit_at_k={},
        recall_at_k={},
        mrr=0,
        ndcg_at_10=0,
        first_relevant_ranks=ranks,
    )


def result(key: str, ranks: list[int | None], weights: float, dim: int = 384) -> BenchmarkResult:
    return BenchmarkResult(
        spec=ModelSpec(key=key, repo=f"org/{key}", revision="r", dim=dim),
        fine_print_commit="c",
        corpus_fingerprint="f",
        num_chunks=1,
        tuning=scores(ranks),
        scifact=scores(ranks),
        timings=[
            Timing(
                device="cpu",
                encode_chunks_per_s=1,
                query_p50_ms=1,
                query_p95_ms=1,
                num_query_samples=1,
            )
        ],
        resources=Resources(parameters=1, weights_mb=weights, peak_rss_mb=1, mps_allocated_mb=0),
        machine=MachineState(
            available_gb=8, swap_used_gb=0, competing_apps=[], timings_provisional=False
        ),
    )


def test_sign_test_matches_hand_calculations() -> None:
    assert sign_test_p(0, 0) == 1
    assert sign_test_p(1, 1) == 1
    assert sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert sign_test_p(4, 1) == pytest.approx(2 * 6 / 32)


def test_paired_comparison_counts_only_questions_that_differ() -> None:
    reference = scores([1, None, 6, 2])
    other = scores([1, 3, 2, None])
    comparison = paired_comparison(reference, other, k=5)
    assert (comparison.wins, comparison.losses) == (2, 1)
    assert comparison.within_noise


def test_a_gap_of_three_questions_is_not_called_noise() -> None:
    comparison = paired_comparison(scores([None] * 3), scores([1] * 3))
    assert comparison.wins == 3
    assert not comparison.within_noise


def test_comparing_different_question_sets_is_an_error() -> None:
    with pytest.raises(ValueError, match="same questions"):
        paired_comparison(scores([1]), scores([1, 2]))


def test_selection_prefers_the_smaller_model_within_two_questions() -> None:
    big = result("big", [1, 1, 1, 1, 1], weights=400)
    small = result("small", [1, 1, 1, None, None], weights=100)
    assert apply_selection_rule([big, small]).spec.key == "small"


def test_selection_keeps_the_better_model_beyond_two_questions() -> None:
    big = result("big", [1, 1, 1, 1], weights=400)
    small = result("small", [1, None, None, None], weights=100)
    assert apply_selection_rule([big, small]).spec.key == "big"


def test_selection_breaks_a_size_tie_on_dimensions() -> None:
    wide = result("wide", [1], weights=100, dim=768)
    narrow = result("narrow", [1], weights=100, dim=384)
    assert apply_selection_rule([wide, narrow]).spec.key == "narrow"


def test_hits_counts_only_the_top_five() -> None:
    assert hits_at(result("m", [1, 5, 6, None], weights=1)) == 2


def test_comparison_table_shows_counts_and_mteb() -> None:
    reference = result("bge-small-en-v1.5", [1, None], weights=1)
    other = result("other", [None, 1], weights=1)
    table = comparison_table([reference, other], "bge-small-en-v1.5")
    assert "| other | 1/2 | 1 / 1 |" in table
    assert "0.713" in table
