"""Compare models question by question, because small gaps on small sets are mostly noise."""

from collections.abc import Sequence
from math import comb

from embedding_model_bench.models import BenchmarkResult, FrozenModel, RetrievalScores
from embedding_model_bench.report import markdown_table

SELECTION_K = 5
NOISE_QUESTIONS = 2

# SciFact nDCG@10 on the public MTEB results repo (embeddings-benchmark/results). A sanity check
# on this pipeline, not a target: MTEB's revision and prefixes can differ from those used here.
MTEB_SCIFACT_NDCG = {
    "bge-small-en-v1.5": 0.713,
    "granite-small-en-r2": 0.755,
    "bge-base-en-v1.5": 0.743,
    "arctic-embed-m-v1.5": 0.718,
    "e5-base-v2": 0.719,
    "nomic-embed-text-v1.5": 0.703,
}


def noise_note(num_queries: int) -> str:
    """Say what one question is worth, so that small gaps aren't over-read."""
    points = 100 / num_queries
    return (
        f"n={num_queries}: one question moves a score by {points:.1f} points, "
        f"so gaps up to {2 * points:.1f} points are likely noise."
    )


class PairedComparison(FrozenModel):
    """How one model did against a reference, on the same questions."""

    num_queries: int
    k: int
    wins: int
    losses: int
    sign_test_p: float
    within_noise: bool


def sign_test_p(wins: int, losses: int) -> float:
    """Return the exact two-sided sign test p-value, ignoring questions that tied."""
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(comb(total, i) for i in range(min(wins, losses) + 1)) / 2**total
    return min(1.0, 2 * tail)


def paired_comparison(
    reference: RetrievalScores,
    other: RetrievalScores,
    k: int = SELECTION_K,
    rows: Sequence[int] | None = None,
) -> PairedComparison:
    """Count the questions only one of two models found in its top k.

    Args:
        reference: The model compared against.
        other: The model being compared.
        k: The cut-off.
        rows: Only these questions, by index. Defaults to all of them.

    Raises:
        ValueError: If the two results don't cover the same number of questions.
    """
    if reference.num_queries != other.num_queries:
        raise ValueError("Results must cover the same questions to be compared")
    reference_hits, other_hits = reference.hit_flags(k), other.hit_flags(k)
    chosen = range(reference.num_queries) if rows is None else rows
    wins = sum(other_hits[i] and not reference_hits[i] for i in chosen)
    losses = sum(reference_hits[i] and not other_hits[i] for i in chosen)
    return PairedComparison(
        num_queries=len(chosen),
        k=k,
        wins=wins,
        losses=losses,
        sign_test_p=round(sign_test_p(wins, losses), 4),
        within_noise=abs(wins - losses) <= NOISE_QUESTIONS,
    )


def hits_at(result: BenchmarkResult, k: int = SELECTION_K) -> int:
    """Return how many tuning questions a model found in its top k."""
    return result.tuning.hits(k)


def apply_selection_rule(results: Sequence[BenchmarkResult]) -> BenchmarkResult:
    """Apply the rule fixed before any results, in CLAUDE.md.

    Models within two questions of the best hit@5 count as tied. Among those, the one with the
    smallest weights wins, then the one with the fewest dimensions, then the fastest.
    """
    best = max(hits_at(result) for result in results)
    tied = [result for result in results if best - hits_at(result) <= NOISE_QUESTIONS]
    return min(
        tied,
        key=lambda r: (r.resources.weights_mb, r.spec.dim, r.timings[0].query_p95_ms),
    )


def comparison_table(results: Sequence[BenchmarkResult], reference_key: str) -> str:
    """Render each model against the reference, and against MTEB's published SciFact score."""
    reference = next(r for r in results if r.spec.key == reference_key)
    tuning_n, scifact_n = reference.tuning.num_queries, reference.scifact.num_queries
    headers = [
        "Model",
        "Tuning hit@5",
        f"Wins / losses (n={tuning_n})",
        "Sign test p",
        "Within noise?",
        f"SciFact wins / losses (n={scifact_n})",
        "SciFact nDCG@10",
        "MTEB nDCG@10",
    ]
    rows = []
    for result in results:
        tuning = paired_comparison(reference.tuning, result.tuning)
        scifact = paired_comparison(reference.scifact, result.scifact)
        mteb = MTEB_SCIFACT_NDCG.get(result.spec.key)
        rows.append(
            [
                result.spec.key,
                f"{hits_at(result)}/{tuning_n}",
                f"{tuning.wins} / {tuning.losses}",
                f"{tuning.sign_test_p:.3f}",
                "yes" if tuning.within_noise else "no",
                f"{scifact.wins} / {scifact.losses}",
                f"{result.scifact.ndcg_at_10:.3f}",
                f"{mteb:.3f}" if mteb is not None else "n/a",
            ]
        )
    intro = (
        f"Compared with {reference_key}, by question. Wins are questions found only by that "
        "model; losses are questions found only by the reference.\n\n"
    )
    return intro + markdown_table(headers, rows)
