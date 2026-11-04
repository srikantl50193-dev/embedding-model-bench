import numpy as np
import pytest

from embedding_model_bench.metrics import DEEPEST, is_relevant, score, top_k
from embedding_model_bench.models import EvalQuestion

QUESTION = EvalQuestion(
    question="Q?", company="acme", doc_type="terms", keywords=["no refunds", "Credits"]
)


def test_relevance_needs_the_document_and_all_keywords() -> None:
    text = "There are NO\nrefunds or credits."
    assert is_relevant(text, "acme", "terms", QUESTION)
    assert not is_relevant(text, "other", "terms", QUESTION)
    assert not is_relevant(text, "acme", "privacy", QUESTION)
    assert not is_relevant("There are no refunds.", "acme", "terms", QUESTION)


def test_top_k_orders_by_dot_product_and_breaks_ties_stably() -> None:
    documents = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    assert top_k(query, documents, 3).tolist() == [[0, 2, 1]]


def filler(*first: int) -> list[int]:
    rest = [doc for doc in range(100, 100 + DEEPEST) if doc not in first]
    return [*first, *rest][:DEEPEST]


def test_scores_match_a_hand_calculation() -> None:
    # Query A: relevant {0} found at rank 1. Query B: relevant {1, 2}, one found at rank 2.
    # Query C: relevant {3} is never found.
    ranked = [filler(0), filler(50, 1), filler(51)]
    scores = score(ranked, [{0}, {1, 2}, {3}])

    assert scores.num_queries == 3
    assert scores.first_relevant_ranks == [1, 2, None]
    assert scores.hit_at_k[1] == pytest.approx(1 / 3)
    assert scores.hit_at_k[5] == pytest.approx(2 / 3)
    assert scores.recall_at_k[5] == pytest.approx((1 + 0.5 + 0) / 3)
    assert scores.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    query_b_ndcg = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert scores.ndcg_at_10 == pytest.approx((1 + query_b_ndcg + 0) / 3)


def test_a_query_with_nothing_relevant_scores_zero_and_is_counted() -> None:
    scores = score([filler(0)], [set()])
    assert scores.num_without_relevant == 1
    assert scores.mrr == 0
    assert scores.ndcg_at_10 == 0
    assert scores.recall_at_k[10] == 0
