"""Exact search and retrieval metrics, written to be checked by hand in the tests."""

import math
from collections.abc import Sequence

import numpy as np

from embedding_model_bench.models import EvalQuestion, RetrievalScores

K_VALUES = (1, 3, 5, 10)
DEEPEST = max(K_VALUES)
NDCG_K = 10


def is_relevant(text: str, company: str, doc_type: str, question: EvalQuestion) -> bool:
    """Apply Fine Print's relevance test: right document, and every keyword present.

    Ported because `fine_print.evaluate` loads chromadb. Case and line breaks are ignored.
    """
    if (company, doc_type) != (question.company, question.doc_type):
        return False
    normalized = " ".join(text.lower().split())
    return all(keyword.lower() in normalized for keyword in question.keywords)


def top_k(queries: np.ndarray, documents: np.ndarray, k: int) -> np.ndarray:
    """Return the indexes of each query's k best documents by exact dot product.

    Exact search keeps an index's approximation out of the comparison between models. The sort
    is stable, so ties always resolve the same way.
    """
    scores = queries @ documents.T
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def _ndcg(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """Compute nDCG with binary gains. A query with nothing relevant scores 0."""
    gain = sum(1 / math.log2(i + 2) for i, doc in enumerate(ranked[:k]) if doc in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return gain / ideal if ideal else 0.0


def score(ranked: Sequence[Sequence[int]], relevant: Sequence[set[int]]) -> RetrievalScores:
    """Score ranked results against the relevant documents of each query.

    Args:
        ranked: For each query, document indexes from best to worst, at least DEEPEST long.
        relevant: For each query, the indexes of its relevant documents.

    Returns:
        The scores. As in Fine Print, a first hit below DEEPEST counts as a miss for MRR.
    """
    count = len(ranked)
    first_ranks: list[int | None] = []
    for docs, rel in zip(ranked, relevant, strict=True):
        hit = next((i for i, doc in enumerate(docs[:DEEPEST], start=1) if doc in rel), None)
        first_ranks.append(hit)
    return RetrievalScores(
        num_queries=count,
        num_without_relevant=sum(not rel for rel in relevant),
        hit_at_k={
            k: sum(rank is not None and rank <= k for rank in first_ranks) / count for k in K_VALUES
        },
        recall_at_k={
            k: sum(
                len(set(docs[:k]) & rel) / len(rel) if rel else 0.0
                for docs, rel in zip(ranked, relevant, strict=True)
            )
            / count
            for k in K_VALUES
        },
        mrr=sum(1 / rank for rank in first_ranks if rank is not None) / count,
        ndcg_at_10=sum(_ndcg(docs, rel, NDCG_K) for docs, rel in zip(ranked, relevant, strict=True))
        / count,
        first_relevant_ranks=first_ranks,
    )
