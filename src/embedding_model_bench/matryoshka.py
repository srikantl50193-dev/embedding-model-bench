"""Measure what shortening embeddings costs in quality and saves in storage.

Vectors are truncated from the cached full-length ones, so this needs no model unless a vector
is missing. Arctic and Nomic were trained for truncation. The BGE models weren't, and are here
as controls: bge-base leads so far, so it matters whether it could be shortened too.
"""

from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import numpy as np

from embedding_model_bench.benchmark import BenchmarkInputs, score_scifact, score_tuning
from embedding_model_bench.compare import PairedComparison, noise_note, paired_comparison
from embedding_model_bench.embedding import Encoder, embed_cached
from embedding_model_bench.machine import peak_rss_mb
from embedding_model_bench.models import FrozenModel, ModelSpec, RetrievalScores
from embedding_model_bench.registry import ARCTIC_M, BGE_BASE, BGE_SMALL, NOMIC

BYTES_PER_FLOAT32 = 4
MB = 1_000_000
ONE_MILLION = 1_000_000
DIMS = (512, 384, 256, 128, 64)


class Truncation(FrozenModel):
    """How a model's vectors are shortened."""

    layer_norm: bool = False
    trained_for_truncation: bool


# Nomic's card applies layer normalization before truncating. Arctic's truncates directly.
TRUNCATIONS = {
    ARCTIC_M.key: Truncation(trained_for_truncation=True),
    NOMIC.key: Truncation(trained_for_truncation=True, layer_norm=True),
    BGE_BASE.key: Truncation(trained_for_truncation=False),
    BGE_SMALL.key: Truncation(trained_for_truncation=False),
}


class TruncatedVariant(FrozenModel):
    """One length of vector, compared with the model's full length."""

    dim: int
    tuning: RetrievalScores
    scifact: RetrievalScores
    tuning_vs_full: PairedComparison
    scifact_vs_full: PairedComparison
    float32_mb_per_million: float


class MatryoshkaResult(FrozenModel):
    """Every vector length tested for one model. The first variant is the full length."""

    spec: ModelSpec
    truncation: Truncation
    fine_print_commit: str
    corpus_fingerprint: str
    variants: list[TruncatedVariant]
    peak_rss_mb: float


def truncate(vectors: np.ndarray, dim: int, *, layer_norm: bool = False) -> np.ndarray:
    """Shorten vectors to `dim` values and make them length 1 again.

    Layer normalization subtracts each vector's mean and divides by its standard deviation. The
    division only rescales the vector, and the final normalization undoes any rescaling, so only
    the subtraction is needed. That is also why it can run on already-normalized vectors: their
    length doesn't change the result. A test checks this against torch's `layer_norm`.

    Args:
        vectors: The full vectors, of shape (count, full dim).
        dim: How many values to keep.
        layer_norm: Whether to apply layer normalization before truncating.

    Returns:
        Length-1 vectors of shape (count, dim).

    Raises:
        ValueError: If `dim` is out of range.
    """
    if not 0 < dim <= vectors.shape[1]:
        raise ValueError(f"Can't truncate {vectors.shape[1]} dimensions to {dim}")
    centred = vectors - vectors.mean(axis=1, keepdims=True) if layer_norm else vectors
    short = centred[:, :dim]
    norms = np.linalg.norm(short, axis=1, keepdims=True)
    return short / np.where(norms == 0, 1, norms)


def float32_mb_per_million(dim: int) -> float:
    """Return the storage for a million float32 vectors, before any index overhead."""
    return dim * BYTES_PER_FLOAT32 * ONE_MILLION / MB


def run_matryoshka(
    spec: ModelSpec,
    truncation: Truncation,
    inputs: BenchmarkInputs,
    cache_dir: Path,
    load_encoder: Callable[[], Encoder],
    dims: Sequence[int] = DIMS,
    log: Callable[[str], None] = lambda _message: None,
) -> MatryoshkaResult:
    """Score one model at its full length and each shorter one, on both corpora.

    Args:
        spec: The model, used with its card's prefixes.
        truncation: How to shorten its vectors.
        inputs: The corpora and questions.
        cache_dir: Where vectors are cached.
        load_encoder: Loads the model, which only happens if a vector isn't cached.
        dims: The lengths to try. Any not shorter than the model's own length are skipped.
        log: Receives progress messages.

    Returns:
        The scores at each length, each compared with the full length.
    """
    encoder = cache(load_encoder)

    def embed(texts: Sequence[str], prefix: str) -> np.ndarray:
        return embed_cached(spec, texts, prefix, cache_dir, encoder)[0]

    chunks = embed([c.embedding_text() for c in inputs.chunks], spec.passage_prefix)
    questions = embed([q.question for q in inputs.questions], spec.query_prefix)
    document_ids, query_ids = list(inputs.scifact.corpus), list(inputs.scifact.queries)
    documents = embed(list(inputs.scifact.corpus.values()), spec.passage_prefix)
    queries = embed(list(inputs.scifact.queries.values()), spec.query_prefix)

    def evaluate(short: Callable[[np.ndarray], np.ndarray]) -> tuple[RetrievalScores, ...]:
        return (
            score_tuning(short(questions), short(chunks), inputs.chunks, inputs.questions),
            score_scifact(
                short(queries), short(documents), query_ids, document_ids, inputs.scifact
            ),
        )

    # The reference is the benchmark's own setup, so it is left exactly as the model produced it.
    full_tuning, full_scifact = evaluate(lambda vectors: vectors)
    scored = [(spec.dim, full_tuning, full_scifact)]
    for dim in (d for d in dims if d < spec.dim):
        log(f"{dim} dimensions")
        tuning, scifact = evaluate(
            lambda vectors, d=dim: truncate(vectors, d, layer_norm=truncation.layer_norm)
        )
        scored.append((dim, tuning, scifact))

    return MatryoshkaResult(
        spec=spec,
        truncation=truncation,
        fine_print_commit=inputs.fine_print_commit,
        corpus_fingerprint=inputs.corpus_fingerprint,
        variants=[
            TruncatedVariant(
                dim=dim,
                tuning=tuning,
                scifact=scifact,
                tuning_vs_full=paired_comparison(full_tuning, tuning),
                scifact_vs_full=paired_comparison(full_scifact, scifact),
                float32_mb_per_million=float32_mb_per_million(dim),
            )
            for dim, tuning, scifact in scored
        ],
        peak_rss_mb=round(peak_rss_mb(), 1),
    )


def matryoshka_table(results: Sequence[MatryoshkaResult]) -> str:
    """Render each model's lengths. Wins and losses are against that model's full length."""
    headers = [
        "Model",
        "Trained to truncate?",
        "Dims",
        "MB per 1M vectors",
        "Tuning hit@5",
        "Wins / losses",
        "p",
        "MRR",
        "SciFact nDCG@10",
        "SciFact wins / losses",
        "p",
    ]
    rows = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for result in results:
        for v in result.variants:
            tuning, scifact = v.tuning_vs_full, v.scifact_vs_full
            hits = sum(r is not None and r <= tuning.k for r in v.tuning.first_relevant_ranks)
            cells = [
                result.spec.key,
                "yes" if result.truncation.trained_for_truncation else "no",
                str(v.dim),
                f"{v.float32_mb_per_million:,.0f}",
                f"{hits}/{v.tuning.num_queries}",
                f"{tuning.wins} / {tuning.losses}",
                f"{tuning.sign_test_p:.3f}",
                f"{v.tuning.mrr:.3f}",
                f"{v.scifact.ndcg_at_10:.3f}",
                f"{scifact.wins} / {scifact.losses}",
                f"{scifact.sign_test_p:.3f}",
            ]
            rows.append("| " + " | ".join(cells) + " |")
    if results:
        rows += ["", noise_note(results[0].variants[0].tuning.num_queries)]
    return "\n".join(rows) + "\n"
