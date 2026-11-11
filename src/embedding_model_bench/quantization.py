"""Measure what storing vectors as int8 or binary costs in quality, and saves in memory and time.

Search uses FAISS, a widely used vector search library, because numpy has no fast int8 or bit
operations and would make the compressed formats look slower than they are. Every index is
exact (flat), so only the number format differs from the float32 baseline.

Schemes, from largest to smallest in memory:

- float32: the baseline, 4 bytes per value.
- int8: FAISS's 8-bit scalar quantizer, with a range per dimension learned from the corpus.
  Queries stay float32. 1 byte per value.
- binary: one bit per value, the sign, compared by Hamming distance. 1/32 of float32.
- binary, rescored: binary search for `RESCORE_MULTIPLIER` times as many candidates, which
  are then re-ranked. "query" re-ranks with the float32 query against the stored bits, so it
  needs nothing more. "float32" re-ranks against float32 vectors, which have to be kept too,
  typically on disk while the bits stay in memory.
"""

import sys
import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from embedding_model_bench.benchmark import BenchmarkInputs, scifact_relevant, tuning_relevant
from embedding_model_bench.compare import PairedComparison, noise_note, paired_comparison
from embedding_model_bench.embedding import Encoder, embed_cached
from embedding_model_bench.machine import machine_state, peak_rss_mb
from embedding_model_bench.matryoshka import truncate
from embedding_model_bench.metrics import DEEPEST, score
from embedding_model_bench.models import FrozenModel, MachineState, ModelSpec, RetrievalScores
from embedding_model_bench.registry import ARCTIC_M, BGE_BASE, BGE_SMALL

RESCORE_MULTIPLIER = 4
# Extra results fetched so that exact ties at the cut-off can be ordered by index.
TIE_MARGIN = 10
LATENCY_CORPUS = 100_000
LATENCY_QUERIES = 300
WARMUP = 10
SEARCH_THREADS = 1
P50, P95 = 50, 95
FLOAT32_BYTES = 4
BITS_PER_BYTE = 8


def load_faiss() -> ModuleType:
    """Import FAISS, refusing if torch is already loaded in this process.

    On macOS, torch and FAISS each bundle their own OpenMP runtime, and loading both aborts the
    process. Quantization works from cached vectors, so it never needs torch.

    Raises:
        RuntimeError: If torch is loaded, which usually means a vector wasn't cached.
    """
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch is loaded, and FAISS can't share a process with it on macOS. If a model was "
            "loaded to encode missing vectors, run `make benchmark` first to fill the cache."
        )
    import faiss  # noqa: PLC0415 - slow to import, and only needed here

    return faiss


class Scheme(StrEnum):
    """How the corpus vectors are stored and searched."""

    FLOAT32 = "float32"
    INT8 = "int8"
    BINARY = "binary"
    BINARY_RESCORE_QUERY = "binary, rescored with float32 query"
    BINARY_RESCORE_FLOAT = "binary, rescored with float32 vectors"


class Config(FrozenModel):
    """A model at one vector length."""

    key: str
    dim: int


# The leaders so far, and Arctic, whose card says it was trained to compress well.
CONFIGS = [
    Config(key=BGE_BASE.key, dim=768),
    Config(key=BGE_BASE.key, dim=256),
    Config(key=BGE_SMALL.key, dim=384),
    Config(key=ARCTIC_M.key, dim=768),
    Config(key=ARCTIC_M.key, dim=256),
]


class SchemeResult(FrozenModel):
    """Quality, memory and search time for one storage scheme."""

    scheme: Scheme
    tuning: RetrievalScores
    scifact: RetrievalScores
    tuning_vs_float32: PairedComparison
    scifact_vs_float32: PairedComparison
    memory_bytes_per_vector: float
    disk_bytes_per_vector: float
    search_p50_ms: float
    search_p95_ms: float


class QuantizationResult(FrozenModel):
    """Every scheme for one model at one length. The first scheme is float32."""

    spec: ModelSpec
    dim: int
    fine_print_commit: str
    corpus_fingerprint: str
    latency_corpus_size: int
    latency_queries: int
    search_threads: int
    schemes: list[SchemeResult]
    peak_rss_mb: float
    machine: MachineState


def binarize(vectors: np.ndarray) -> np.ndarray:
    """Pack each value's sign into one bit: 1 for positive, 0 otherwise."""
    return np.packbits(vectors > 0, axis=1)


def unpack_signs(packed: np.ndarray, dim: int) -> np.ndarray:
    """Turn packed bits back into +1 and -1 values."""
    return np.unpackbits(packed, axis=1)[:, :dim].astype(np.float32) * 2 - 1


class SearchIndex:
    """A FAISS index for one scheme, holding a corpus and answering top-k queries."""

    def __init__(self, scheme: Scheme, documents: np.ndarray) -> None:
        faiss = load_faiss()
        self.scheme = scheme
        self.dim = documents.shape[1]
        self._documents = np.ascontiguousarray(documents, dtype=np.float32)
        self._index: Any
        if scheme == Scheme.FLOAT32:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self._documents)
        elif scheme == Scheme.INT8:
            self._index = faiss.IndexScalarQuantizer(
                self.dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
            )
            self._index.train(self._documents)
            self._index.add(self._documents)
        else:
            self._bits = binarize(self._documents)
            self._index = faiss.IndexBinaryFlat(self.dim)
            self._index.add(self._bits)

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return the indexes of each query's k best documents, best first.

        Exact ties go to the lower index, as in `metrics.top_k`. The corpus has identical chunks
        in different documents, so without this, float32 here wouldn't reproduce the benchmark.
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        count = self._documents.shape[0]
        if self.scheme in (Scheme.FLOAT32, Scheme.INT8):
            scores, ids = self._index.search(queries, min(k + TIE_MARGIN, count))
            return np.stack([_best_first(i, -s, k) for s, i in zip(scores, ids, strict=True)])
        if self.scheme == Scheme.BINARY:
            distances, ids = self._index.search(binarize(queries), min(k + TIE_MARGIN, count))
            return np.stack([_best_first(i, d, k) for d, i in zip(distances, ids, strict=True)])
        depth = min(k * RESCORE_MULTIPLIER, count)
        candidates = self._index.search(binarize(queries), depth)[1]
        ranked = []
        for query, ids in zip(queries, candidates, strict=True):
            if self.scheme == Scheme.BINARY_RESCORE_QUERY:
                scores = unpack_signs(self._bits[ids], self.dim) @ query
            else:
                scores = self._documents[ids] @ query
            ranked.append(_best_first(ids, -scores, k))
        return np.stack(ranked)


def _best_first(ids: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    """Order ids by key, smallest first, breaking exact ties by the lower id, and keep k."""
    return ids[np.lexsort((ids, keys))][:k]


def bytes_per_vector(scheme: Scheme, dim: int) -> tuple[float, float]:
    """Return the bytes held in memory and on disk per vector, before any index overhead."""
    float32, int8, bits = dim * FLOAT32_BYTES, float(dim), dim / BITS_PER_BYTE
    if scheme == Scheme.FLOAT32:
        return float32, float32
    if scheme == Scheme.INT8:
        return int8, int8
    if scheme == Scheme.BINARY_RESCORE_FLOAT:
        return bits, bits + float32
    return bits, bits


def time_search(
    scheme: Scheme, documents: np.ndarray, queries: np.ndarray, corpus_size: int
) -> tuple[float, float]:
    """Time one query at a time against a corpus of `corpus_size` vectors.

    The corpus repeats the real vectors until it is large enough. Speed doesn't depend on what
    the vectors say, so this measures a realistic size without encoding more text.

    Returns:
        The median and 95th percentile time per query, in milliseconds.
    """
    repeats = -(-corpus_size // len(documents))
    corpus = np.tile(documents, (repeats, 1))[:corpus_size]
    index = SearchIndex(scheme, corpus)
    for query in queries[:WARMUP]:
        index.search(query[None, :], DEEPEST)
    times = []
    for query in queries:
        start = time.perf_counter()
        index.search(query[None, :], DEEPEST)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, P50)), float(np.percentile(times, P95))


def run_quantization(
    spec: ModelSpec,
    dim: int,
    inputs: BenchmarkInputs,
    cache_dir: Path,
    load_encoder: Callable[[], Encoder],
    latency_corpus: int = LATENCY_CORPUS,
    log: Callable[[str], None] = lambda _message: None,
) -> QuantizationResult:
    """Score and time every scheme for one model at one vector length.

    Args:
        spec: The model, used with its card's prefixes.
        dim: The vector length. Shorter than the model's own means truncated, as in Matryoshka.
        inputs: The corpora and questions.
        cache_dir: Where vectors are cached.
        load_encoder: Loads the model, which only happens if a vector isn't cached.
        latency_corpus: How many vectors to search when timing.
        log: Receives progress messages.

    Returns:
        Each scheme's results, compared with float32 at the same length.
    """
    machine = machine_state()
    encoder = cache(load_encoder)

    def embed(texts: Sequence[str], prefix: str) -> np.ndarray:
        vectors = embed_cached(spec, texts, prefix, cache_dir, encoder)[0]
        return vectors if dim == spec.dim else truncate(vectors, dim)

    chunks = embed([c.embedding_text() for c in inputs.chunks], spec.passage_prefix)
    questions = embed([q.question for q in inputs.questions], spec.query_prefix)
    documents = embed(list(inputs.scifact.corpus.values()), spec.passage_prefix)
    queries = embed(list(inputs.scifact.queries.values()), spec.query_prefix)
    tuning_sets = tuning_relevant(inputs.chunks, inputs.questions)
    scifact_sets = scifact_relevant(inputs.scifact)
    load_faiss().omp_set_num_threads(SEARCH_THREADS)

    scored = []
    for scheme in Scheme:
        log(f"{dim} dims, {scheme}")
        tuning = score(SearchIndex(scheme, chunks).search(questions, DEEPEST).tolist(), tuning_sets)
        scifact = score(
            SearchIndex(scheme, documents).search(queries, DEEPEST).tolist(), scifact_sets
        )
        p50, p95 = time_search(scheme, documents, queries[:LATENCY_QUERIES], latency_corpus)
        scored.append((scheme, tuning, scifact, p50, p95))

    _, float_tuning, float_scifact, _, _ = scored[0]
    return QuantizationResult(
        spec=spec,
        dim=dim,
        fine_print_commit=inputs.fine_print_commit,
        corpus_fingerprint=inputs.corpus_fingerprint,
        latency_corpus_size=latency_corpus,
        latency_queries=min(LATENCY_QUERIES, len(queries)),
        search_threads=SEARCH_THREADS,
        schemes=[
            SchemeResult(
                scheme=scheme,
                tuning=tuning,
                scifact=scifact,
                tuning_vs_float32=paired_comparison(float_tuning, tuning),
                scifact_vs_float32=paired_comparison(float_scifact, scifact),
                memory_bytes_per_vector=bytes_per_vector(scheme, dim)[0],
                disk_bytes_per_vector=bytes_per_vector(scheme, dim)[1],
                search_p50_ms=round(p50, 3),
                search_p95_ms=round(p95, 3),
            )
            for scheme, tuning, scifact, p50, p95 in scored
        ],
        peak_rss_mb=round(peak_rss_mb(), 1),
        machine=machine,
    )


def quantization_table(results: Sequence[QuantizationResult]) -> str:
    """Render each configuration's schemes. Wins and losses are against its float32."""
    headers = [
        "Model",
        "Dims",
        "Scheme",
        "Memory B/vector",
        "Tuning hit@5",
        "Wins / losses",
        "MRR",
        "SciFact nDCG@10",
        "SciFact wins / losses",
        "p",
        "Search p95 ms",
    ]
    rows = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for result in results:
        for s in result.schemes:
            tuning, scifact = s.tuning_vs_float32, s.scifact_vs_float32
            hits = sum(r is not None and r <= tuning.k for r in s.tuning.first_relevant_ranks)
            cells = [
                result.spec.key,
                str(result.dim),
                s.scheme.value,
                f"{s.memory_bytes_per_vector:,.0f}",
                f"{hits}/{s.tuning.num_queries}",
                f"{tuning.wins} / {tuning.losses}",
                f"{s.tuning.mrr:.3f}",
                f"{s.scifact.ndcg_at_10:.3f}",
                f"{scifact.wins} / {scifact.losses}",
                f"{scifact.sign_test_p:.3f}",
                f"{s.search_p95_ms:.2f}",
            ]
            rows.append("| " + " | ".join(cells) + " |")
    if results:
        first = results[0]
        rows += [
            "",
            noise_note(first.schemes[0].tuning.num_queries),
            f"Search times: one query at a time over {first.latency_corpus_size:,} vectors, "
            f"{first.search_threads} thread, exact FAISS indexes, {first.latency_queries} "
            "SciFact queries. They exclude encoding the query.",
        ]
    return "\n".join(rows) + "\n"
