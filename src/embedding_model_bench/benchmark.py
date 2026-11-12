"""Measure one embedding model: quality on both corpora, speed on each device, and memory."""

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import numpy as np
from fine_print.models import Chunk

from embedding_model_bench.compare import SELECTION_K, noise_note
from embedding_model_bench.embedding import Embed, Encoder, EncoderFactory, cached_embedder
from embedding_model_bench.files import read_jsonl
from embedding_model_bench.machine import machine_state, mps_allocated_mb, peak_rss_mb
from embedding_model_bench.metrics import DEEPEST, is_relevant, score, top_k
from embedding_model_bench.models import (
    BenchmarkResult,
    EvalQuestion,
    ModelSpec,
    Resources,
    RetrievalScores,
    SciFact,
    Timing,
)
from embedding_model_bench.report import markdown_table

MIB = 1024**2
TIMING_SAMPLE = 500
TIMING_SEED = 0
QUERY_REPEATS = 5
WARMUP_QUERIES = 5
P95 = 95
P50 = 50


def tuning_relevant(chunks: Sequence[Chunk], questions: Sequence[EvalQuestion]) -> list[set[int]]:
    """Return, for each question, the indexes of the chunks that pass Fine Print's keyword test."""
    return [
        {
            index
            for index, chunk in enumerate(chunks)
            if is_relevant(chunk.text, chunk.company, chunk.doc_type, question)
        }
        for question in questions
    ]


def scifact_relevant(scifact: SciFact) -> list[set[int]]:
    """Return, for each query in order, the positions of its relevant documents in the corpus."""
    position = {doc_id: index for index, doc_id in enumerate(scifact.corpus)}
    return [{position[doc] for doc in scifact.qrels[query]} for query in scifact.queries]


@dataclass(frozen=True)
class BenchmarkInputs:
    """What every model is measured on, with the provenance recorded in its result.

    The texts and relevance sets are computed once, on first use, and shared by every model and
    variant scored in the process.
    """

    chunks: Sequence[Chunk]
    questions: Sequence[EvalQuestion]
    scifact: SciFact
    fine_print_commit: str
    corpus_fingerprint: str

    @cached_property
    def chunk_texts(self) -> list[str]:
        """The chunks as they are embedded, led by their section title."""
        return [chunk.embedding_text() for chunk in self.chunks]

    @cached_property
    def question_texts(self) -> list[str]:
        """The tuning questions."""
        return [question.question for question in self.questions]

    @cached_property
    def tuning_relevant(self) -> list[set[int]]:
        """The relevant chunks for each tuning question."""
        return tuning_relevant(self.chunks, self.questions)

    @cached_property
    def scifact_relevant(self) -> list[set[int]]:
        """The relevant documents for each SciFact query."""
        return scifact_relevant(self.scifact)


@dataclass(frozen=True)
class Encoded:
    """One model's vectors for both corpora and both question sets."""

    chunks: np.ndarray
    questions: np.ndarray
    documents: np.ndarray
    queries: np.ndarray


def encode_inputs(
    inputs: BenchmarkInputs, embed: Embed, query_prefix: str, passage_prefix: str
) -> Encoded:
    """Embed everything a model is scored on, with the given prefixes."""
    return Encoded(
        chunks=embed(inputs.chunk_texts, passage_prefix),
        questions=embed(inputs.question_texts, query_prefix),
        documents=embed(list(inputs.scifact.corpus.values()), passage_prefix),
        queries=embed(list(inputs.scifact.queries.values()), query_prefix),
    )


def load_tuning_questions(path: Path) -> list[EvalQuestion]:
    """Read the tuning questions. Held-out questions are deliberately not loaded here."""
    return read_jsonl(path, EvalQuestion)


def score_vectors(
    query_vectors: np.ndarray, document_vectors: np.ndarray, relevant: Sequence[set[int]]
) -> RetrievalScores:
    """Rank the documents for each query by exact search, and score the ranking."""
    return score(top_k(query_vectors, document_vectors, DEEPEST).tolist(), relevant)


def score_encoded(
    inputs: BenchmarkInputs,
    encoded: Encoded,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[RetrievalScores, RetrievalScores]:
    """Score the tuning set with Fine Print's keyword test, and SciFact with its judgments.

    Args:
        inputs: The corpora and questions.
        encoded: The vectors to score.
        transform: Applied to every set of vectors first, such as truncation.

    Returns:
        The tuning scores, then the SciFact scores.
    """

    def apply(vectors: np.ndarray) -> np.ndarray:
        return vectors if transform is None else transform(vectors)

    return (
        score_vectors(apply(encoded.questions), apply(encoded.chunks), inputs.tuning_relevant),
        score_vectors(apply(encoded.queries), apply(encoded.documents), inputs.scifact_relevant),
    )


def measure_timing(
    encoder: Encoder,
    device: str,
    passages: Sequence[str],
    queries: Sequence[str],
    chunk_vectors: np.ndarray,
) -> Timing:
    """Time encoding passages, and answering one query at a time as a live system would.

    This never uses the embedding cache, so timings can be measured again without recomputing
    anything else. A query's time covers encoding it and an exact search over the corpus.

    Args:
        encoder: The model on the device being timed.
        device: The device's name, for the report.
        passages: Texts for the throughput measure, with prefixes. A fixed sample is taken.
        queries: Queries for the latency measure, with prefixes.
        chunk_vectors: The corpus to search.

    Returns:
        The timing. The first few queries are a warm-up and aren't counted.
    """
    sample = random.Random(TIMING_SEED).sample(list(passages), min(TIMING_SAMPLE, len(passages)))
    encoder.encode(sample[:8])
    start = time.perf_counter()
    encoder.encode(sample)
    throughput = len(sample) / (time.perf_counter() - start)

    for query in queries[:WARMUP_QUERIES]:
        encoder.encode([query])
    latencies = []
    for _ in range(QUERY_REPEATS):
        for query in queries:
            start = time.perf_counter()
            top_k(encoder.encode([query]), chunk_vectors, DEEPEST)
            latencies.append((time.perf_counter() - start) * 1000)
    return Timing(
        device=device,
        encode_chunks_per_s=round(throughput, 1),
        query_p50_ms=round(float(np.percentile(latencies, P50)), 2),
        query_p95_ms=round(float(np.percentile(latencies, P95)), 2),
        num_query_samples=len(latencies),
    )


def run_benchmark(
    spec: ModelSpec,
    inputs: BenchmarkInputs,
    cache_dir: Path,
    encoder_factory: EncoderFactory,
    devices: Sequence[str],
    log: Callable[[str], None] = lambda _message: None,
) -> BenchmarkResult:
    """Measure one model. Run one model per process, so that peak memory is its own.

    Args:
        spec: The model.
        inputs: The corpora and questions.
        cache_dir: Where vectors are cached.
        encoder_factory: Loads a model onto a device.
        devices: The devices to time, the first of which computes any uncached vectors.
        log: Receives progress messages.

    Returns:
        The measurements.
    """
    machine = machine_state()
    primary = cache(lambda: encoder_factory(spec, devices[0]))
    embed = cached_embedder(spec, cache_dir, primary, log)
    encoded = encode_inputs(inputs, embed, spec.query_prefix, spec.passage_prefix)
    tuning, scifact = score_encoded(inputs, encoded)

    passages = [spec.passage_prefix + text for text in inputs.chunk_texts]
    queries = [spec.query_prefix + text for text in inputs.question_texts]
    timings = []
    for device in devices:
        encoder = primary() if device == devices[0] else encoder_factory(spec, device)
        log(f"timing on {device}")
        timings.append(measure_timing(encoder, device, passages, queries, encoded.chunks))

    encoder = primary()
    return BenchmarkResult(
        spec=spec,
        fine_print_commit=inputs.fine_print_commit,
        corpus_fingerprint=inputs.corpus_fingerprint,
        num_chunks=len(inputs.chunks),
        tuning=tuning,
        scifact=scifact,
        timings=timings,
        resources=Resources(
            parameters=encoder.num_parameters(),
            weights_mb=round(encoder.weights_bytes() / MIB, 1),
            peak_rss_mb=round(peak_rss_mb(), 1),
            mps_allocated_mb=round(mps_allocated_mb() if "mps" in devices else 0.0, 1),
        ),
        machine=machine,
    )


def summary_table(results: Sequence[BenchmarkResult]) -> str:
    """Render a Markdown table. Tuning scores are Fine Print's hit rate, not true recall."""
    headers = [
        "Model",
        "Tuning hit@1",
        "hit@5",
        "hit@10",
        "MRR",
        "nDCG@10",
        "SciFact recall@5",
        "recall@10",
        "nDCG@10",
        "Chunks/s",
        "p95 query ms",
        "Params (M)",
        "Weights MB",
        "Peak RSS MB",
    ]
    rows = [
        [
            r.spec.key,
            f"{r.tuning.hit_at_k[1]:.3f}",
            f"{r.tuning.hit_at_k[SELECTION_K]:.3f}",
            f"{r.tuning.hit_at_k[10]:.3f}",
            f"{r.tuning.mrr:.3f}",
            f"{r.tuning.ndcg_at_10:.3f}",
            f"{r.scifact.recall_at_k[5]:.3f}",
            f"{r.scifact.recall_at_k[10]:.3f}",
            f"{r.scifact.ndcg_at_10:.3f}",
            f"{r.timings[0].encode_chunks_per_s:.0f} ({r.timings[0].device})",
            f"{r.timings[0].query_p95_ms:.1f}",
            f"{r.resources.parameters / 1e6:.0f}",
            f"{r.resources.weights_mb:.0f}",
            f"{r.resources.peak_rss_mb:.0f}",
        ]
        for r in results
    ]
    footer = [noise_note(results[0].tuning.num_queries)] if results else []
    return markdown_table(headers, rows, footer)
