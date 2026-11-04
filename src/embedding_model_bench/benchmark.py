"""Measure one embedding model: quality on both corpora, speed on each device, and memory."""

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fine_print.models import Chunk

from embedding_model_bench.compare import noise_note
from embedding_model_bench.embedding import Encoder, EncoderFactory, embed_cached
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

MIB = 1024**2
TIMING_SAMPLE = 500
TIMING_SEED = 0
QUERY_REPEATS = 5
WARMUP_QUERIES = 5
P95 = 95
P50 = 50


@dataclass(frozen=True)
class BenchmarkInputs:
    """What every model is measured on, with the provenance recorded in its result."""

    chunks: Sequence[Chunk]
    questions: Sequence[EvalQuestion]
    scifact: SciFact
    fine_print_commit: str
    corpus_fingerprint: str


def load_tuning_questions(path: Path) -> list[EvalQuestion]:
    """Read the tuning questions. Held-out questions are deliberately not loaded here."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalQuestion.model_validate_json(line) for line in lines if line.strip()]


def score_tuning(
    query_vectors: np.ndarray,
    chunk_vectors: np.ndarray,
    chunks: Sequence[Chunk],
    questions: Sequence[EvalQuestion],
) -> RetrievalScores:
    """Score Fine Print's questions using its keyword test for relevance."""
    relevant = [
        {
            index
            for index, chunk in enumerate(chunks)
            if is_relevant(chunk.text, chunk.company, chunk.doc_type, question)
        }
        for question in questions
    ]
    return score(top_k(query_vectors, chunk_vectors, DEEPEST).tolist(), relevant)


def score_scifact(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    scifact: SciFact,
) -> RetrievalScores:
    """Score SciFact using its human relevance judgments."""
    position = {doc_id: index for index, doc_id in enumerate(document_ids)}
    relevant = [{position[doc] for doc in scifact.qrels[query]} for query in query_ids]
    return score(top_k(query_vectors, document_vectors, DEEPEST).tolist(), relevant)


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
    chunks, questions, scifact = inputs.chunks, inputs.questions, inputs.scifact
    machine = machine_state()
    loaded: dict[str, Encoder] = {}

    def primary() -> Encoder:
        if devices[0] not in loaded:
            loaded[devices[0]] = encoder_factory(spec, devices[0])
        return loaded[devices[0]]

    def embed(texts: Sequence[str], prefix: str, label: str) -> np.ndarray:
        vectors, hit = embed_cached(spec, texts, prefix, cache_dir, primary)
        log(f"{label}: {len(texts)} texts, {'from cache' if hit else 'computed'}")
        return vectors

    chunk_vectors = embed([c.embedding_text() for c in chunks], spec.passage_prefix, "chunks")
    query_vectors = embed([q.question for q in questions], spec.query_prefix, "questions")
    tuning = score_tuning(query_vectors, chunk_vectors, chunks, questions)

    document_ids, query_ids = list(scifact.corpus), list(scifact.queries)
    scifact_docs = embed(list(scifact.corpus.values()), spec.passage_prefix, "SciFact documents")
    scifact_queries = embed(list(scifact.queries.values()), spec.query_prefix, "SciFact queries")
    scifact_scores = score_scifact(scifact_queries, scifact_docs, query_ids, document_ids, scifact)

    passages = [spec.passage_prefix + c.embedding_text() for c in chunks]
    queries = [spec.query_prefix + q.question for q in questions]
    timings = []
    for device in devices:
        encoder = primary() if device == devices[0] else encoder_factory(spec, device)
        log(f"timing on {device}")
        timings.append(measure_timing(encoder, device, passages, queries, chunk_vectors))

    encoder = primary()
    return BenchmarkResult(
        spec=spec,
        fine_print_commit=inputs.fine_print_commit,
        corpus_fingerprint=inputs.corpus_fingerprint,
        num_chunks=len(chunks),
        tuning=tuning,
        scifact=scifact_scores,
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
    rows = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for r in results:
        timing = r.timings[0]
        cells = [
            r.spec.key,
            f"{r.tuning.hit_at_k[1]:.3f}",
            f"{r.tuning.hit_at_k[5]:.3f}",
            f"{r.tuning.hit_at_k[10]:.3f}",
            f"{r.tuning.mrr:.3f}",
            f"{r.tuning.ndcg_at_10:.3f}",
            f"{r.scifact.recall_at_k[5]:.3f}",
            f"{r.scifact.recall_at_k[10]:.3f}",
            f"{r.scifact.ndcg_at_10:.3f}",
            f"{timing.encode_chunks_per_s:.0f} ({timing.device})",
            f"{timing.query_p95_ms:.1f}",
            f"{r.resources.parameters / 1e6:.0f}",
            f"{r.resources.weights_mb:.0f}",
            f"{r.resources.peak_rss_mb:.0f}",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    if results:
        rows += ["", noise_note(results[0].tuning.num_queries)]
    return "\n".join(rows) + "\n"
