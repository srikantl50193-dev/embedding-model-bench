"""Project storage, query latency and embedding time to 100k and 1M chunks.

Measured here, at each size: the memory an exact FAISS index adds, and the search time for one
query at a time on one thread. The corpus repeats the real vectors until it is large enough,
which is fine for timing because search speed doesn't depend on what the vectors say, but means
nothing here measures quality at that size. For the scheme that rescores with float32 vectors,
those vectors are a memory-mapped file on disk, read in place.

Projected, not measured: total query latency adds the benchmark's time to encode one query to the
search time, and embedding time divides the corpus by the benchmark's encoding throughput.
"""

import gc
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from embedding_model_bench.benchmark import BenchmarkInputs, encode_inputs
from embedding_model_bench.embedding import Encoder, cached_embedder
from embedding_model_bench.machine import machine_state, peak_rss_mb
from embedding_model_bench.matryoshka import truncate
from embedding_model_bench.metrics import DEEPEST
from embedding_model_bench.models import BenchmarkResult, FrozenModel, MachineState, ModelSpec
from embedding_model_bench.quantization import (
    SEARCH_THREADS,
    Scheme,
    SearchIndex,
    bytes_per_vector,
    load_faiss,
)
from embedding_model_bench.registry import BGE_BASE, BGE_SMALL
from embedding_model_bench.report import markdown_table

SIZES = (100_000, 1_000_000)
QUERIES = 200
WARMUP = 10
P50, P95 = 50, 95
MB = 1_000_000
HOUR = 3600


class CostConfig(FrozenModel):
    """A model, a vector length and a storage scheme."""

    key: str
    dim: int
    scheme: Scheme

    @property
    def name(self) -> str:
        """A short label for tables."""
        return f"{self.key} {self.dim}d {self.scheme.value}"


# The leaders so far, at full length and shortened, and their compressed forms.
CONFIGS = [
    CostConfig(key=BGE_SMALL.key, dim=384, scheme=Scheme.FLOAT32),
    CostConfig(key=BGE_SMALL.key, dim=384, scheme=Scheme.BINARY_RESCORE_FLOAT),
    CostConfig(key=BGE_BASE.key, dim=768, scheme=Scheme.FLOAT32),
    CostConfig(key=BGE_BASE.key, dim=768, scheme=Scheme.INT8),
    CostConfig(key=BGE_BASE.key, dim=768, scheme=Scheme.BINARY_RESCORE_FLOAT),
    CostConfig(key=BGE_BASE.key, dim=256, scheme=Scheme.FLOAT32),
]


class SizePoint(FrozenModel):
    """One configuration at one corpus size."""

    num_vectors: int
    memory_mb: float
    disk_mb: float
    faiss_index_mb: float
    build_seconds: float
    search_p50_ms: float
    search_p95_ms: float
    projected_query_p95_ms_mps: float
    projected_query_p95_ms_cpu: float
    projected_embed_hours_mps: float
    projected_embed_hours_cpu: float


class CostResult(FrozenModel):
    """Every size for one configuration."""

    config: CostConfig
    query_encode_p95_ms_mps: float
    query_encode_p95_ms_cpu: float
    chunks_per_s_mps: float
    chunks_per_s_cpu: float
    num_queries: int
    search_threads: int
    points: list[SizePoint]
    peak_rss_mb: float
    machine: MachineState


def write_corpus(base: np.ndarray, size: int, path: Path) -> np.memmap:
    """Write `size` vectors to a memory-mapped file by repeating `base`, without holding them all.

    Returns:
        The file, opened for reading.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(size, base.shape[1])
    )
    for start in range(0, size, len(base)):
        end = min(start + len(base), size)
        store[start:end] = base[: end - start]
    store.flush()
    del store
    return np.load(path, mmap_mode="r")


def measure_point(
    scheme: Scheme, corpus: np.ndarray, train: np.ndarray, queries: np.ndarray
) -> tuple[float, float, float, float]:
    """Build an index over `corpus`, then time one query at a time.

    Returns:
        The index's size in MB as FAISS reports it, the build time in seconds, and the median and
        95th percentile search time in milliseconds.
    """
    gc.collect()
    start = time.perf_counter()
    index = SearchIndex(scheme, corpus, train_sample=train)
    build = time.perf_counter() - start
    size_mb = index.index_bytes / MB
    for query in queries[:WARMUP]:
        index.search(query[None, :], DEEPEST)
    times = []
    for query in queries:
        start = time.perf_counter()
        index.search(query[None, :], DEEPEST)
        times.append((time.perf_counter() - start) * 1000)
    del index
    gc.collect()
    return size_mb, build, float(np.percentile(times, P50)), float(np.percentile(times, P95))


def run_cost(  # noqa: PLR0913
    config: CostConfig,
    spec: ModelSpec,
    benchmark: BenchmarkResult,
    inputs: BenchmarkInputs,
    cache_dir: Path,
    scratch_dir: Path,
    load_encoder: Callable[[], Encoder],
    *,
    sizes: Sequence[int] = SIZES,
    log: Callable[[str], None] = lambda _message: None,
) -> CostResult:
    """Measure one configuration at each corpus size.

    Args:
        config: The model, length and scheme.
        spec: The model's spec.
        benchmark: The model's benchmark result, for encoding speeds.
        inputs: The real corpora, whose vectors are repeated to make the large ones.
        cache_dir: Where vectors are cached.
        scratch_dir: Where the large corpora are written, and deleted afterwards.
        load_encoder: Loads the model, only if a vector isn't cached.
        sizes: The corpus sizes.
        log: Receives progress messages.

    Returns:
        The measurements and projections.
    """
    machine = machine_state()
    load_faiss().omp_set_num_threads(SEARCH_THREADS)
    embed = cached_embedder(spec, cache_dir, load_encoder)
    encoded = encode_inputs(inputs, embed, spec.query_prefix, spec.passage_prefix)

    def shorten(vectors: np.ndarray) -> np.ndarray:
        return vectors if config.dim == spec.dim else truncate(vectors, config.dim)

    # Both real corpora together make a more varied base to repeat than either alone.
    base = shorten(np.concatenate([encoded.chunks, encoded.documents]))
    queries = np.ascontiguousarray(shorten(encoded.queries)[:QUERIES])
    timing = {t.device: t for t in benchmark.timings}
    mps, cpu = timing.get("mps", timing["cpu"]), timing["cpu"]
    memory_per, disk_per = bytes_per_vector(config.scheme, config.dim)

    points = []
    for size in sizes:
        log(f"{config.name}: {size:,} vectors")
        path = scratch_dir / f"{config.key}-{config.dim}-{size}.npy"
        try:
            corpus = write_corpus(base, size, path)
            index_mb, build, p50, p95 = measure_point(config.scheme, corpus, base, queries)
            del corpus
        finally:
            path.unlink(missing_ok=True)
        points.append(
            SizePoint(
                num_vectors=size,
                memory_mb=round(memory_per * size / MB, 1),
                disk_mb=round(disk_per * size / MB, 1),
                faiss_index_mb=round(index_mb, 1),
                build_seconds=round(build, 2),
                search_p50_ms=round(p50, 3),
                search_p95_ms=round(p95, 3),
                projected_query_p95_ms_mps=round(mps.query_p95_ms + p95, 2),
                projected_query_p95_ms_cpu=round(cpu.query_p95_ms + p95, 2),
                projected_embed_hours_mps=round(size / mps.encode_chunks_per_s / HOUR, 2),
                projected_embed_hours_cpu=round(size / cpu.encode_chunks_per_s / HOUR, 2),
            )
        )
    return CostResult(
        config=config,
        query_encode_p95_ms_mps=mps.query_p95_ms,
        query_encode_p95_ms_cpu=cpu.query_p95_ms,
        chunks_per_s_mps=mps.encode_chunks_per_s,
        chunks_per_s_cpu=cpu.encode_chunks_per_s,
        num_queries=len(queries),
        search_threads=SEARCH_THREADS,
        points=points,
        peak_rss_mb=round(peak_rss_mb(), 1),
        machine=machine,
    )


def cost_table(results: Sequence[CostResult]) -> str:
    """Render every configuration at every size."""
    headers = [
        "Configuration",
        "Chunks",
        "RAM MB",
        "Disk MB",
        "FAISS index MB",
        "Search p50 ms",
        "Search p95 ms",
        "Query p95 ms (MPS / CPU), projected",
        "Embed hours (MPS / CPU), projected",
    ]
    rows = [
        [
            r.config.name,
            f"{p.num_vectors:,}",
            f"{p.memory_mb:,.0f}",
            f"{p.disk_mb:,.0f}",
            f"{p.faiss_index_mb:,.0f}",
            f"{p.search_p50_ms:.2f}",
            f"{p.search_p95_ms:.2f}",
            f"{p.projected_query_p95_ms_mps:.1f} / {p.projected_query_p95_ms_cpu:.1f}",
            f"{p.projected_embed_hours_mps:.1f} / {p.projected_embed_hours_cpu:.1f}",
        ]
        for r in results
        for p in r.points
    ]
    footer = []
    if results:
        first = results[0]
        footer = [
            "RAM and disk are bytes per vector times chunks. FAISS index MB is what FAISS reports "
            "holding for the stored vectors. Rescoring with float32 vectors also reads them from "
            "disk, where the page cache may keep them in memory.",
            f"Search: exact FAISS indexes, one query at a time on {first.search_threads} thread, "
            f"{first.num_queries} SciFact queries, over repeated real vectors.",
            "Projected query p95 adds the benchmark's p95 to encode one query to the search p95, "
            "so it is conservative. Projected embedding time is chunks divided by the "
            "benchmark's throughput.",
        ]
    return markdown_table(headers, rows, footer)
