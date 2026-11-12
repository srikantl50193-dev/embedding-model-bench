from pathlib import Path

import numpy as np

from conftest_fakes import INPUTS, SPEC, FakeEncoder
from embedding_model_bench.benchmark import run_benchmark
from embedding_model_bench.cost import CostConfig, cost_table, run_cost, write_corpus
from embedding_model_bench.quantization import Scheme, SearchIndex


def test_a_large_corpus_repeats_the_real_vectors(tmp_path: Path) -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    corpus = write_corpus(base, 7, tmp_path / "c.npy")
    assert corpus.shape == (7, 4)
    np.testing.assert_array_equal(corpus[3:6], base)
    np.testing.assert_array_equal(corpus[6], base[0])


def test_faiss_reports_the_bytes_per_vector_the_projection_assumes() -> None:
    documents = np.random.default_rng(0).standard_normal((100, 64)).astype(np.float32)
    assert SearchIndex(Scheme.FLOAT32, documents).index_bytes == 100 * 64 * 4
    assert SearchIndex(Scheme.INT8, documents).index_bytes == 100 * 64
    assert SearchIndex(Scheme.BINARY_RESCORE_FLOAT, documents).index_bytes == 100 * 64 // 8


def test_a_run_measures_each_size_and_projects_from_the_benchmark(tmp_path: Path) -> None:
    benchmark = run_benchmark(
        SPEC, INPUTS, tmp_path / "cache", lambda _s, _d: FakeEncoder(), ["cpu"]
    )
    config = CostConfig(key=SPEC.key, dim=32, scheme=Scheme.BINARY_RESCORE_FLOAT)
    scratch = tmp_path / "scratch"
    result = run_cost(
        config, SPEC, benchmark, INPUTS, tmp_path / "cache", scratch, FakeEncoder, sizes=(50, 200)
    )
    assert [p.num_vectors for p in result.points] == [50, 200]
    big = result.points[1]
    assert big.faiss_index_mb == round(200 * 32 / 8 / 1e6, 1)
    assert big.disk_mb == round(200 * (32 / 8 + 32 * 4) / 1e6, 1)
    timing = benchmark.timings[0]
    assert big.projected_query_p95_ms_cpu == round(timing.query_p95_ms + big.search_p95_ms, 2)
    assert big.projected_embed_hours_cpu == round(200 / timing.encode_chunks_per_s / 3600, 2)
    assert not list(scratch.glob("*.npy")), "large corpora are deleted afterwards"
    assert "200" in cost_table([result])
