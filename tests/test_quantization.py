import sys
from pathlib import Path

import numpy as np
import pytest

from conftest_fakes import INPUTS, SPEC, FakeEncoder
from embedding_model_bench.embedding import Encoder
from embedding_model_bench.quantization import (
    CONFIGS,
    Scheme,
    SearchIndex,
    binarize,
    bytes_per_vector,
    load_faiss,
    quantization_table,
    run_quantization,
    unpack_signs,
)
from embedding_model_bench.registry import MODELS

RNG = np.random.default_rng(0)


def unit_vectors(count: int, dim: int = 64) -> np.ndarray:
    vectors = RNG.standard_normal((count, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def must_not_load() -> Encoder:
    raise AssertionError("everything should be cached")


def test_bits_record_the_sign_and_unpack_to_plus_or_minus_one() -> None:
    vectors = np.array([[0.5, -0.1, 0.0, 2.0, -3.0, 1.0, 1.0, -1.0]], dtype=np.float32)
    packed = binarize(vectors)
    assert packed.shape == (1, 1)
    np.testing.assert_array_equal(unpack_signs(packed, 8), [[1, -1, -1, 1, -1, 1, 1, -1]])


@pytest.mark.parametrize("scheme", list(Scheme))
def test_every_scheme_finds_a_document_when_queried_with_itself(scheme: Scheme) -> None:
    documents = unit_vectors(200)
    ranked = SearchIndex(scheme, documents).search(documents[:20], 5)
    assert ranked.shape == (20, 5)
    assert (ranked[:, 0] == np.arange(20)).all()


def test_float32_matches_exact_numpy_search() -> None:
    documents, queries = unit_vectors(300), unit_vectors(10)
    expected = np.argsort(-(queries @ documents.T), axis=1)[:, :10]
    np.testing.assert_array_equal(
        SearchIndex(Scheme.FLOAT32, documents).search(queries, 10), expected
    )


def test_rescoring_never_asks_for_more_candidates_than_exist() -> None:
    documents = unit_vectors(12)
    ranked = SearchIndex(Scheme.BINARY_RESCORE_FLOAT, documents).search(documents[:2], 10)
    assert ranked.shape == (2, 10)
    assert (ranked >= 0).all()


def test_storage_per_vector() -> None:
    assert bytes_per_vector(Scheme.FLOAT32, 768) == (3072, 3072)
    assert bytes_per_vector(Scheme.INT8, 768) == (768, 768)
    assert bytes_per_vector(Scheme.BINARY, 768) == (96, 96)
    assert bytes_per_vector(Scheme.BINARY_RESCORE_QUERY, 768) == (96, 96)
    assert bytes_per_vector(Scheme.BINARY_RESCORE_FLOAT, 768) == (96, 96 + 3072)


def test_every_configuration_names_a_known_model_at_a_valid_length() -> None:
    for config in CONFIGS:
        assert 0 < config.dim <= MODELS[config.key].dim
        assert config.dim % 8 == 0


def test_a_run_compares_every_scheme_with_float32(tmp_path: Path) -> None:
    result = run_quantization(SPEC, 32, INPUTS, tmp_path, FakeEncoder, latency_corpus=50)
    assert [s.scheme for s in result.schemes] == list(Scheme)
    first = result.schemes[0].tuning_vs_float32
    assert (first.wins, first.losses) == (0, 0)
    assert result.schemes[0].tuning.num_queries == len(INPUTS.questions)
    assert all(s.search_p95_ms >= 0 for s in result.schemes)


def test_a_rerun_loads_no_model(tmp_path: Path) -> None:
    run_quantization(SPEC, SPEC.dim, INPUTS, tmp_path, FakeEncoder, latency_corpus=50)
    run_quantization(SPEC, SPEC.dim, INPUTS, tmp_path, must_not_load, latency_corpus=50)


def test_the_table_explains_the_search_times(tmp_path: Path) -> None:
    result = run_quantization(SPEC, 32, INPUTS, tmp_path, FakeEncoder, latency_corpus=50)
    table = quantization_table([result])
    assert Scheme.BINARY_RESCORE_FLOAT.value in table
    assert "50 vectors" in table
    assert f"n={len(INPUTS.questions)}" in table


def test_faiss_refuses_to_share_a_process_with_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="make benchmark"):
        load_faiss()


@pytest.mark.parametrize("scheme", list(Scheme))
def test_identical_documents_tie_in_index_order_as_in_the_benchmark(scheme: Scheme) -> None:
    documents = unit_vectors(30)
    documents[20] = documents[5]  # an identical chunk later in the corpus
    ranked = SearchIndex(scheme, documents).search(documents[5:6], 2)
    assert ranked[0].tolist() == [5, 20]
