from pathlib import Path

import numpy as np
import pytest

from conftest_fakes import INPUTS, SPEC, FakeEncoder
from embedding_model_bench.embedding import Encoder
from embedding_model_bench.matryoshka import (
    TRUNCATIONS,
    Truncation,
    float32_mb_per_million,
    matryoshka_table,
    run_matryoshka,
    truncate,
)
from embedding_model_bench.registry import MODELS, NOMIC

PLAIN = Truncation(trained_for_truncation=True)
DIMS = (32, 8, 64, 128)


def must_not_load() -> Encoder:
    raise AssertionError("everything should be cached")


def test_truncated_vectors_keep_the_first_values_and_have_length_one() -> None:
    vectors = np.array([[3.0, 4.0, 12.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    short = truncate(vectors, 2)
    np.testing.assert_allclose(short, [[0.6, 0.8], [0.0, 1.0]], atol=1e-6)


def test_an_all_zero_prefix_stays_zero_rather_than_dividing_by_zero() -> None:
    vectors = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(truncate(vectors, 2), [[0.0, 0.0]])


@pytest.mark.parametrize("dim", [0, 4])
def test_an_impossible_length_is_rejected(dim: int) -> None:
    with pytest.raises(ValueError, match="truncate"):
        truncate(np.ones((1, 3), dtype=np.float32), dim)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """torch's `layer_norm` without weights: (x - mean) / sqrt(var + eps), per row.

    Written out here because tests can't load torch alongside FAISS on macOS.
    """
    mean = x.mean(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(x.var(axis=1, keepdims=True) + eps)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_layer_norm_matches_nomics_recipe_from_normalized_vectors() -> None:
    raw = np.random.default_rng(0).standard_normal((5, 768)) * 7 + 2
    # The recipe on Nomic's card, run on the raw pooled output.
    recipe = normalize(layer_norm(raw)[:, :256])

    # This project's route, starting from the normalized vectors in the cache.
    np.testing.assert_allclose(truncate(normalize(raw), 256, layer_norm=True), recipe, atol=1e-5)


def test_only_nomic_uses_layer_norm_and_every_model_is_known() -> None:
    assert set(TRUNCATIONS) <= set(MODELS)
    assert [key for key, t in TRUNCATIONS.items() if t.layer_norm] == [NOMIC.key]


def test_storage_is_four_bytes_per_value() -> None:
    assert float32_mb_per_million(768) == 3072


def test_the_first_variant_is_the_full_length_and_longer_dims_are_skipped(tmp_path: Path) -> None:
    result = run_matryoshka(SPEC, PLAIN, INPUTS, tmp_path, FakeEncoder, DIMS)
    assert [v.dim for v in result.variants] == [SPEC.dim, 32, 8]
    full = result.variants[0]
    assert (full.tuning_vs_full.wins, full.tuning_vs_full.losses) == (0, 0)
    assert full.tuning.num_queries == len(INPUTS.questions)


def test_a_rerun_loads_no_model(tmp_path: Path) -> None:
    run_matryoshka(SPEC, PLAIN, INPUTS, tmp_path, FakeEncoder, DIMS)
    run_matryoshka(SPEC, PLAIN, INPUTS, tmp_path, must_not_load, DIMS)


def test_the_model_is_loaded_once_however_much_is_uncached(tmp_path: Path) -> None:
    loads = []

    def load() -> Encoder:
        loads.append(1)
        return FakeEncoder()

    run_matryoshka(SPEC, PLAIN, INPUTS, tmp_path, load, DIMS)
    assert len(loads) == 1


def test_the_table_shows_lengths_storage_and_the_question_count(tmp_path: Path) -> None:
    result = run_matryoshka(SPEC, PLAIN, INPUTS, tmp_path, FakeEncoder, DIMS)
    table = matryoshka_table([result])
    assert "| 32 | 128 |" in table
    assert f"n={len(INPUTS.questions)}" in table
