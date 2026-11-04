from pathlib import Path

import numpy as np

from conftest_fakes import FakeEncoder
from embedding_model_bench.embedding import cache_key, embed_cached
from embedding_model_bench.models import ModelSpec

SPEC = ModelSpec(key="fake", repo="org/fake", revision="abc", dim=64, query_prefix="q: ")


def test_the_second_call_comes_from_the_cache_without_loading_a_model(tmp_path: Path) -> None:
    fake = FakeEncoder()
    first, first_hit = embed_cached(SPEC, ["hello world"], "q: ", tmp_path, lambda: fake)
    assert not first_hit

    def must_not_load() -> FakeEncoder:
        raise AssertionError("the model should not be loaded")

    second, second_hit = embed_cached(SPEC, ["hello world"], "q: ", tmp_path, must_not_load)
    assert second_hit
    assert np.array_equal(first, second)
    assert fake.calls == 1


def test_the_prefix_is_applied_to_what_gets_encoded(tmp_path: Path) -> None:
    with_prefix, _ = embed_cached(SPEC, ["hello"], "unusual ", tmp_path, FakeEncoder)
    assert np.array_equal(with_prefix, FakeEncoder().encode(["unusual hello"]))


def test_the_key_changes_with_revision_prefix_and_text() -> None:
    base = cache_key(SPEC, "q: ", ["a"])
    other_revision = SPEC.model_copy(update={"revision": "def"})
    assert cache_key(other_revision, "q: ", ["a"]) != base
    assert cache_key(SPEC, "", ["a"]) != base
    assert cache_key(SPEC, "q: ", ["b"]) != base


def test_text_boundaries_are_part_of_the_key() -> None:
    assert cache_key(SPEC, "", ["ab", "c"]) != cache_key(SPEC, "", ["a", "bc"])


def test_an_interrupted_write_leaves_nothing_that_looks_cached(tmp_path: Path) -> None:
    embed_cached(SPEC, ["hello"], "", tmp_path, FakeEncoder)
    assert not list(tmp_path.glob("*.partial.npy"))
