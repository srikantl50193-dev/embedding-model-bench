"""Encode text with a model, and cache the vectors on disk.

Prefixes are added here rather than through the models' `prompts` settings, so that the prefix
experiment can turn them on and off and the cache key sees exactly what was encoded.
"""

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from embedding_model_bench.models import ModelSpec

BATCH_SIZE = 32  # Fine Print's embedding_batch_size.
CACHE_FORMAT = 1


class Encoder(Protocol):
    """Turns text into length-1 vectors. Tests supply a tiny fake."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...

    def num_parameters(self) -> int:
        """Return the model's parameter count."""
        ...

    def weights_bytes(self) -> int:
        """Return the size of the model's weights on disk."""
        ...


type EncoderFactory = Callable[[ModelSpec, str], Encoder]


class SentenceTransformerEncoder:
    """A real model, loaded from the shared Hugging Face cache."""

    def __init__(self, spec: ModelSpec, device: str) -> None:
        # Imported here because it loads torch, which would slow down every CLI command.
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._spec = spec

        def load(*, local_only: bool) -> SentenceTransformer:
            return SentenceTransformer(
                spec.repo,
                device=device,
                revision=spec.revision,
                trust_remote_code=spec.trust_remote_code,
                local_files_only=local_only,
            )

        try:
            self._model = load(local_only=True)
        except OSError:
            self._model = load(local_only=False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized vectors, so that a dot product equals cosine similarity."""
        vectors = self._model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def num_parameters(self) -> int:
        """Return the model's parameter count."""
        return sum(parameter.numel() for parameter in self._model.parameters())

    def weights_bytes(self) -> int:
        """Return the size of the cached weights file, or 0 if it can't be found."""
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415

        path = try_to_load_from_cache(
            self._spec.repo, "model.safetensors", revision=self._spec.revision
        )
        return Path(path).stat().st_size if isinstance(path, str) else 0


def cache_key(spec: ModelSpec, prefix: str, texts: Sequence[str]) -> str:
    """Hash everything that decides the vectors: the revision, the prefix and the exact text."""
    digest = hashlib.sha256()
    settings = {
        "format": CACHE_FORMAT,
        "repo": spec.repo,
        "revision": spec.revision,
        "prefix": prefix,
        "normalized": True,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def embed_cached(
    spec: ModelSpec,
    texts: Sequence[str],
    prefix: str,
    cache_dir: Path,
    encoder: Callable[[], Encoder],
) -> tuple[np.ndarray, bool]:
    """Return the vectors for some texts, computing them only if they aren't cached.

    Args:
        spec: The model.
        texts: The texts, without their prefix.
        prefix: What to put in front of each text.
        cache_dir: Where vectors are stored.
        encoder: Called only when the vectors have to be computed.

    Returns:
        The vectors, and whether they came from the cache.
    """
    key = cache_key(spec, prefix, texts)
    path = cache_dir / f"{spec.key}-{key[:24]}.npy"
    if path.exists():
        return np.load(path), True
    vectors = encoder().encode([prefix + text for text in texts])
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Written then renamed, so that an interrupted run can't leave a partial file in the cache.
    partial = path.with_suffix(".partial.npy")
    np.save(partial, vectors)
    os.replace(partial, path)
    return vectors, False
