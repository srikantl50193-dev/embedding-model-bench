"""Encode text with a model, and cache the vectors on disk.

Prefixes are added here rather than through the models' `prompts` settings, so that the prefix
experiment can turn them on and off and the cache key sees exactly what was encoded.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Protocol

import numpy as np

from embedding_model_bench.config import PROJECT_ROOT
from embedding_model_bench.models import ModelSpec

BATCH_SIZE = 32  # Fine Print's embedding_batch_size.
CACHE_FORMAT = 2


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
# Embeds texts with a prefix: the shape every experiment uses, cached or not.
type Embed = Callable[[Sequence[str], str], np.ndarray]
WEIGHT_PATTERNS = ("*.safetensors", "*.bin")


def resolve_repo(repo: str) -> str:
    """Return the folder of a model saved in this project, or a Hugging Face repo id unchanged.

    Fine-tuned models are recorded relative to the project root, so that results and cache keys
    don't contain this machine's paths.
    """
    local = PROJECT_ROOT / repo
    return str(local) if local.is_dir() else repo


class SentenceTransformerEncoder:
    """A real model, loaded from the shared Hugging Face cache."""

    def __init__(self, spec: ModelSpec, device: str) -> None:
        # Imported here because they load torch, which would slow down every CLI command.
        import torch  # noqa: PLC0415
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._spec = spec

        def load(*, local_only: bool) -> SentenceTransformer:
            return SentenceTransformer(
                resolve_repo(spec.repo),
                device=device,
                revision=spec.revision,
                trust_remote_code=spec.trust_remote_code,
                local_files_only=local_only,
                # Some weights are stored in half precision, which would run slower on CPU
                # and make the speed comparison partly about number formats.
                model_kwargs={"dtype": torch.float32},
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
        """Return the size of the weights on disk, summed over every weights file.

        Raises:
            FileNotFoundError: If no weights file is found. A size of 0 would win the selection
                rule's tie-break on size, so it must never be reported.
        """
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        folder = Path(resolve_repo(self._spec.repo))
        if not folder.is_dir():
            folder = Path(
                snapshot_download(
                    self._spec.repo, revision=self._spec.revision, local_files_only=True
                )
            )
        for pattern in WEIGHT_PATTERNS:
            # Prefer safetensors: repos often ship the same weights in both formats.
            files = list(folder.glob(pattern))
            if files:
                return sum(path.stat().st_size for path in files)
        raise FileNotFoundError(f"No weights file found for {self._spec.key} in {folder}")


def cache_key(spec: ModelSpec, prefix: str, texts: Sequence[str]) -> str:
    """Hash everything that decides the vectors: the revision, the prefix and the exact text."""
    digest = hashlib.sha256()
    settings = {
        "format": CACHE_FORMAT,
        "repo": spec.repo,
        "revision": spec.revision,
        "prefix": prefix,
        "normalized": True,
        "dtype": "float32",
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
    partial.replace(path)
    return vectors, False


def cached_embedder(
    spec: ModelSpec,
    cache_dir: Path,
    load: Callable[[], Encoder],
    log: Callable[[str], None] | None = None,
) -> Embed:
    """Return a function that embeds through the cache, loading the model at most once.

    Args:
        spec: The model.
        cache_dir: Where vectors are cached.
        load: Loads the model. Called only on the first cache miss.
        log: If given, told whether each request was cached.
    """
    encoder = cache(load)

    def embed(texts: Sequence[str], prefix: str) -> np.ndarray:
        vectors, hit = embed_cached(spec, texts, prefix, cache_dir, encoder)
        if log:
            log(f"{len(texts)} texts, prefix {prefix!r}: {'from cache' if hit else 'computed'}")
        return vectors

    return embed
