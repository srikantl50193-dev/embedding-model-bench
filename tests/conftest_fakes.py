import hashlib
import re
from collections.abc import Sequence

import numpy as np

DIM = 64


class FakeEncoder:
    """A bag-of-words hash: texts that share words get similar vectors, with no model involved."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def num_parameters(self) -> int:
        return 1_000_000

    def weights_bytes(self) -> int:
        return 4 * 1024**2
