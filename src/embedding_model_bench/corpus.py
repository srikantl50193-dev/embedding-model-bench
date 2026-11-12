"""Load and chunk the corpus with Fine Print's own code, so results stay comparable.

Only `fine_print.chunking`, `.ingest`, `.models` and `.config` are imported. Unlike
`fine_print.evaluate` and `.store`, they don't load torch or chromadb.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

from fine_print.chunking import chunk_document
from fine_print.config import Settings as FinePrintSettings
from fine_print.ingest import load_documents
from fine_print.models import Chunk

STRATEGY = "section"


def chunk_settings() -> tuple[int, int]:
    """Return Fine Print's default chunk size and overlap.

    These come from the field defaults rather than an instance, so a FINE_PRINT_* environment
    variable or .env file can't quietly change what is being compared.
    """
    fields = FinePrintSettings.model_fields
    size, overlap = fields["chunk_size"].default, fields["chunk_overlap"].default
    if not isinstance(size, int) or not isinstance(overlap, int):
        raise TypeError("Fine Print's chunk_size and chunk_overlap defaults should be integers")
    return size, overlap


def load_chunks(raw_dir: Path) -> tuple[int, list[Chunk]]:
    """Chunk every document in a folder.

    Args:
        raw_dir: The folder of policy files.

    Returns:
        The number of documents and all of their chunks, in a stable order.
    """
    size, overlap = chunk_settings()
    documents = load_documents(raw_dir)
    chunks = [c for d in documents for c in chunk_document(d, STRATEGY, size, overlap)]
    return len(documents), chunks


def corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Hash exactly what gets embedded, so a cached embedding can be tied to its text."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode())
        digest.update(b"\0")
        digest.update(chunk.embedding_text().encode())
        digest.update(b"\0")
    return digest.hexdigest()
