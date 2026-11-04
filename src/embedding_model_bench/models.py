"""Domain models shared across the project."""

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    """A model whose instances can't change after they are created."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalQuestion(FrozenModel):
    """An answerable question, in the format of Fine Print's question files."""

    question: str = Field(min_length=1)
    company: str = Field(min_length=1)
    doc_type: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)


class DataManifest(FrozenModel):
    """What the symlinked Fine Print data was when it was linked, to detect later changes."""

    fine_print_commit: str
    fine_print_dirty: bool
    file_sha256: dict[str, str]
    tree_digest: str
    num_documents: int
    num_chunks: int
    corpus_fingerprint: str
    num_tuning_questions: int
    num_holdout_questions: int


class SciFact(FrozenModel):
    """The SciFact test split: documents and queries by ID, and the relevant documents."""

    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
