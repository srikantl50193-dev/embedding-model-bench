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


class ModelSpec(FrozenModel):
    """An embedding model, the exact revision under test, and how it expects its input."""

    key: str
    repo: str
    revision: str
    dim: int = Field(gt=0)
    query_prefix: str = ""
    passage_prefix: str = ""
    trust_remote_code: bool = False


class RetrievalScores(FrozenModel):
    """Quality over one question set, with per-question ranks for paired comparisons.

    `hit_at_k` is the share of queries with at least one relevant result in the top k, which
    is what Fine Print calls recall@k. `recall_at_k` is the standard BEIR measure: the share of
    each query's relevant documents found in the top k, averaged over the queries.
    """

    num_queries: int
    num_without_relevant: int
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_10: float
    first_relevant_ranks: list[int | None]

    def hit_flags(self, k: int) -> list[bool]:
        """Return, for each query, whether a relevant result was in the top k."""
        return [rank is not None and rank <= k for rank in self.first_relevant_ranks]

    def hits(self, k: int) -> int:
        """Return how many queries had a relevant result in the top k."""
        return sum(self.hit_flags(k))


class Timing(FrozenModel):
    """Speed on one device."""

    device: str
    encode_chunks_per_s: float
    query_p50_ms: float
    query_p95_ms: float
    num_query_samples: int


class Resources(FrozenModel):
    """What a model costs to hold and run. MPS memory is separate from the process's RSS."""

    parameters: int
    weights_mb: float
    peak_rss_mb: float
    mps_allocated_mb: float


class MachineState(FrozenModel):
    """How busy the machine was, which decides whether timings can be trusted."""

    available_gb: float
    swap_used_gb: float
    competing_apps: list[str]
    timings_provisional: bool


class BenchmarkResult(FrozenModel):
    """One model measured on one corpus, with what is needed to reproduce it."""

    spec: ModelSpec
    fine_print_commit: str
    corpus_fingerprint: str
    num_chunks: int
    tuning: RetrievalScores
    scifact: RetrievalScores
    timings: list[Timing]
    resources: Resources
    machine: MachineState
