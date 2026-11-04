"""The embedding models under test, each pinned to an exact revision."""

from embedding_model_bench.models import ModelSpec

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BGE_SMALL = ModelSpec(
    key="bge-small-en-v1.5",
    repo="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dim=384,
    query_prefix=BGE_QUERY_PREFIX,
)

MODELS = {spec.key: spec for spec in (BGE_SMALL,)}
