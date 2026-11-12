"""The embedding models under test, each pinned to an exact revision.

Prefixes come from each model's card. Passages take no prefix unless the card asks for one.
EmbeddingGemma was planned too, but it is gated and needs a Hugging Face login.
"""

from embedding_model_bench.models import ModelSpec

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BGE_SMALL = ModelSpec(
    key="bge-small-en-v1.5",
    repo="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dim=384,
    query_prefix=BGE_QUERY_PREFIX,
)
GRANITE_SMALL = ModelSpec(
    key="granite-small-en-r2",
    repo="ibm-granite/granite-embedding-small-english-r2",
    revision="2ab6fa8ea2d674564defd37171ae19079b864b33",
    dim=384,
)
BGE_BASE = ModelSpec(
    key="bge-base-en-v1.5",
    repo="BAAI/bge-base-en-v1.5",
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    dim=768,
    query_prefix=BGE_QUERY_PREFIX,
)
ARCTIC_M = ModelSpec(
    key="arctic-embed-m-v1.5",
    repo="Snowflake/snowflake-arctic-embed-m-v1.5",
    revision="e58a8f756156a1293d763f17e3aae643474e9b8a",
    dim=768,
    query_prefix=BGE_QUERY_PREFIX,
)
E5_BASE = ModelSpec(
    key="e5-base-v2",
    repo="intfloat/e5-base-v2",
    revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
    dim=768,
    query_prefix="query: ",
    passage_prefix="passage: ",
)
# trust_remote_code stays off: the card says transformers 5.5+ and sentence-transformers 5.3+
# support this model natively. With it on, the code would come from another, unpinned repo.
NOMIC = ModelSpec(
    key="nomic-embed-text-v1.5",
    repo="nomic-ai/nomic-embed-text-v1.5",
    revision="e9b6763023c676ca8431644204f50c2b100d9aab",
    dim=768,
    query_prefix="search_query: ",
    passage_prefix="search_document: ",
)

MODELS = {spec.key: spec for spec in (BGE_SMALL, GRANITE_SMALL, BGE_BASE, ARCTIC_M, E5_BASE, NOMIC)}
