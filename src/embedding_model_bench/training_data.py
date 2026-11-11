"""Turn synthetic questions into training examples: filter, check for leakage, mine negatives.

The thresholds were fixed before any synthetic question was generated, then revised once after
reading samples of the first pass, before any training or scoring (`REVISIONS`). Only training
data and tuning-set pairs were read; held-out questions were compared, never read.

1. Format: a question must end in "?", have 4 to 30 words, name the company and not refer to
   "the passage" or "this section", which a real user couldn't see.
2. Leakage: no eval question may reach training data. A synthetic question is dropped if it
   matches any eval question (tuning, Fine Print's spent held-out set, the fresh held-out set,
   and their out-of-scope questions) exactly, by embedding similarity of `LEAK_COSINE` or more,
   or by sharing `LEAK_JACCARD` or more of its content words. Counts are reported per eval set.
   Held-out questions are only compared, never printed or stored.
3. Round trip: the baseline must rank the source chunk, or an identical copy, in its top
   `ROUNDTRIP_K`. A question it can't trace back is usually vague or off the passage.
4. Hard negatives: the baseline's best-ranked chunk that isn't the source, a copy of it, or a
   chunk overlapping it, and scores under `FALSE_NEGATIVE_RATIO` of the source's score. Close
   seconds are skipped because they often answer the question too.
5. Split: `VALIDATION_SHARE` of source chunks, chosen at random, go to validation, so no chunk
   is in both.
"""

import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from fine_print.models import Chunk

from embedding_model_bench.models import FrozenModel
from embedding_model_bench.synthetic import COMPANY_NAMES, SyntheticRecord, text_sha256

MIN_WORDS, MAX_WORDS = 4, 30
LEAK_COSINE = 0.80
LEAK_JACCARD = 0.5
REPORTED_COSINES = (0.75, 0.80, 0.85, 0.90)
ROUNDTRIP_K = 100
REVISIONS = {
    "leak_cosine": (
        "0.85 -> 0.80: near-copies of tuning questions sat just under 0.85, such as 0.848 for "
        "two questions about whether Discord sells data. Stricter can only remove training data."
    ),
    "roundtrip_k": (
        "20 -> 100: most sampled drops were good questions that the baseline itself ranked "
        "poorly, which are the cases fine-tuning should learn from."
    ),
}
NEGATIVE_DEPTH = 30
FALSE_NEGATIVE_RATIO = 0.95
VALIDATION_SHARE = 0.1
SPLIT_SEED = 20261111
SELF_REFERENCE = re.compile(r"\b(this|the|above) (passage|section|text|clause|paragraph)\b", re.I)
STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    )
)


class TrainingExample(FrozenModel):
    """A question, the chunk it came from, and a hard negative, as the model will see them."""

    query: str
    positive: str
    negative: str | None
    chunk_id: str
    negative_chunk_id: str | None
    company: str
    doc_type: str
    split: str


class LeakageCount(FrozenModel):
    """How many synthetic questions came close to one eval set."""

    eval_set: str
    num_eval_questions: int
    exact: int
    cosine_at: dict[str, int]
    jaccard: int
    dropped: int


class TrainingDataReport(FrozenModel):
    """What each filter removed, so the training data can be audited."""

    num_records: int
    num_questions: int
    dropped_format: dict[str, int]
    leakage: list[LeakageCount]
    dropped_leakage: int
    dropped_roundtrip: int
    num_train: int
    num_validation: int
    num_without_negative: int
    thresholds: dict[str, float]
    revisions: dict[str, str]


def company_name(company: str) -> str:
    """Return the company as a user writes it."""
    return COMPANY_NAMES.get(company, company.title())


def format_problem(question: str, company: str) -> str | None:
    """Return why a question fails the format checks, or None if it passes."""
    words = question.split()
    if not question.endswith("?"):
        return "no question mark"
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return "too short or long"
    if company_name(company).lower() not in question.lower():
        return "doesn't name the company"
    if SELF_REFERENCE.search(question):
        return "refers to the passage"
    return None


def normalize(text: str) -> str:
    """Lowercase and keep only words, for exact matching."""
    return " ".join(re.findall(r"\w+", text.lower()))


def content_words(text: str) -> set[str]:
    """Return the words that carry meaning, without common function words."""
    return {word for word in re.findall(r"\w+", text.lower()) if word not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    """Return the overlap of two word sets, from 0 to 1."""
    return len(a & b) / len(a | b) if a or b else 0.0


def leakage(
    questions: Sequence[str],
    question_vectors: np.ndarray,
    eval_sets: Mapping[str, Sequence[str]],
    eval_vectors: Mapping[str, np.ndarray],
) -> tuple[list[bool], list[LeakageCount]]:
    """Flag synthetic questions too close to any eval question.

    Returns:
        Whether each question leaks, and the counts for each eval set.
    """
    leaks = [False] * len(questions)
    words = [content_words(q) for q in questions]
    counts = []
    for name, eval_questions in eval_sets.items():
        if not eval_questions:
            continue
        exact_set = {normalize(q) for q in eval_questions}
        exact = [normalize(q) in exact_set for q in questions]
        best = (question_vectors @ eval_vectors[name].T).max(axis=1)
        eval_words = [content_words(q) for q in eval_questions]
        overlap = [max(jaccard(w, e) for e in eval_words) >= LEAK_JACCARD for w in words]
        dropped = [e or c >= LEAK_COSINE or j for e, c, j in zip(exact, best, overlap, strict=True)]
        leaks = [a or b for a, b in zip(leaks, dropped, strict=True)]
        counts.append(
            LeakageCount(
                eval_set=name,
                num_eval_questions=len(eval_questions),
                exact=sum(exact),
                cosine_at={f"{t:.2f}": int((best >= t).sum()) for t in REPORTED_COSINES},
                jaccard=sum(overlap),
                dropped=sum(dropped),
            )
        )
    return leaks, counts


def overlaps(a: Chunk, b: Chunk) -> bool:
    """Say whether two chunks share text: same document and overlapping character ranges."""
    same_document = (a.company, a.doc_type) == (b.company, b.doc_type)
    return same_document and a.start_char < b.end_char and b.start_char < a.end_char


def prepare_training_data(
    records: Sequence[SyntheticRecord],
    chunks: Sequence[Chunk],
    chunk_vectors: np.ndarray,
    embed_queries: Callable[[Sequence[str]], np.ndarray],
    eval_sets: Mapping[str, Sequence[str]],
    log: Callable[[str], None] = lambda _message: None,
) -> tuple[list[TrainingExample], TrainingDataReport]:
    """Filter the synthetic questions and build training examples with hard negatives.

    Args:
        records: The generated questions.
        chunks: The whole corpus, in the order of `chunk_vectors`.
        chunk_vectors: The baseline's vectors for every chunk.
        embed_queries: The baseline's query encoder, prefix included.
        eval_sets: Every eval question, by set name. Used only for the leakage check.
        log: Receives progress messages.

    Returns:
        The examples, and a report of what each step removed.
    """
    position = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    copies: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        copies[text_sha256(chunk.text)].append(index)

    candidates: list[tuple[str, SyntheticRecord]] = []
    dropped_format: Counter[str] = Counter()
    for record in records:
        for question in record.questions:
            problem = format_problem(question, record.company)
            if problem:
                dropped_format[problem] += 1
            else:
                candidates.append((question, record))
    log(f"{len(candidates)} questions pass the format checks")

    questions = [question for question, _ in candidates]
    vectors = embed_queries(questions) if questions else np.zeros((0, chunk_vectors.shape[1]))
    eval_vectors = {name: embed_queries(list(qs)) for name, qs in eval_sets.items() if qs}
    leaks, leak_counts = leakage(questions, vectors, eval_sets, eval_vectors)

    validation_chunks = _validation_chunks(records)
    scores_all = vectors @ chunk_vectors.T
    examples = []
    dropped_roundtrip = 0
    for row, ((question, record), leak) in enumerate(zip(candidates, leaks, strict=True)):
        if leak:
            continue
        scores = scores_all[row]
        ranked = np.argsort(-scores, kind="stable")[: max(ROUNDTRIP_K, NEGATIVE_DEPTH)]
        source = chunks[position[record.chunk_id]]
        positives = set(copies[record.text_sha256])
        if not positives & set(ranked[:ROUNDTRIP_K].tolist()):
            dropped_roundtrip += 1
            continue
        ceiling = FALSE_NEGATIVE_RATIO * scores[position[record.chunk_id]]
        negative = next(
            (
                chunks[int(i)]
                for i in ranked[1:NEGATIVE_DEPTH]
                if int(i) not in positives
                and not overlaps(chunks[int(i)], source)
                and scores[i] < ceiling
            ),
            None,
        )
        examples.append(
            TrainingExample(
                query=question,
                positive=source.embedding_text(),
                negative=negative.embedding_text() if negative else None,
                chunk_id=source.chunk_id,
                negative_chunk_id=negative.chunk_id if negative else None,
                company=record.company,
                doc_type=record.doc_type,
                split="validation" if record.chunk_id in validation_chunks else "train",
            )
        )

    report = TrainingDataReport(
        num_records=len(records),
        num_questions=sum(len(r.questions) for r in records),
        dropped_format=dict(dropped_format),
        leakage=leak_counts,
        dropped_leakage=sum(leaks),
        dropped_roundtrip=dropped_roundtrip,
        num_train=sum(e.split == "train" for e in examples),
        num_validation=sum(e.split == "validation" for e in examples),
        num_without_negative=sum(e.negative is None for e in examples),
        thresholds={
            "leak_cosine": LEAK_COSINE,
            "leak_jaccard": LEAK_JACCARD,
            "roundtrip_k": ROUNDTRIP_K,
            "negative_depth": NEGATIVE_DEPTH,
            "false_negative_ratio": FALSE_NEGATIVE_RATIO,
            "validation_share": VALIDATION_SHARE,
        },
        revisions=REVISIONS,
    )
    return examples, report


def _validation_chunks(records: Sequence[SyntheticRecord]) -> set[str]:
    """Pick the source chunks whose questions go to validation."""
    ids = sorted({record.chunk_id for record in records})
    count = round(len(ids) * VALIDATION_SHARE)
    return set(random.Random(SPLIT_SEED).sample(ids, count))
