"""Generate synthetic search questions for each chunk with a local Ollama model.

The run is resumable: each chunk's questions are appended to a JSONL file and synced to disk as
soon as they are made, and a rerun skips chunks already done. Records are tied to the prompt
version and the model's digest, so changing either starts over in a new file rather than mixing
outputs. Filtering, leakage checks and hard negatives come later, in a separate step, because
they need an embedding model and must never run alongside Ollama.
"""

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
from fine_print.models import Chunk
from pydantic import BaseModel, ValidationError

from embedding_model_bench.files import read_jsonl
from embedding_model_bench.models import FrozenModel

PROMPT_VERSION = 1
QUESTIONS_PER_CHUNK = 2
TEMPERATURE = 0.7
ORDER_SEED = 20261111
MAX_ATTEMPTS = 2
MAX_CONSECUTIVE_FAILURES = 5
DOC_LABELS = {"terms": "terms of service", "privacy": "privacy policy"}
# Names as users write them, where capitalizing the folder name gets them wrong.
COMPANY_NAMES = {"paypal": "PayPal"}

PROMPT = """You are helping build search training data for a tool that answers questions \
about {company}'s {doc_label}.

Passage, from the section "{section}":
\"\"\"
{text}
\"\"\"

Write {count} different questions that a regular {company} user might type into a search box, \
where this passage contains the answer.
- Use plain everyday words, the way someone who isn't a lawyer would ask. Don't reuse \
distinctive phrases from the passage.
- Name {company} in each question.
- Each question asks about something different, in under 20 words.
- If the passage has nothing a user would ask about, such as only a heading, a list of links \
or an address, return an empty list."""

SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}, "maxItems": QUESTIONS_PER_CHUNK}
    },
    "required": ["questions"],
}


class Generated(FrozenModel):
    """What the model returned for one chunk."""

    questions: list[str]
    seconds: float
    output_tokens: int


class SyntheticRecord(FrozenModel):
    """One chunk's questions, with what is needed to reproduce and trust them."""

    chunk_id: str
    company: str
    doc_type: str
    text_sha256: str
    questions: list[str]
    model: str
    model_digest: str
    prompt_version: int
    seconds: float
    output_tokens: int


class _Reply(BaseModel):
    questions: list[str]


type Generate = Callable[[Chunk], Generated]


def text_sha256(text: str) -> str:
    """Hash a chunk's text, ignoring differences in whitespace."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def unique_chunks(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Keep the first chunk of each distinct text, in a fixed random order.

    Fine Print's corpus repeats some passages across documents, and there is no point asking for
    the same passage twice. The random order means that a partial run is a fair sample.
    """
    seen: set[str] = set()
    kept = []
    for chunk in chunks:
        digest = text_sha256(chunk.text)
        if digest not in seen:
            seen.add(digest)
            kept.append(chunk)
    random.Random(ORDER_SEED).shuffle(kept)
    return kept


def build_prompt(chunk: Chunk) -> str:
    """Write the request for one chunk."""
    return PROMPT.format(
        company=COMPANY_NAMES.get(chunk.company, chunk.company.title()),
        doc_label=DOC_LABELS.get(chunk.doc_type, chunk.doc_type),
        section=chunk.section,
        text=chunk.text.strip(),
        count=QUESTIONS_PER_CHUNK,
    )


def chunk_seed(chunk: Chunk) -> int:
    """Return a sampling seed fixed by the chunk, so a rerun asks the same way."""
    return int(hashlib.sha256(chunk.chunk_id.encode()).hexdigest()[:8], 16)


def clean_questions(raw: Sequence[str]) -> list[str]:
    """Strip whitespace, and drop empty and repeated questions."""
    kept: list[str] = []
    for question in raw:
        text = " ".join(question.split())
        if text and text.lower() not in {q.lower() for q in kept}:
            kept.append(text)
    return kept


class OllamaGenerator:
    """Asks a local Ollama model for questions, with thinking turned off for speed."""

    def __init__(self, url: str, model: str, timeout: float, client: httpx.Client) -> None:
        self.url, self.model, self._client = url, model, client
        self._timeout = timeout
        tags = client.get(f"{url}/api/tags", timeout=timeout).raise_for_status().json()
        digests = {m["name"]: m["digest"] for m in tags.get("models", [])}
        if model not in digests:
            raise ValueError(f"{model} isn't pulled in Ollama")
        self.digest = digests[model]

    def __call__(self, chunk: Chunk) -> Generated:
        """Generate questions for one chunk."""
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": SCHEMA,
            "options": {"temperature": TEMPERATURE, "seed": chunk_seed(chunk)},
            "messages": [{"role": "user", "content": build_prompt(chunk)}],
        }
        start = time.perf_counter()
        response = self._client.post(f"{self.url}/api/chat", json=body, timeout=self._timeout)
        reply = response.raise_for_status().json()
        questions = _Reply.model_validate_json(reply["message"]["content"]).questions
        return Generated(
            questions=clean_questions(questions)[:QUESTIONS_PER_CHUNK],
            seconds=round(time.perf_counter() - start, 3),
            output_tokens=int(reply.get("eval_count", 0)),
        )

    def unload(self) -> None:
        """Free the model's memory now rather than when Ollama's keep-alive runs out."""
        self._client.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "keep_alive": 0},
            timeout=self._timeout,
        )


def output_path(data_dir: Path, model_digest: str) -> Path:
    """Return the file for this prompt version and model."""
    return data_dir / "synthetic" / f"questions-v{PROMPT_VERSION}-{model_digest[:12]}.jsonl"


def repair_tail(path: Path) -> bool:
    """Drop a partly written last line, left if a run was killed mid-write.

    Returns:
        Whether anything was removed.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return False
    with path.open("r+b") as file:
        file.truncate(data.rfind(b"\n") + 1)
    return True


def load_records(path: Path) -> list[SyntheticRecord]:
    """Read the records written so far."""
    return read_jsonl(path, SyntheticRecord) if path.exists() else []


def _append(path: Path, record: SyntheticRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(record.model_dump_json() + "\n")
        file.flush()
        os.fsync(file.fileno())


class RunSummary(FrozenModel):
    """What one invocation did, and how long the rest should take."""

    total_chunks: int
    done_before: int
    generated: int
    failed: list[str]
    remaining: int
    mean_seconds: float
    estimated_remaining_hours: float


def run_generation(
    chunks: Sequence[Chunk],
    path: Path,
    generate: Generate,
    model: str,
    model_digest: str,
    limit: int | None = None,
    log: Callable[[str], None] = lambda _message: None,
) -> RunSummary:
    """Generate questions for every unique chunk not already in `path`.

    Args:
        chunks: The whole corpus. Repeated passages are asked about once.
        path: The JSONL file to append to.
        generate: Makes one chunk's questions.
        model: The model's name, for the records.
        model_digest: The model's exact digest, for the records.
        limit: Stop after this many new chunks. None runs to the end.
        log: Receives progress messages.

    Returns:
        A summary, with an estimate for the chunks left.

    Raises:
        RuntimeError: If too many chunks fail in a row, which usually means Ollama stopped.
    """
    if repair_tail(path):
        log("removed a partly written last line from an interrupted run")
    todo = unique_chunks(chunks)
    done = {record.chunk_id for record in load_records(path)}
    pending = [chunk for chunk in todo if chunk.chunk_id not in done]
    batch = pending if limit is None else pending[:limit]
    times: list[float] = []
    failed: list[str] = []
    consecutive = 0
    for number, chunk in enumerate(batch, start=1):
        result = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = generate(chunk)
                break
            except (httpx.HTTPError, ValidationError, json.JSONDecodeError, KeyError) as error:
                log(f"{chunk.chunk_id}: attempt {attempt} failed: {error!r}")
        if result is None:
            failed.append(chunk.chunk_id)
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"{consecutive} chunks failed in a row; is Ollama running?")
            continue
        consecutive = 0
        _append(
            path,
            SyntheticRecord(
                chunk_id=chunk.chunk_id,
                company=chunk.company,
                doc_type=chunk.doc_type,
                text_sha256=text_sha256(chunk.text),
                questions=result.questions,
                model=model,
                model_digest=model_digest,
                prompt_version=PROMPT_VERSION,
                seconds=result.seconds,
                output_tokens=result.output_tokens,
            ),
        )
        times.append(result.seconds)
        if number % 25 == 0 or number == len(batch):
            log(f"{len(done) + number}/{len(todo)} chunks, {result.seconds:.1f}s for the last")

    remaining = len(pending) - len(times)
    # The first request includes loading the model, so it is left out of the average.
    steady = times[1:] or times
    mean = sum(steady) / len(steady) if steady else 0.0
    return RunSummary(
        total_chunks=len(todo),
        done_before=len(done),
        generated=len(times),
        failed=failed,
        remaining=remaining,
        mean_seconds=round(mean, 2),
        estimated_remaining_hours=round(mean * remaining / 3600, 2),
    )
