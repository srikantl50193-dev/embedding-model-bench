import json
from pathlib import Path

import httpx
import pytest
from fine_print.models import Chunk

from conftest_fakes import CHUNKS, make_chunk
from embedding_model_bench.synthetic import (
    QUESTIONS_PER_CHUNK,
    Generated,
    OllamaGenerator,
    build_prompt,
    clean_questions,
    load_records,
    repair_tail,
    run_generation,
    unique_chunks,
)

DIGEST = "sha256abcdef0123456789"


def fake_generate(chunk: Chunk) -> Generated:
    return Generated(questions=[f"What about {chunk.chunk_id}?"], seconds=1.0, output_tokens=9)


def run(path: Path, limit: int | None = None, generate=fake_generate):
    return run_generation(CHUNKS, path, generate, "fake", DIGEST, limit=limit)


def test_repeated_passages_are_asked_about_once_in_a_fixed_order() -> None:
    chunks = [*CHUNKS, make_chunk("copy", "  we do not provide\nrefunds ")]
    first, second = unique_chunks(chunks), unique_chunks(chunks)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len(first) == len(CHUNKS)
    assert "copy" not in {c.chunk_id for c in first}


def test_the_prompt_names_the_company_document_and_section() -> None:
    prompt = build_prompt(make_chunk("a", "we do not provide refunds", section="Refunds"))
    assert "Acme's terms of service" in prompt
    assert '"Refunds"' in prompt
    assert "we do not provide refunds" in prompt
    assert f"Write {QUESTIONS_PER_CHUNK} different questions" in prompt


def test_questions_are_tidied_and_deduplicated() -> None:
    assert clean_questions(["  Can I  get a refund? ", "", "can i get a refund?", "Why?"]) == [
        "Can I get a refund?",
        "Why?",
    ]


def test_a_second_run_resumes_where_the_first_stopped(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    first = run(path, limit=3)
    assert (first.generated, first.done_before) == (3, 0)
    second = run(path)
    assert (second.done_before, second.generated, second.remaining) == (3, len(CHUNKS) - 3, 0)
    ids = [record.chunk_id for record in load_records(path)]
    assert sorted(ids) == sorted(c.chunk_id for c in CHUNKS)


def test_the_estimate_leaves_out_the_first_request_which_loads_the_model(tmp_path: Path) -> None:
    times = iter([30.0, 2.0, 4.0])

    def timed(_chunk: Chunk) -> Generated:
        return Generated(questions=["Q?"], seconds=next(times), output_tokens=1)

    summary = run(tmp_path / "q.jsonl", limit=3, generate=timed)
    assert summary.mean_seconds == 3.0
    assert summary.estimated_remaining_hours == round(3.0 * (len(CHUNKS) - 3) / 3600, 2)


def test_a_partly_written_last_line_is_removed(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    run(path, limit=2)
    with path.open("a", encoding="utf-8") as file:
        file.write('{"chunk_id": "half')
    assert repair_tail(path)
    assert len(load_records(path)) == 2
    assert run(path, limit=1).generated == 1


def test_a_failing_chunk_is_skipped_and_retried_next_time(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    target = unique_chunks(CHUNKS)[1].chunk_id

    def flaky(chunk: Chunk) -> Generated:
        if chunk.chunk_id == target:
            raise httpx.ReadTimeout("slow")
        return fake_generate(chunk)

    summary = run(path, limit=3, generate=flaky)
    assert summary.failed == [target]
    assert target not in {r.chunk_id for r in load_records(path)}
    assert run(path, limit=1).generated == 1
    assert target in {r.chunk_id for r in load_records(path)}


def test_many_failures_in_a_row_stop_the_run(tmp_path: Path) -> None:
    def broken(_chunk: Chunk) -> Generated:
        raise httpx.ConnectError("refused")

    with pytest.raises(RuntimeError, match="Ollama"):
        run(tmp_path / "q.jsonl", generate=broken)


def ollama_transport(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "m:1", "digest": DIGEST}]})
    body = json.loads(request.content)
    assert body["think"] is False
    assert body["format"]["required"] == ["questions"]
    content = json.dumps({"questions": [" Does Acme refund? ", "does acme refund?"]})
    return httpx.Response(200, json={"message": {"content": content}, "eval_count": 12})


def test_the_ollama_client_records_the_digest_and_cleans_replies() -> None:
    client = httpx.Client(transport=httpx.MockTransport(ollama_transport))
    generator = OllamaGenerator("http://ollama", "m:1", 5.0, client)
    assert generator.digest == DIGEST
    result = generator(CHUNKS[0])
    assert result.questions == ["Does Acme refund?"]
    assert result.output_tokens == 12


def test_a_model_that_isnt_pulled_is_rejected() -> None:
    client = httpx.Client(transport=httpx.MockTransport(ollama_transport))
    with pytest.raises(ValueError, match="isn't pulled"):
        OllamaGenerator("http://ollama", "other:1", 5.0, client)


def test_company_names_are_written_as_users_write_them() -> None:
    chunk = make_chunk("p", "fees apply").model_copy(update={"company": "paypal"})
    assert "PayPal's terms of service" in build_prompt(chunk)
