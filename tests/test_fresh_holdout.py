import hashlib

from embedding_model_bench.config import PROJECT_ROOT, get_settings
from embedding_model_bench.models import EvalQuestion

QUESTIONS = PROJECT_ROOT / "eval" / "fresh_holdout" / "questions.jsonl"


def test_the_frozen_set_has_not_changed() -> None:
    digest = hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()
    assert digest == get_settings().fresh_holdout_sha256


def test_the_set_is_valid_and_covers_every_document() -> None:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    questions = [EvalQuestion.model_validate_json(line) for line in lines]
    assert len(questions) == 30
    assert len({(q.company, q.doc_type) for q in questions}) == 20
    assert len({q.question for q in questions}) == 30
