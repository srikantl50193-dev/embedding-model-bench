from pathlib import Path

import pytest

POLICY = """# Refunds

We do not provide refunds for partial periods.

# Data retention

We keep personal information only as long as necessary.
"""


@pytest.fixture
def fine_print_tree(tmp_path: Path) -> Path:
    """A tiny stand-in for the Fine Print repo, with fake documents and questions."""
    root = tmp_path / "fine-print"
    (root / "data" / "raw").mkdir(parents=True)
    (root / "eval" / "holdout").mkdir(parents=True)
    (root / "data" / "raw" / "acme_terms.md").write_text(POLICY, encoding="utf-8")
    question = (
        '{"question": "Refunds?", "company": "acme", "doc_type": "terms", '
        '"keywords": ["do not provide refunds"]}\n'
    )
    (root / "eval" / "questions.jsonl").write_text(question * 2, encoding="utf-8")
    (root / "eval" / "out_of_scope.jsonl").write_text('{"question": "Pizza?"}\n', encoding="utf-8")
    (root / "eval" / "holdout" / "questions.jsonl").write_text(question, encoding="utf-8")
    (root / "eval" / "holdout" / "out_of_scope.jsonl").write_text("", encoding="utf-8")
    return root
