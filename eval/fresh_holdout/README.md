# Fresh held-out set

30 questions about Fine Print's corpus, written for this project because Fine Print's own `eval/holdout/` is already spent. They were frozen before any benchmark ran here.

- **Who wrote them:** Claude, not an independent human. They aren't reviewed by the project's owner.
- **How the passages were chosen:** `scripts/sample_holdout_chunks.py` draws two chunks per document with a fixed seed, skipping short chunks and any that answer a tuning question. 30 questions were written from the 40 chunks drawn, at least one per document. The other 10 were boilerplate, table fragments, or would have over-weighted one document.
- **Checked before freezing:** every keyword occurs in its source chunk and in no question's own wording, and no question closely matches a tuning question or an old held-out one. That last check printed only scores, so the old questions stayed unread.
- **Relevance:** a chunk counts if it is from the right document and contains every keyword, the same test Fine Print uses. It inherits that test's weaknesses.

`questions.jsonl` is in Fine Print's format, and its hash is checked by `tests/test_fresh_holdout.py`. `sources.jsonl` records the chunk each question was written from. **Don't edit the questions after seeing any score from them.**
