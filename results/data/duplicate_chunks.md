# Duplicate chunks in Fine Print's corpus

Of 1,581 section chunks, 60 texts appear more than once, making 61 redundant chunks.

| Where | Chunks |
| --- | --- |
| Lyft terms and Lyft privacy, identical text in both | 43 in each |
| Strava terms and Strava privacy | 5 in each |
| Coursera terms and Coursera privacy | 1 in each |
| Reddit terms, repeated within the document (EEA and US versions) | 9 |
| Pinterest privacy, repeated within the document | 2 |

Identical chunks get identical scores, so their order is decided by tie-breaking. Fine Print's
keyword test also checks the document type, so for two Lyft questions a tie decides whether the
first relevant chunk is at rank 1 or 2.

This project breaks exact ties by corpus position, which favors the privacy copy, in both numpy
and FAISS search. Reversing the tie order changes no model's tuning hit@5. It lowers every
model's tuning MRR by 0.017 to 0.033, through the same two questions, so the comparisons between
models don't change:

| Model | hit@5, ties to earlier / later | MRR, ties to earlier / later |
| --- | --- | --- |
| bge-small-en-v1.5 | 20 / 20 | 0.618 / 0.585 |
| granite-small-en-r2 | 13 / 13 | 0.360 / 0.343 |
| bge-base-en-v1.5 | 23 / 23 | 0.596 / 0.564 |
| arctic-embed-m-v1.5 | 20 / 20 | 0.587 / 0.555 |
| e5-base-v2 | 19 / 19 | 0.587 / 0.555 |
| nomic-embed-text-v1.5 | 18 / 18 | 0.525 / 0.493 |

n=31. Fine Print's own published scores are affected too, since Chroma also has to order these
ties. Removing duplicates would be a change to Fine Print's ingestion, and is a recommendation in
the README rather than something done here, because Fine Print is read-only for this project.
