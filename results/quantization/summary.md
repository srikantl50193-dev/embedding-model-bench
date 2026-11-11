| Model | Dims | Scheme | Memory B/vector | Tuning hit@5 | Wins / losses | MRR | SciFact nDCG@10 | SciFact wins / losses | p | Search p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-base-en-v1.5 | 768 | float32 | 3,072 | 23/31 | 0 / 0 | 0.596 | 0.740 | 0 / 0 | 1.000 | 6.32 |
| bge-base-en-v1.5 | 768 | int8 | 768 | 23/31 | 0 / 0 | 0.596 | 0.741 | 0 / 0 | 1.000 | 12.96 |
| bge-base-en-v1.5 | 768 | binary | 96 | 21/31 | 0 / 2 | 0.527 | 0.672 | 2 / 28 | 0.000 | 0.39 |
| bge-base-en-v1.5 | 768 | binary, rescored with float32 query | 96 | 22/31 | 0 / 1 | 0.563 | 0.723 | 5 / 13 | 0.096 | 0.43 |
| bge-base-en-v1.5 | 768 | binary, rescored with float32 vectors | 96 | 23/31 | 0 / 0 | 0.596 | 0.732 | 2 / 3 | 1.000 | 0.42 |
| bge-base-en-v1.5 | 256 | float32 | 1,024 | 22/31 | 0 / 0 | 0.539 | 0.695 | 0 / 0 | 1.000 | 2.08 |
| bge-base-en-v1.5 | 256 | int8 | 256 | 22/31 | 0 / 0 | 0.539 | 0.695 | 0 / 0 | 1.000 | 4.51 |
| bge-base-en-v1.5 | 256 | binary | 32 | 16/31 | 1 / 7 | 0.352 | 0.509 | 4 / 57 | 0.000 | 0.23 |
| bge-base-en-v1.5 | 256 | binary, rescored with float32 query | 32 | 19/31 | 0 / 3 | 0.455 | 0.581 | 4 / 34 | 0.000 | 0.26 |
| bge-base-en-v1.5 | 256 | binary, rescored with float32 vectors | 32 | 21/31 | 0 / 1 | 0.524 | 0.643 | 3 / 17 | 0.003 | 0.26 |
| bge-small-en-v1.5 | 384 | float32 | 1,536 | 20/31 | 0 / 0 | 0.618 | 0.713 | 0 / 0 | 1.000 | 3.19 |
| bge-small-en-v1.5 | 384 | int8 | 384 | 20/31 | 0 / 0 | 0.618 | 0.713 | 0 / 0 | 1.000 | 6.69 |
| bge-small-en-v1.5 | 384 | binary | 48 | 19/31 | 1 / 2 | 0.469 | 0.626 | 11 / 35 | 0.001 | 0.32 |
| bge-small-en-v1.5 | 384 | binary, rescored with float32 query | 48 | 19/31 | 0 / 1 | 0.512 | 0.659 | 9 / 23 | 0.020 | 0.36 |
| bge-small-en-v1.5 | 384 | binary, rescored with float32 vectors | 48 | 20/31 | 0 / 0 | 0.597 | 0.692 | 4 / 10 | 0.180 | 0.35 |
| arctic-embed-m-v1.5 | 768 | float32 | 3,072 | 20/31 | 0 / 0 | 0.587 | 0.716 | 0 / 0 | 1.000 | 6.21 |
| arctic-embed-m-v1.5 | 768 | int8 | 768 | 20/31 | 0 / 0 | 0.587 | 0.716 | 0 / 0 | 1.000 | 13.01 |
| arctic-embed-m-v1.5 | 768 | binary | 96 | 23/31 | 3 / 0 | 0.610 | 0.649 | 3 / 25 | 0.000 | 0.38 |
| arctic-embed-m-v1.5 | 768 | binary, rescored with float32 query | 96 | 20/31 | 1 / 1 | 0.576 | 0.679 | 3 / 18 | 0.002 | 0.46 |
| arctic-embed-m-v1.5 | 768 | binary, rescored with float32 vectors | 96 | 20/31 | 0 / 0 | 0.588 | 0.707 | 1 / 4 | 0.375 | 0.42 |
| arctic-embed-m-v1.5 | 256 | float32 | 1,024 | 20/31 | 0 / 0 | 0.611 | 0.698 | 0 / 0 | 1.000 | 2.20 |
| arctic-embed-m-v1.5 | 256 | int8 | 256 | 20/31 | 0 / 0 | 0.611 | 0.697 | 0 / 0 | 1.000 | 4.54 |
| arctic-embed-m-v1.5 | 256 | binary | 32 | 20/31 | 2 / 2 | 0.525 | 0.573 | 7 / 40 | 0.000 | 0.24 |
| arctic-embed-m-v1.5 | 256 | binary, rescored with float32 query | 32 | 20/31 | 2 / 2 | 0.559 | 0.635 | 7 / 24 | 0.003 | 0.26 |
| arctic-embed-m-v1.5 | 256 | binary, rescored with float32 vectors | 32 | 21/31 | 1 / 0 | 0.612 | 0.668 | 4 / 15 | 0.019 | 0.26 |

n=31: one question moves a score by 3.2 points, so gaps up to 6.5 points are likely noise.
Search times: one query at a time over 100,000 vectors, 1 thread, exact FAISS indexes, 300 SciFact queries. They exclude encoding the query.
