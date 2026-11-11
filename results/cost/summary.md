| Configuration | Chunks | RAM MB | Disk MB | FAISS index MB | Search p50 ms | Search p95 ms | Query p95 ms (MPS / CPU), projected | Embed hours (MPS / CPU), projected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-small-en-v1.5 384d float32 | 100,000 | 154 | 154 | 154 | 3.04 | 3.26 | 12.1 / 14.8 | 0.2 / 0.4 |
| bge-small-en-v1.5 384d float32 | 1,000,000 | 1,536 | 1,536 | 1,536 | 30.21 | 30.83 | 39.6 / 42.4 | 2.2 / 4.0 |
| bge-small-en-v1.5 384d binary, rescored with float32 vectors | 100,000 | 5 | 158 | 5 | 0.34 | 0.37 | 9.2 / 11.9 | 0.2 / 0.4 |
| bge-small-en-v1.5 384d binary, rescored with float32 vectors | 1,000,000 | 48 | 1,584 | 48 | 3.10 | 3.25 | 12.0 / 14.8 | 2.2 / 4.0 |
| bge-base-en-v1.5 768d float32 | 100,000 | 307 | 307 | 307 | 6.09 | 6.37 | 19.5 / 40.5 | 0.6 / 1.0 |
| bge-base-en-v1.5 768d float32 | 1,000,000 | 3,072 | 3,072 | 3,072 | 60.28 | 61.65 | 74.7 / 95.8 | 5.7 / 9.6 |
| bge-base-en-v1.5 768d int8 | 100,000 | 77 | 77 | 77 | 12.75 | 12.89 | 26.0 / 47.1 | 0.6 / 1.0 |
| bge-base-en-v1.5 768d int8 | 1,000,000 | 768 | 768 | 768 | 126.99 | 128.69 | 141.8 / 162.9 | 5.7 / 9.6 |
| bge-base-en-v1.5 768d binary, rescored with float32 vectors | 100,000 | 10 | 317 | 10 | 0.41 | 0.46 | 13.6 / 34.6 | 0.6 / 1.0 |
| bge-base-en-v1.5 768d binary, rescored with float32 vectors | 1,000,000 | 96 | 3,168 | 96 | 3.65 | 3.74 | 16.8 / 37.9 | 5.7 / 9.6 |
| bge-base-en-v1.5 256d float32 | 100,000 | 102 | 102 | 102 | 2.04 | 2.20 | 15.3 / 36.4 | 0.6 / 1.0 |
| bge-base-en-v1.5 256d float32 | 1,000,000 | 1,024 | 1,024 | 1,024 | 20.23 | 20.64 | 33.7 / 54.8 | 5.7 / 9.6 |

RAM and disk are bytes per vector times chunks. FAISS index MB is what FAISS reports holding for the stored vectors. Rescoring with float32 vectors also reads them from disk, where the page cache may keep them in memory.
Search: exact FAISS indexes, one query at a time on 1 thread, 200 SciFact queries, over repeated real vectors.
Projected query p95 adds the benchmark's p95 to encode one query to the search p95, so it is conservative. Projected embedding time is chunks divided by the benchmark's throughput.
