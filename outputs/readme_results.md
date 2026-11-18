### Held out test results

| Method | P@10 | MRR | NDCG@10 | NDCG@10 graded |
| --- | --- | --- | --- | --- |
| TF IDF | 0.457 | 0.575 | 0.452 | 0.535 |
| BM25 | 0.466 | 0.609 | 0.465 | 0.531 |
| minilm full | 0.476 | 0.617 | 0.473 | 0.562 |
| minilm chunked | 0.454 | 0.584 | 0.445 | 0.53 |
| bge_small full | 0.503 | 0.627 | 0.494 | 0.557 |
| bge_small chunked | 0.451 | 0.627 | 0.451 | 0.529 |
| Skills only | 0.182 | 0.173 | 0.16 | 0.243 |
| Hybrid | 0.486 | 0.596 | 0.48 | 0.561 |

### Graded NDCG@10 with 95% bootstrap confidence intervals

| method | NDCG@10 graded | CI low | CI high |
| --- | --- | --- | --- |
| TF IDF | 0.535 | 0.459 | 0.613 |
| BM25 | 0.531 | 0.46 | 0.602 |
| Best embedding (minilm full) | 0.562 | 0.481 | 0.637 |
| Hybrid | 0.561 | 0.479 | 0.637 |

* Hybrid minus TF IDF: 0.026 (95% CI -0.015 to 0.068)
* Hybrid minus BM25: 0.031 (95% CI -0.019 to 0.079)

Best embedding variant (chosen on tuning queries): minilm full
Hybrid weights: {'bm25': 0.1, 'embedding': 0.8, 'skills': 0.1}