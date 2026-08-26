# Cross-model retrieval in the shared latent space

![fig2_retrieval](fig2_retrieval.png)

**Caption.** On the flagship CPTAC · 10× · 256px group (6 encoders). Patch-level retrieval where the database is indexed with one encoder and queried with another, after both are mapped into a shared space. Bars are alignment methods (darkest = GCCA); the unaligned control sits at ~0. Aligned methods lift cross-model retrieval from chance (~0.001) to Recall@1 ≈ 0.93, and a rigid rotation (Procrustes) is clearly insufficient.

## How it was computed

- **Task:** identity retrieval — the one correct answer for a query patch is that same patch encoded by the database model. Chance Recall@1 = 1 / database size.
- **Split:** aligners are fit on a training patch split; every score is computed on held-out patches (30% test).
- **Conditions:** GCCA, joint PCA, MCCA, shared autoencoder, and generalized Procrustes, each at latent dim 64, plus two unaligned controls (per-model independent PCA / truncation to 64 dims) that isolate alignment from dimensionality reduction.
- **Metrics:** Recall@{1,5,10}, mAP, NDCG. For identity retrieval mAP equals MRR by definition, so MRR is omitted here.
- Values are means over all ordered encoder pairs (same-model pairs excluded).

## Source data

`results/full_run/analysis/retrieval/retrieval_summary.csv`

## Files

- `fig2_retrieval.png` — raster preview
- `fig2_retrieval.pdf` — vector, for the paper
- `fig2_retrieval.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
