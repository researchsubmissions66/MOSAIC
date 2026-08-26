# Representational similarity across pathology foundation models

![fig1_similarity](fig1_similarity.png)

**Caption.** Pairwise similarity between the six encoders of the flagship group (`cptac_benchmark/10x_256px`), one panel per metric. Each panel carries its own colour scale — the metrics sit at different levels, and a shared scale would flatten all but the widest-ranging. Only the lower triangle is shown (the matrices are symmetric). ResNet50, the ImageNet-supervised control, is consistently the least similar to the pathology encoders.

## How it was computed

- **Encoders (6):** CONCH, CTransPath, Prov-GigaPath, KEEP, ResNet50, UNI2 — the only patch encoders sharing the 10x/256px coordinate grid, hence row-paired by patch index.
- **Data:** 50,000 patches sampled across ≤500 CPTAC slides (shared subsample, seed 0); the O(n²) metrics subsample 8,000.
- **Metrics:** linear CKA (feature-space form), RBF-kernel CKA (median-heuristic bandwidth), SVCCA (99% variance retained), and normalised orthogonal Procrustes similarity.
- Every matrix is computed on column-centered features; the diagonal is 1.0 by construction.

## Source data

`results/full_run/analysis/similarity/matrices/{linear_cka,kernel_cka,svcca,procrustes}.csv`

## Files

- `fig1_similarity.png` — raster preview
- `fig1_similarity.pdf` — vector, for the paper
- `fig1_similarity.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
