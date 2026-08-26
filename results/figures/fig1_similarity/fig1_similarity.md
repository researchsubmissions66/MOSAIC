# Representational similarity across pathology foundation models

![fig1_similarity](fig1_similarity.png)

**Caption.** Pairwise similarity between the six encoders of the flagship group (`cptac_benchmark/10x_256px`), one panel per metric. Each panel carries its own colour scale — the metrics sit at different levels, and a shared scale would flatten all but the widest-ranging. Only the lower triangle is shown (the matrices are symmetric). ResNet50, the ImageNet-supervised control, is consistently the least similar to the pathology encoders. All seven metrics are shown: PWCCA ranks the pairs differently from the others (Spearman 0.175 against Procrustes), so reading any single metric as *the* similarity overstates how settled the picture is.

## How it was computed

- **Encoders (6):** CONCH, CTransPath, Prov-GigaPath, KEEP, ResNet50, UNI2 — the only patch encoders sharing the 10x/256px coordinate grid, hence row-paired by patch index.
- **Data:** 50,000 patches sampled across ≤500 CPTAC slides (shared subsample, seed 0); the O(n²) metrics subsample 8,000.
- **Metrics (7):** linear CKA (feature-space form), RBF-kernel CKA (median-heuristic bandwidth), SVCCA (99% variance retained), PWCCA, normalised orthogonal Procrustes, cosine RSA and distance correlation. `metric_agreement.csv` holds the Spearman agreement between them; distance correlation is redundant with kernel CKA (1.000) while PWCCA is the outlier.
- Every matrix is computed on column-centered features; the diagonal is 1.0 by construction.

## Source data

`results/full_run/analysis/similarity/matrices/*.csv`

## Files

- `fig1_similarity.png` — raster preview
- `fig1_similarity.pdf` — vector, for the paper
- `fig1_similarity.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
