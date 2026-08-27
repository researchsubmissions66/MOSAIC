# Similarity by patch grid

One figure per `(patch grid, cohort, magnification)`. These are separate
results, not panels of one figure: similarity needs row-paired patches, and
trident writes one coordinate grid per `(magnification, patch_size)`, so
encoders on different grids share no row index and were never compared.

All seven metrics are shown in every figure.

## 224px

_GPFM, H-optimus-0, Virchow, Virchow2 (Virchow only at 20x)_

| figure | cohort | magnification | encoders | source |
|---|---|---|---|---|
| `224px/cptac_5x_224px.pdf` | CPTAC | 5x | 3 | `results/full_run/analysis/similarity_224px/cptac_benchmark_5x_224px/matrices` |
| `224px/cptac_10x_224px.pdf` | CPTAC | 10x | 3 | `results/full_run/analysis/similarity_224px/cptac_benchmark_10x_224px/matrices` |
| `224px/cptac_20x_224px.pdf` | CPTAC | 20x | 4 | `results/groups/cptac_20x_224/similarity/matrices` |
| `224px/tcga_5x_224px.pdf` | TCGA | 5x | 3 | `results/full_run/analysis/similarity_224px/master_benchmark_5x_224px/matrices` |
| `224px/tcga_10x_224px.pdf` | TCGA | 10x | 3 | `results/full_run/analysis/similarity_224px/master_benchmark_10x_224px/matrices` |
| `224px/tcga_20x_224px.pdf` | TCGA | 20x | 4 | `results/groups/tcga_20x_224/similarity/matrices` |

## 256px

_CONCH / CONCH v1.5, CTransPath, Prov-GigaPath, KEEP, ResNet50, UNI2-h_

| figure | cohort | magnification | encoders | source |
|---|---|---|---|---|
| `256px/cptac_10x_256px.pdf` | CPTAC | 10x | 6 | `results/full_run/analysis/similarity/matrices` |
| `256px/cptac_20x_256px.pdf` | CPTAC | 20x | 5 | `results/groups/cptac_20x_256/similarity/matrices` |
| `256px/tcga_10x_256px.pdf` | TCGA | 10x | 6 | `results/full_run/analysis/similarity_256px/master_benchmark_10x_256px/matrices` |
| `256px/tcga_20x_256px.pdf` | TCGA | 20x | 6 | `results/full_run/analysis/similarity_256px/master_benchmark_20x_256px/matrices` |

## 512px

_CONCH vs CONCH v1.5_

| figure | cohort | magnification | encoders | source |
|---|---|---|---|---|
| `512px/cptac_5x_512px.pdf` | CPTAC | 5x | 2 | `results/full_run/analysis/similarity_512px/cptac_benchmark_5x_512px/matrices` |
| `512px/cptac_10x_512px.pdf` | CPTAC | 10x | 2 | `results/full_run/analysis/similarity_512px/cptac_benchmark_10x_512px/matrices` |
| `512px/cptac_20x_512px.pdf` | CPTAC | 20x | 2 | `results/full_run/analysis/similarity_512px/cptac_benchmark_20x_512px/matrices` |
| `512px/tcga_5x_512px.pdf` | TCGA | 5x | 2 | `results/full_run/analysis/similarity_512px/master_benchmark_5x_512px/matrices` |
| `512px/tcga_10x_512px.pdf` | TCGA | 10x | 2 | `results/full_run/analysis/similarity_512px/master_benchmark_10x_512px/matrices` |
| `512px/tcga_20x_512px.pdf` | TCGA | 20x | 2 | `results/full_run/analysis/similarity_512px/master_benchmark_20x_512px/matrices` |

## Not represented

- **384px** — MUSK is the only encoder on it, so similarity is undefined.
- **5x/256px** — covered by the magnification series rather than a
  standalone matrix; see `results/figures/magnification/`.
