# Shared latent space — alignment methods, TCGA

![fig9_alignment_tcga](fig9_alignment_tcga.png)

**Caption.** The TCGA counterpart of the alignment-quality figure, on `master_benchmark/10x_256px` (5 encoders). The method ranking matches CPTAC almost exactly — GCCA best on retrieval and shared-space CKA, Procrustes best on reconstruction and neighbourhood preservation but weakest on alignment — which shows the alignment ↔ reconstruction trade-off is not cohort-specific.

## How it was computed

- Identical to the CPTAC alignment figure: five methods (generalized Procrustes, MCCA, autoencoder, joint PCA, GCCA) at latent dim 64, on five higher-is-better metrics; optimal transport excluded (see the CPTAC figure).
- **Encoders (5):** CONCH, CTransPath, Prov-GigaPath, ResNet50, UNI2.

## Source data

`results/groups/tcga_10x_256/alignment/aligner_comparison.csv`

## Files

- `fig9_alignment_tcga.png` — raster preview
- `fig9_alignment_tcga.pdf` — vector, for the paper
- `fig9_alignment_tcga.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
