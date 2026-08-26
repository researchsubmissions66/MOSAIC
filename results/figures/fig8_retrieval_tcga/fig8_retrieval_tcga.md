# Cross-model retrieval in the shared latent space — TCGA

![fig8_retrieval_tcga](fig8_retrieval_tcga.png)

**Caption.** The TCGA counterpart of the retrieval figure, on the `master_benchmark/10x_256px` group (5 encoders — the flagship set minus KEEP, which was not extracted for TCGA at this grid). The pattern replicates the CPTAC result: alignment lifts cross-model retrieval from chance to Recall@1 ≈ 0.95 (GCCA), and a rigid rotation is insufficient.

## How it was computed

- Identical protocol to the CPTAC retrieval figure — identity retrieval on held-out patches (30% test), aligners fit on the training split, latent dim 64, two unaligned PCA/truncation controls.
- **Encoders (5):** CONCH, CTransPath, Prov-GigaPath, ResNet50, UNI2. TCGA's 10×/256px group lacks KEEP, so a direct CPTAC↔TCGA comparison would restrict to these five shared encoders.

## Source data

`results/groups/tcga_10x_256/retrieval/retrieval_summary.csv`

## Files

- `fig8_retrieval_tcga.png` — raster preview
- `fig8_retrieval_tcga.pdf` — vector, for the paper
- `fig8_retrieval_tcga.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
