# Downstream MIL — single encoder vs. concatenation vs. shared space

![fig3_downstream_auc](fig3_downstream_auc.png)

**Caption.** Slide-level prediction under three input representations, split into the two difficulty regimes. Top: morphological/clinical tasks saturate near AUC 0.97 and the three representations are indistinguishable. Bottom: molecular (mutation) tasks are where representations separate — MOSAIC (the shared space) wins on some (BRCA GATA3, COAD PIK3CA) and loses on others (LUAD/COAD TP53). The dashed line is chance (AUC 0.5).

## How it was computed

- **Classifier held constant across all three bars:** attention-MIL over a bag of patch embeddings → slide label. Only the *input representation* changes.
  - **Best single** — one encoder's patches; a separate MIL is trained per encoder and the bar reports the best of the 6.
  - **Concat** — all 6 encoders concatenated per patch (dim ≈ 5000); strictly more information than the shared space.
  - **MOSAIC** — the 6 encoders mapped through a GCCA aligner into a 64-d shared space; the aligner is fit on **training slides only**.
- **MIL heads:** ABMIL and TransMIL, 80 epochs, ≤4000 patches/bag; each bar averages the two heads (per-head values are in the CSV).
- **Splits:** patient-grouped and stratified — no patient appears in both train and test (TCGA/CPTAC have several slides per patient).
- **Metric:** macro/binary AUC. Every task is imbalanced, so AUC and balanced accuracy are the metrics to read; BRCA MAP3K1 (8% positive) is near-degenerate and its sub-chance values are noise.

## Source data

`results/full_run/downstream/downstream/*/results.csv`

## Files

- `fig3_downstream_auc.png` — raster preview
- `fig3_downstream_auc.pdf` — vector, for the paper
- `fig3_downstream_auc.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
