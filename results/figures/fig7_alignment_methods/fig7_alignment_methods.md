# Shared latent space — quality of the alignment methods

![fig7_alignment_methods](fig7_alignment_methods.png)

**Caption.** On the flagship CPTAC · 10× · 256px group. The alignment methods compared on five shared-space quality metrics (all higher-is-better, rescaled to %). GCCA (darkest) is strongest on cross-model retrieval and shared-space agreement; the shared autoencoder and joint PCA trail; rigid generalized Procrustes is weakest on alignment but best on reconstruction — the expected trade-off, since a rotation preserves geometry but cannot warp one model onto another. Note that reconstruction and alignment pull in opposite directions, which is why both families of metric are shown together.

## How it was computed

- **Methods shown (5 of 6):** generalized Procrustes, MCCA, shared autoencoder, joint PCA, GCCA — all at latent dim 64, fit on a training patch split and evaluated on held-out patches. **Optimal transport is implemented but excluded from this run:** its unsupervised mode is unreliable and supervised OT reduces to Procrustes, so it would add a redundant bar.
- **Metrics:** cross-model Recall@1 (retrieval in the shared space), reconstruction R² (round-trip fidelity), paired cosine and shared-space CKA (how well the models agree once aligned), and k-NN neighbourhood preservation (a collapse detector). Reported side by side because optimising alignment alone can collapse the space, and optimising reconstruction alone leaves models unaligned.
- **Excluded from the plot:** `alignment_error` (lower-is-better, opposite direction) and `effective_rank` (0–64 scale) live in the CSV but would not share this axis.

## Source data

`results/full_run/analysis/alignment/aligner_comparison.csv`

## Files

- `fig7_alignment_methods.png` — raster preview
- `fig7_alignment_methods.pdf` — vector, for the paper
- `fig7_alignment_methods.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
