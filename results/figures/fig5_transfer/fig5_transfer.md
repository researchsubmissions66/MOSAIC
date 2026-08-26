# Cross-model transfer (source → target)

![fig5_transfer](fig5_transfer.png)

**Caption.** Converting one encoder's embeddings into another's through the shared space. Left: mean cosine of the translated vector to the target's real embedding. Right: Recall@1 retrieving from the target's real index in its native space. Rows are the source encoder, columns the target. Both are directional, so the full 6×6 grid is shown. Note the bright ResNet50 *column*: ResNet50 is the easiest target to hit (lowest-dimensional, least structured), which is not the same as it being well reconstructed.

## How it was computed

- **Operation:** encode patches with the source model → GCCA shared space → decode as the target model. Evaluated on held-out patches (30% test), aligner fit on the training split.
- **Cosine panel:** mean row-wise cosine between the translated embedding and the target's *real* embedding of the same patch.
- **Recall@1 panel:** the translated query is used to retrieve from a database of the target's untouched real embeddings — the deployment-realistic test.
- A companion linear-probe analysis (in `transfer_summary.csv`, not plotted here) shows the discriminative content transfers even where the geometry does not: a probe trained on the target's real features keeps ~96% of its accuracy on translated ones.

## Source data

`results/full_run/analysis/transfer/matrix_{cosine,retrieval_recall1}.csv`

## Files

- `fig5_transfer.png` — raster preview
- `fig5_transfer.pdf` — vector, for the paper
- `fig5_transfer.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
