# Representational agreement decreases with magnification

![fig4_magnification](fig4_magnification.png)

**Caption.** The same similarity analysis repeated independently at 5×, 10× and 20× (CPTAC, 256px, the five encoders common to all three magnifications). Every one of the seven metrics declines monotonically as resolution increases: models agree most on coarse tissue architecture (5×) and diverge on fine nuclear detail (20×). Lines and legend are ordered by their value at 20×.

## How it was computed

- **Each magnification is an independent replication**, not a paired comparison: trident writes a separate coordinate grid per magnification, so a 5× patch has no correspondence to a 20× patch. What is compared across magnifications is the *result* (the similarity matrix), never the embeddings.
- **Held fixed:** the encoder set (intersected across magnifications → CTransPath, GigaPath, KEEP, ResNet50, UNI2) and the slide set (only slides present at all three), with one shared seed.
- **Plotted value:** the mean off-diagonal similarity across all encoder pairs, per metric, at each magnification. 50,000 patches sampled per magnification; O(n²) metrics subsample 8,000.

## Source data

`results/magnification/cptac_benchmark_256px/magnification_summary.csv`

## Files

- `fig4_magnification.png` — raster preview
- `fig4_magnification.pdf` — vector, for the paper
- `fig4_magnification.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
