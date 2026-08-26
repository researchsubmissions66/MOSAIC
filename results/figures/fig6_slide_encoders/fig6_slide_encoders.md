# Slide-level encoders — all six, across four patch grids

![fig6_slide_encoders](fig6_slide_encoders.png)

**Caption.** Linear CKA between the six slide-level encoders, for TCGA (left) and CPTAC (right). Because slide encoders emit one vector per slide, the patch-grid pairing constraint does not apply — all six are directly comparable even though they were built on four different patch grids, across every slide in each cohort. Feather↔TITAN is the strongest pair (shared lineage); CHIEF and Madeleine share a grid yet are only weakly similar, so the grid is not what drives agreement.

## How it was computed

- **Encoders (6):** CHIEF, Madeleine (10x/256px), PRISM (20x/224px), GigaPath-slide (20x/256px), TITAN, Feather (20x/512px).
- **Data:** one embedding per slide, paired by slide id across 2,169 TCGA / 2,296 CPTAC slides (all slides, no subsampling of slides).
- **Metric:** linear CKA on column-centered slide embeddings; lower triangle shown.
- **Caveat:** n ≈ 2,200 slides against 512–1280 dims is comfortable for CKA but near the floor for the CCA family, which is why only CKA is shown for the slide-level analysis.

## Source data

`results/slide_encoders/{master_benchmark,cptac_benchmark}/matrices/linear_cka.csv`

## Files

- `fig6_slide_encoders.png` — raster preview
- `fig6_slide_encoders.pdf` — vector, for the paper
- `fig6_slide_encoders.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
