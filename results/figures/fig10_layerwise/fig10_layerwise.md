# Layer-wise representational alignment (architecture depth)

![fig10_layerwise](fig10_layerwise.png)

**Caption.** Phase IV — how similarity evolves through network depth, not just at the output, across five encoders. Left: CKA between every block of UNI2 (24) and GigaPath (40); the dashed line is the lockstep diagonal, and the bright deep-block region shows the two ViTs converging. Right: UNI2's best-match CKA to each other encoder against relative depth. **The result stratifies by architecture:** the ViT-based pathology models (GigaPath, CONCH) climb to ≈0.95–0.97 with depth — growing *more* similar to UNI2, against the universal-early / specific-late intuition — while the hybrid CTransPath stays low (≈0.45–0.50) and the ResNet50 control rises then falls. Architecture family, not depth alone, governs where two networks end up.

## How it was computed

- **This stage does not read the feature store** — trident saved only final pooled embeddings, so intermediate activations were recreated by re-running the models with forward hooks on patches re-cropped from downloaded slides at trident's own coordinates. The `patches/` coordinate grid survived even though the `features_*` were deleted, so the analysis is reproducible from the slides alone.
- **Encoders (5):** UNI2 (24 ViT blocks), GigaPath (40), CONCH (12 visual blocks — CoCa image tower), CTransPath (12 Swin blocks), ResNet50 (3 conv stages, global-average pooled). CTransPath and GigaPath were loaded in trident's own conda env (`timm==0.9.16` + `timm_ctp`), which the project's base env could not provide.
- **Metric:** linear CKA between block activations, CLS-token pooled (a `mean`-pooled variant is also on disk). 512 patches from 4 TCGA slides.
- **Divergence depth** is the *last sustained* crossing below CKA 0.5, not the first dip — real trajectories are not monotone.

## Source data

`results/layerwise/tcga_10x_256_cls/{matrices/*.csv,divergence_profile.csv}`

## Files

- `fig10_layerwise.png` — raster preview
- `fig10_layerwise.pdf` — vector, for the paper
- `fig10_layerwise.md` — this file

Regenerate with `python scripts/render_figures.py --out results/figures --format pdf`.
