# TCGA · 20× · 224px — the four figures and every number behind them

This folder holds the complete four-stage study of one group. It is the only
group in `results/groups/` with all four stages on disk, and the only one whose
numbers are unaffected by the slide withholding, so it is the group to reach for
when something needs to be quotable today.

```
similarity.pdf    7 metrics x a 4x4 encoder matrix
retrieval.pdf     7 conditions (5 aligners + 2 unaligned controls)
alignment.pdf     5 aligners x 11 metrics
transfer.pdf      cross-model translation, 4x4
```

Data: `results/groups/tcga_20x_224/`. Metric definitions for the alignment stage
are in `results/groups/tcga_20x_224/alignment/README.md`; the group design and
the other seventeen grids are in `results/groups/README.md`.

## The group

Four encoders share the TCGA 20×/224px coordinate grid. That is why these four
and no others appear — see `results/groups/README.md` for why a group's encoder
set is imposed by its grid rather than chosen.

| encoder | dim | architecture | objective | pretrain | params |
|---|---|---|---|---|---|
| GPFM | 1024 | ViT-L/14 | expert distillation | ~190k WSI | 307M |
| H-optimus-0 | 1536 | ViT-g/14 | DINOv2 | ~500k WSI | 1.1B |
| Virchow | 2560 | ViT-H/14 | DINOv2 | 1.5M WSI | 632M |
| Virchow2 | 2560 | ViT-H/14 | DINOv2 | 3.1M WSI | 632M |

**All four are vision-SSL.** There is no vision-language encoder and no ImageNet
control on this grid — CONCH and KEEP are 256px/512px, ResNet50 (ImageNet) is
256px. Any claim about *model families* has to come from the 10×/256px flagship
instead. What this group supports is a comparison *within* one family, where
Virchow and Virchow2 differ only in pretraining scale (1.5M → 3.1M WSI) and are
otherwise the same architecture at the same dimension.

**n = 2169 slides**, spanning BRCA (1126), LUAD (531) and LUSC (512).

## Currency

The four encoders hold **exactly 2169 slides each with zero withheld slides
among them**. Virchow2 has 2726 at this grid including 557 TCGA-RCC, but those
557 are absent from the other three, so the intersection drops them regardless.
`configs/excluded_slides.txt` therefore cannot change this group's membership,
and the 2026-08-25 run date carries no risk — unlike every CPTAC group, where
all encoders hold all 134 CPTAC-LSCC slides and pre-withholding runs are stale.

**The PDFs are one style generation behind.** Rendered 2026-08-26 18:43; the
rounded-tile restoration landed 2026-08-27 15:01 in `a3604a2`. These four draw
the matrix as an `imshow` raster with square butted tiles rather than separated
rounded tiles. Structurally visible in the file: each carries 14 image XObjects
(7 heatmap rasters + 7 colourbars) where a current render carries 7. The
*numbers* are current; only the drawing is stale.

## similarity.pdf

Seven panels, one per metric, each a 4×4 matrix of pairwise similarity. Mean
off-diagonal and the full matrices:

| metric | mean off-diag | GPFM↔Hopt | GPFM↔V | GPFM↔V2 | Hopt↔V | Hopt↔V2 | V↔V2 |
|---|---|---|---|---|---|---|---|
| Linear CKA | 0.459 | **0.710** | 0.496 | 0.354 | 0.524 | 0.374 | 0.295 |
| Kernel CKA | 0.488 | **0.728** | 0.514 | 0.402 | 0.535 | 0.415 | 0.336 |
| SVCCA | 0.568 | 0.590 | 0.600 | 0.604 | 0.519 | 0.533 | 0.564 |
| PWCCA | 0.767 | 0.720 | 0.811 | **0.821** | 0.760 | 0.765 | 0.726 |
| Procrustes | 0.650 | 0.668 | 0.646 | 0.646 | 0.635 | 0.634 | 0.669 |
| Cosine RSA | 0.396 | **0.604** | 0.416 | 0.312 | 0.453 | 0.321 | 0.272 |
| Distance Correlation | 0.693 | **0.852** | 0.719 | 0.632 | 0.733 | 0.641 | 0.582 |

### The seven metrics form two blocks that rank pairs in opposite orders

This is the most important thing in the folder, and it is easy to miss because
each panel looks reasonable on its own.

`similarity/matrices/metric_agreement.csv` holds the Spearman correlation
between metrics over the six encoder pairs:

- **Linear CKA, Kernel CKA, Cosine RSA and Distance Correlation are mutually
  ρ = 1.000.** Four metrics, four different formulas, and they induce the
  *identical* ranking of all six pairs. On this group they are one measurement.
- **PWCCA anti-correlates with all four at ρ = −0.371**, and with Procrustes at
  **ρ = −0.486**.
- SVCCA sits with PWCCA (ρ = +0.486) and against the four (ρ = −0.200).

Concretely: the geometry block calls **GPFM↔H-optimus-0 the most similar pair**
(0.710 linear CKA, 0.604 RSA, 0.852 distance correlation) and **Virchow↔Virchow2
the least** (0.295, 0.272, 0.582). PWCCA reverses it — GPFM↔H-optimus-0 is its
*lowest* pair at 0.720 and GPFM↔Virchow2 its highest at 0.821.

Two models from the same lab, same architecture, same dimension, differing only
in pretraining scale, are the **least** similar pair by four metrics. A paper
that reports only linear CKA and a paper that reports only PWCCA would draw
opposite conclusions about which encoders resemble each other, from the same
features.

### Why the disagreement, and which block to trust here

PWCCA and SVCCA fit a linear map before measuring, which requires inverting a
covariance and so needs n ≫ d (`utils/cka.py:244` warns at n ≤ d). The largest
d here is Virchow/Virchow2 at **2560**. The similarity stage samples patches, not
slides, so n is large enough to avoid the hard degeneracy seen in the
slide-encoder study — but PWCCA's absolute level (0.767 mean, against 0.459 for
linear CKA) is the signature of a metric absorbing differences into its fitted
map. The geometry block inverts nothing.

Read the geometry block as the finding and PWCCA as the counter-example that
shows the choice of metric is doing work. Do not average across the seven.

## retrieval.pdf

Every patch is embedded by one encoder, projected into the shared space, and
looked up in a database built from a **different** encoder. 5000 queries against
5000 candidates. Chance recall@1 is 1/5000 = 0.0002.

| condition | recall@1 | recall@5 | mAP | median rank |
|---|---|---|---|---|
| GCCA | **0.9953** | 0.9997 | 0.9973 | 1.0 |
| autoencoder | 0.9699 | 0.9936 | 0.9805 | 1.0 |
| joint PCA | 0.9684 | 0.9945 | 0.9803 | 1.0 |
| MCCA | 0.9282 | 0.9800 | 0.9518 | 1.0 |
| Procrustes | 0.4862 | 0.7439 | 0.6032 | 1.67 |
| unaligned (truncate) | 0.00023 | 0.00105 | 0.0019 | 2364 |
| unaligned (PCA) | 0.00017 | 0.00080 | 0.0015 | 2685 |

**Both unaligned controls sit at chance.** 0.0002 against a chance level of
0.0002 — without alignment, a patch's embedding under one encoder carries no
usable information about where it lands under another, at rank-1. Median rank
2364 of 5000 is the middle of the list. This is the floor that makes GCCA's
0.9953 meaningful, and it is why the controls belong in the figure.

Per-pair, GCCA is uniform (0.9916-0.9978 across all twelve ordered off-diagonal
pairs) while Procrustes is not (0.354-0.639), and Procrustes is **asymmetric**:
GPFM→H-optimus-0 reads 0.604 but H-optimus-0→GPFM reads 0.639, and Virchow→GPFM
0.380 against GPFM→Virchow 0.470. A rigid map that fits one direction well does
not fit its inverse equally.

## alignment.pdf

Eleven metrics per aligner, grouped by which failure they catch. Definitions and
line references: `results/groups/tcga_20x_224/alignment/README.md`.

| aligner | recon R² | recon cos | align err | paired cos | shared CKA | recall@1 | nbhd pres | eff rank |
|---|---|---|---|---|---|---|---|---|
| joint_pca | 0.540 | 0.771 | 0.037 | 0.950 | 0.928 | 0.981 | 0.544 | 56.8 |
| gcca | 0.469 | 0.739 | **0.030** | **0.962** | 0.927 | **0.997** | 0.465 | **63.9** |
| mcca | 0.510 | 0.760 | 0.154 | 0.823 | 0.693 | 0.952 | 0.535 | 63.9 |
| procrustes | **0.588** | **0.795** | 0.335 | 0.670 | 0.459 | 0.604 | **0.744** | 57.0 |
| autoencoder | 0.570 | 0.787 | 0.064 | 0.917 | **0.951** | 0.974 | 0.579 | 55.4 |

Reconstruction and alignment trade off: Procrustes is best on reconstruction and
worst on alignment, and best on neighbourhood preservation while worst on
correspondence. It is the only rigid aligner, so it preserves each view's
internal geometry and pays for it. GCCA inverts the trade and uses nearly all 64
latent dimensions (effective rank 63.9) to do it. Both halves replicate on the
CPTAC flagship with a disjoint encoder set.

**`recall@1` here is not `recall@1` in retrieval.pdf.** Procrustes reads 0.604
in this table and 0.486 in the retrieval one — different evaluations, different
protocols. Name which you quote.

## transfer.pdf

Translation between encoders through the shared space: take a patch's embedding
under encoder A, map it into B's space, and ask whether it lands where B would
have put it. `self_roundtrip` is the same operation with A = B, and is the
ceiling — it measures only what the round trip through a 64-d bottleneck costs.

| | cosine | R² | recall@1 | mAP |
|---|---|---|---|---|
| cross-model | 0.702 | 0.393 | 0.470 | 0.572 |
| self round-trip | 0.784 | 0.563 | 0.807 | 0.863 |

Cross-model reaches 58% of the self round-trip's recall@1 (0.470 / 0.807), so
roughly **four tenths of the loss in translating between encoders is the
bottleneck itself, not the translation.** Reporting cross-model transfer without
the self round-trip overstates the cross-model penalty.

Per-encoder round-trip cosine varies more than the aggregate suggests — the
`matrix_cosine` diagonal runs 0.649 (H-optimus-0) to 0.902 (Virchow) — and the
matrix is strongly **column-dominated**: everything reconstructs *into* Virchow
well (0.838-0.850) and *into* H-optimus-0 poorly (0.572-0.588), largely
independent of the source. Being an easy target and being a good source are
different properties, which a symmetric similarity matrix cannot express. This
is what the transfer stage adds over `similarity.pdf`.

## Regenerating

```
python scripts/run_study.py --out results/groups/tcga_20x_224 \
    --preset full --group master_benchmark/20x_224px \
    --stages similarity alignment transfer retrieval
python scripts/render_figures.py --out results/figures --format pdf
```

`--preset full` is five aligners; the standalone stage scripts default to three.
The alignment, retrieval and transfer stages sample **500 slides / 50 000
patches** (seed 0, latent_dim 64) while similarity reads the full group — so
those three carry sampling variance that `similarity.pdf` does not.

## Known issues

- The four PDFs predate the rounded-tile style restoration (see **Currency**).
  Re-rendering costs nothing but a login-node run of `render_figures.py`.
- Bar figures emit `tight_layout` warnings from `utils/paperfigs.py:672`;
  cosmetic, no effect on output.
