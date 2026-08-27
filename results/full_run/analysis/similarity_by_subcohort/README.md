# Similarity within a subcohort, instead of pooled across one

Every other similarity matrix in this study samples patches across a whole
benchmark cohort, which mixes tissue types. CPTAC is ~53% lung, ~30% breast,
~17% colon; TCGA is ~52% breast. That makes any cohort-level comparison partly a
comparison of tissue composition, and it cannot be separated after the fact —
the pooled matrices carry no subcohort structure.

These runs recompute the seven metrics with the patch sample restricted to one
subcohort at a time, so tissue is held fixed. Sampling matches the pooled runs
exactly (`--n-patches 20000 --max-slides 200 --max-samples 5000`, seed 0), so
the two are directly comparable.

Produced by `scripts/similarity_by_subcohort.py`.

**Status: partial.** 14 of 51 runs are on disk, all CPTAC. The TCGA subcohorts
and the CPTAC slide encoders are still queued. Everything below is CPTAC-only
and will be extended, not revised, when the rest lands.

## Subcohorts

Six, kept at tissue granularity: `TCGA-LUAD`, `TCGA-LUSC`, `TCGA-BRCA`,
`CPTAC-LUAD`, `CPTAC-BRCA`, `CPTAC-COAD`. LUAD and LUSC are deliberately *not*
merged into TCGA-NSCLC — NSCLC is a task construct (the LUAD-vs-LUSC
classification), not a tissue unit, and holding tissue fixed is the point. They
can be pooled afterwards for reporting; they cannot be un-pooled.

TCGA-RCC and CPTAC-LSCC never appear: they are withheld study-wide in
`configs/excluded_slides.txt`.

## The headline: pooling inflates the ImageNet control gap

The flagship claim — pathology encoders agree with each other more than with the
ImageNet-supervised control — is the sanity check the shared-manifold argument
rests on. Linear CKA on the flagship six at `10x/256px`:

| | pathology↔pathology | vs ResNet50 | **gap** |
|---|---|---|---|
| **POOLED** | 0.678 | 0.479 | **0.199** |
| CPTAC-LUAD | 0.661 | 0.512 | 0.149 |
| CPTAC-COAD | 0.718 | 0.644 | 0.073 |
| CPTAC-BRCA | 0.715 | 0.670 | **0.045** |

**The pooled gap is larger than any subcohort's gap.** Not an average of them —
larger than the maximum. Pooling barely moves pathology↔pathology (0.678 vs
0.661–0.718) but drives the ResNet50 column down hard (0.479 vs 0.512–0.670).

The mechanism is between-tissue variance. ResNet50 separates tissue types more
sharply than the pathology encoders do, so mixing tissues pushes its geometry
away from theirs. Most of the headline gap is that separation, not a
within-tissue disagreement about morphology.

**Consequence for the paper: the control result is a lung result.** It is
0.149 on LUAD, 0.073 on COAD, and **0.045 on breast** — where ResNet50–CONCH
(0.72) actually exceeds CONCH–GigaPath (0.68), i.e. the control is closer to a
pathology encoder than two pathology encoders are to each other. Quoting 0.199
without saying it is pooled overstates the effect by roughly a third.

This also bears on the cohort comparison recorded in `../similarity/README.md`:
the gap is much weaker on TCGA (0.060 at 10x) than CPTAC (0.199 pooled). TCGA is
52% breast and CPTAC 53% lung, and breast is exactly where the gap collapses —
so that difference is at least partly tissue, not cohort. The queued TCGA
subcohort runs test it directly, BRCA against BRCA.

## Tissue rank depends on the grid

Mean off-diagonal linear CKA:

| grid | mag | BRCA | COAD | LUAD |
|---|---|---|---|---|
| 224px | 10x | 0.496 | 0.528 | **0.567** |
| 224px | 20x | 0.506 | **0.445** | 0.511 |
| 256px | 10x | **0.700** | 0.693 | 0.611 |
| 256px | 20x | 0.666 | 0.659 | — |
| 512px | 10x | 0.834 | 0.835 | **0.874** |

LUAD is the *lowest* subcohort at 256px and the *highest* at 224px and 512px.
"Which tissue do encoders agree on" is therefore not a property of the tissue —
it depends on the patch grid, and by extension on which encoders that grid
carries. Do not report a tissue ordering without naming the grid.

Note the pooled CPTAC 10x/256px value (0.600) sits **below all three of its own
subcohorts** (0.611–0.700), which is the same pooling effect as above.

## Reading these tables

- Compare **down a column within one grid+magnification block**. There the
  encoders and the magnification are fixed and only tissue varies.
- Do **not** compare across grids: the encoder sets differ. 224px is 3 encoders
  at 5x/10x and 4 at 20x (Virchow exists only at 20x); 256px is 6 at 10x and 5
  at 20x (no CONCH); 512px is CONCH vs CONCH v1.5, a single pair.
- The 512px rows are one pair of one model family at two versions. They are high
  for that reason and are not comparable to the multi-encoder rows.
- CPTAC-COAD is the smallest subcohort (369 slides). For the slide-encoder runs
  still queued, that will put n/d below 1 and make SVCCA and PWCCA degenerate;
  read CKA, cosine RSA, distance correlation and Procrustes there.

## Contents

```
<group>/<subcohort>/matrices/{metric}.csv   7 metrics per (group, subcohort)
similarity_by_subcohort.csv                 combined summary (written at end of a full run)
```

Figures: `results/figures/similarity_by_subcohort/` — per-subcohort panels by
grid, plus `compare/` where one metric is shown across subcohorts on a shared
colour scale.
