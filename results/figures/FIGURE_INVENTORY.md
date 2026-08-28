# Figure inventory — cohort and encoders for every figure

Which cohort and which encoders each figure is computed on, for the figures
inherited from the original release as well as the ones added since. Two facts
make this worth writing down rather than reading off a title:

1. **A figure's encoder set is decided by its patch grid, not chosen.**
   Similarity needs row-paired patches, and trident writes one coordinate grid
   per `(magnification, patch_size)` (`utils/features.py:10-23`). Encoders on
   different grids share no row index, so a figure shows every encoder on its
   grid and no others. Where a figure shows five encoders instead of six, that
   is almost always a missing extraction rather than a decision.
2. **The two cohorts do not carry the same encoder sets**, and neither do two
   magnifications of the same cohort. Several figure pairs that read as
   CPTAC-against-TCGA are computed on different sets.

## Cohorts

| cohort | store key | slides in scope | composition |
|---|---|---|---|
| CPTAC | `cptac_benchmark` | 2162 | LUAD 1139, BRCA 654, COAD 369 |
| TCGA | `master_benchmark` | 2169 | BRCA 1126, LUAD 531, LUSC 512 |

Withheld study-wide by `configs/excluded_slides.txt` (1074 slides): **TCGA-RCC**
(KICH/KIRC/KIRP, 940) and **CPTAC-LSCC** (134). Features remain on disk; the
filter is applied in `FeatureGroup.slides()` and `SlideEncoderSet.slides()`.

Group sizes are not uniform within a cohort. `master_benchmark/20x_256px` and
`/5x_512px` hold 1126 slides and are **TCGA-BRCA only** — a single-subcohort
group, not the mixed cohort, which is why numbers there sit higher.

## Patch encoders

Twelve are registered. Six share the flagship `10x/256px` grid:

| encoder | dim | family |
|---|---|---|
| CONCH | 512 | vision-language |
| CTransPath | 768 | vision-SSL |
| Prov-GigaPath | 1536 | vision-SSL |
| KEEP | 768 | vision-language |
| ResNet50 (ImageNet) | 1024 | supervised, control |
| UNI2-h | 1536 | vision-SSL |

The other six sit on grids that cannot be paired with those: **GPFM,
H-optimus-0, Virchow, Virchow2** (224px), **CONCH v1.5** (768-d; 512px and
`master/20x_256px`), and **MUSK** (384px, alone on its grid — similarity is
undefined at n=1, which is why MUSK appears in no similarity figure).

## Slide encoders

Six, declared in `configs/encoders.yaml` under `slide_encoders`: **CHIEF,
Feather, GigaPath, Madeleine, PRISM, TITAN**. They emit one vector per slide, so
the row-pairing constraint does not apply and all six compare at once despite
sitting on different grids (CHIEF and Madeleine 10x/256px, GigaPath 20x/256px,
PRISM 20x/224px, Feather and TITAN 20x/512px). Each exists at exactly one grid,
so **grid is perfectly aliased with encoder identity** here — no difference
between two slide encoders can be attributed to the model rather than its grid.

`care` and `prism2` are excluded: unregistered, TCGA-only, and extracted for
breast slides only, which had pinned the TCGA slide set to 1126 pure-BRCA
slides. Dropping them takes TCGA to 2169 mixed slides and PWCCA saturation from
13/28 pairs to 0/15.

## Figure map

| figure | cohort | grid | encoders |
|---|---|---|---|
| `fig1_similarity` | CPTAC | 10x/256px | 6 — the flagship set |
| `fig2_retrieval` | CPTAC | 10x/256px | 6, through 5 aligners + 2 unaligned controls |
| `fig3_downstream_auc` | both | 10x/256px | **CPTAC 6, TCGA 5** — see below |
| `fig4_magnification` | CPTAC | 256px, 5x/10x/20x | **5** — no CONCH |
| `fig4b_magnification_tcga` | TCGA | 256px, 5x/10x/20x | 5, same set |
| `fig4c_magnification_tcga_224px` | TCGA | 224px, 5x/10x/20x | **3** — GPFM, H-optimus-0, Virchow2 |
| `fig5_transfer` | CPTAC | 10x/256px | 6 |
| `fig6_slide_encoders` | both | slide-level | 6 slide encoders, each cohort |
| `fig7_alignment_methods` | CPTAC | 10x/256px | 6, across 5 aligners |
| `fig8_retrieval_tcga` | TCGA | 10x/256px | **5 — no KEEP** |
| `fig9_alignment_tcga` | TCGA | 10x/256px | **5 — no KEEP** |
| `fig10_layerwise` | TCGA | 10x/256px | block-wise, `cls` and `mean` pooling |
| `fig_control_gap_linear_cka` | CPTAC | 10x/256px | 6, pooled against each subcohort |
| `fig_slide_retrieval_cptac` | CPTAC | slide-level | 6 slide encoders |
| `similarity_by_grid/` | both | all grids | one row per group, all 7 metrics |
| `similarity_by_subcohort/` | CPTAC | per grid | as the grid, split LUAD / BRCA / COAD |
| `slide_encoders/` | both | slide-level | 6, all 7 metrics |

### Encoder sets by similarity group

The sets behind those rows, read off the matrices themselves:

| group | n | encoders |
|---|---|---|
| CPTAC 10x/256px | 6 | CONCH, CTransPath, Prov-GigaPath, KEEP, ResNet50, UNI2-h |
| CPTAC 20x/256px | 5 | as above, **no CONCH** |
| TCGA 10x/256px | 6 | CONCH, CTransPath, Prov-GigaPath, KEEP, ResNet50, UNI2-h |
| TCGA 20x/256px | 6 | **CONCH v1.5** in place of CONCH — a different six |
| both, 5x + 10x/224px | 3 | GPFM, H-optimus-0, Virchow2 |
| both, 20x/224px | 4 | adds Virchow |
| both, all 512px | 2 | CONCH, CONCH v1.5 |

## Caveats that change interpretation

**fig3 is six encoders on CPTAC and five on TCGA.** KEEP has no TCGA downstream
run. `Best single` filters KEEP so the maximum is over the same five in both
cohorts (KEEP never wins a task, so no plotted value moves), but the `concat`
and `shared` conditions are *fitted* on all six for CPTAC and five for TCGA — no
figure-level filter can reach inside a fitted model. Downstream runs two MIL
heads, **ABMIL and TransMIL**, over three conditions: `single`, `concat`,
`shared`.

**fig8 and fig9 are five-encoder figures against six-encoder counterparts.**
They are the TCGA versions of fig2 and fig7, and were computed before KEEP
reached the TCGA 10x/256px group. The similarity data for that group now carries
six; the retrieval and alignment results still carry five. Any CPTAC-vs-TCGA
statement drawn from that pair is confounded by the extra encoder unless it is
restricted to the five.

**fig4 and fig4b use five encoders, not six.** CONCH has no 5x or 20x
extraction at 256px, so including it would have collapsed the sweep to a single
magnification.

**"TCGA 20x/256px" is a different six from "TCGA 10x/256px."** CONCH v1.5
replaces CONCH there, and the group is TCGA-BRCA only (1126 slides). Two rows of
the all-grids table that both read "256px TCGA, 6 encoders" are therefore not
comparable — the encoder set and the tissue composition both change.

**The 512px rows are a single pair.** CONCH against CONCH v1.5 — one model
family at two versions. They read high (0.83-0.89 linear CKA) for that reason
and do not belong on the same axis as the multi-encoder rows.

**`gigapath` names two different models.** The patch encoder Prov-GigaPath
(d=1536) and the GigaPath *slide* encoder (d=768) share a registry key, so
`clean_label` renders both as "Prov-GigaPath". The TCGA slide-encoder matrices
carry that label for a slide encoder today. Cosmetic in the data, misleading in
a figure: a reader comparing fig1 with fig6 will take them for the same model.
They are not.

**Small groups bias the CCA family upward.** `utils/cka.py:244` warns whenever
n <= d. At the subcohort level this binds: CPTAC-COAD is 369 slides against
d=1280, so SVCCA and PWCCA saturate toward 1.0 there. CKA, RSA, Procrustes and
distance correlation invert nothing and stay usable at those sizes.
