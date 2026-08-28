# Groups — the unit of analysis, and the eighteen of them

Everything in this folder is one *group*: a `(cohort, magnification,
patch_size)` triple, studied through four stages — similarity, alignment,
transfer, retrieval.

## Why the group is the unit

Similarity between two encoders requires their features to be **row-paired**:
row *i* of one matrix and row *i* of the other must be the same patch of the
same slide. Trident writes one coordinate grid per `(magnification,
patch_size)` and every encoder run against that grid inherits its row order
(`utils/features.py:10-23`). Encoders extracted at different grids share no row
index, so no amount of resampling makes them comparable.

The consequence runs through the whole study: **a group's encoder set is not
chosen, it is whatever sits on that grid.** A figure showing five encoders
where another shows six is almost always reporting a missing extraction, not a
decision. Nothing in a rendered figure says which case it is, which is why the
sets are tabulated here and in `results/figures/FIGURE_INVENTORY.md`.

## The eighteen groups

A grid qualifies when two or more *registered* encoders share it — two is the
minimum for a similarity matrix to have an off-diagonal entry. Enumerated from
the feature tree rather than from what happens to exist under `results/`:

| group | encoders | n | slides | job |
|---|---|---|---|---|
| `cptac_10x_256` | CONCH, CTransPath, Prov-GigaPath, KEEP, ResNet50, UNI2-h | 6 | 2162 | 21513862 |
| `tcga_10x_256` | same six | 6 | 2169 | 21481825 |
| `cptac_20x_256` | CTransPath, Prov-GigaPath, KEEP, ResNet50, UNI2-h | 5 | 2162 | 21481829 |
| `cptac_5x_256` | same five | 5 | 2162 | 21531730 |
| `tcga_5x_256` | same five | 5 | 2169 | 21531724 |
| `tcga_20x_256` | **CONCH v1.5** + CTransPath, GigaPath, KEEP, ResNet50, UNI2-h | 6 | **1126** | 21531725 |
| `cptac_20x_224` | GPFM, H-optimus-0, Virchow, Virchow2 | 4 | 2162 | 21481826 |
| `tcga_20x_224` | GPFM, H-optimus-0, Virchow, Virchow2 | 4 | 2169 | **on disk** |
| `cptac_10x_224` | GPFM, H-optimus-0, Virchow2 | 3 | 2162 | 21531729 |
| `cptac_5x_224` | GPFM, H-optimus-0, Virchow2 | 3 | 2162 | 21531728 |
| `tcga_10x_224` | GPFM, H-optimus-0, Virchow2 | 3 | 2169 | 21531723 |
| `tcga_5x_224` | GPFM, H-optimus-0, Virchow2 | 3 | 2169 | 21531722 |
| `cptac_5x_512` | CONCH, CONCH v1.5 | 2 | 2162 | 21531731 |
| `cptac_10x_512` | CONCH, CONCH v1.5 | 2 | 2162 | 21531732 |
| `cptac_20x_512` | CONCH, CONCH v1.5 | 2 | 2162 | 21481828 |
| `tcga_10x_512` | CONCH, CONCH v1.5 | 2 | 2169 | 21531727 |
| `tcga_20x_512` | CONCH, CONCH v1.5 | 2 | 2169 | 21531734 |
| `tcga_5x_512` | CONCH, CONCH v1.5 | 2 | **1126** | 21531726 |

`slides` is the size of the encoder intersection *after* the withholding in
`configs/excluded_slides.txt`. MUSK appears in no row: it is alone on 10x/384px,
and a one-encoder grid has no off-diagonal.

## Three things the table makes visible

**Virchow exists only at 20×.** That is why the 224px family is four encoders at
20× and three at 5× and 10× — not a sampling choice.

**`tcga_20x_256` is not the same six as `tcga_10x_256`.** CONCH v1.5 replaces
CONCH, and the intersection collapses to 1126 slides of **pure BRCA**. Together
with `tcga_5x_512`, also 1126 BRCA-only, these are a different cohort from every
other TCGA group. Two rows of a cross-grid table that both read "TCGA, 256px, 6
encoders" therefore differ in both encoder set and tissue composition. Their
figure titles carry the caveat; a table quoting them needs one too.

**The 512px groups are one model family at two versions.** CONCH against CONCH
v1.5, a single pair, which is why they read 0.83-0.89 linear CKA where six-encoder
groups read 0.5-0.65. They are not evidence that agreement is high at 512px.

## How the withholding interacts with grids

`configs/excluded_slides.txt` withholds 1074 slides — 940 TCGA-RCC and 134
CPTAC-LSCC — and the filter is applied inside `FeatureGroup.slides()`, so no
stage can forget it. But its effect is very uneven across grids, because a slide
only survives into a group if **every** encoder on that grid extracted it:

- **CPTAC-LSCC hits every CPTAC group.** All six flagship encoders hold all 134
  LSCC slides, so the intersection was 2296 before withholding and 2162 after.
  Every CPTAC result computed before the withholding is affected.
- **TCGA-RCC hits exactly one group.** Kidney was extracted for CONCH and
  CONCH v1.5 (512px) and MUSK (384px) only. At 10x/256px, KEEP holds 939 kidney
  slides but no other encoder does, so the intersection already excluded them —
  the withholding changed nothing there. `tcga_10x_512` is the single group where
  both encoders have kidney: **935 of its ~3104 slides were TCGA-RCC**.

That asymmetry is why the kidney problem stayed invisible: seventeen of eighteen
groups drop kidney on their own, and the eighteenth is a two-encoder group whose
row is easy to read past.

## What a group directory holds

```
<group>/
  similarity/matrices/     7 metrics + metric_agreement, pairwise_long
  alignment/               aligner_comparison.csv + reports/<aligner>/
  retrieval/               retrieval_summary.csv + recall_matrix_<aligner>.csv
  transfer/                transfer_summary.csv, matrix_cosine, matrix_retrieval_recall1
```

`results/groups/tcga_20x_224/alignment/README.md` documents the eleven alignment
metrics — what each one measures and which failure mode it catches. Figures are
rendered into `results/figures/groups/<label>px/` by `figs_all_groups`, which
skips any group with no results yet, so the figures fill in as jobs land.

## Caveats on reading across groups

The alignment, retrieval and transfer stages sample **500 slides** of a group
(50 000 patches, seed 0), while similarity reads the full group. Cross-stage
comparisons within a group therefore carry sampling variance that similarity
does not.

`aligner_comparison.csv` and `retrieval/retrieval_summary.csv` both report
recall@1 from **different** evaluations — on `tcga_20x_224`, Procrustes reads
0.604 in one and 0.486 in the other. Quote one and name which.
