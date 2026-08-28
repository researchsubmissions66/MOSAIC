# Alignment stage — what is measured, and what the numbers say

`shared_latent_space.py` fits several aligners on the same held-out views and
scores each one eleven ways. `aligner_comparison.csv` is the summary, one row
per aligner; `reports/<aligner>/` holds the per-encoder detail behind it.

The reason for eleven is that no single number distinguishes a shared space that
works from one that has cheated. An aligner can satisfy its own objective by
collapsing every patch onto a blob (perfect agreement, no information), or by
preserving each view perfectly without bringing them into correspondence. The
metrics below are grouped by which failure they catch.

## The metrics

### Round-trip fidelity — `reports/<aligner>/reconstruction.csv`

Each view goes into the shared space and back out; the reconstruction is scored
against the original. Per encoder, three columns
(`utils/alignment_metrics.py:52-92`):

| column | definition | reading |
|---|---|---|
| `nmse` | `‖X − X̂‖² / ‖X − mean(X)‖²` | below 1 beats predicting the mean; above 1 is worse than useless |
| `r2` | `1 − nmse` | fraction of the encoder's variance the shared space retains |
| `cosine` | mean cosine between original and reconstructed rows | direction fidelity — what cosine retrieval actually depends on |

**`r2` and `nmse` are the same measurement.** Line 91 computes `r2` as the
literal subtraction `1.0 - nmse`, not as an independently derived statistic.
Across the 25 rows in this folder, `max |nmse + r2 − 1| = 0.000e+00`. Reporting
both as separate columns states one quantity twice, and invites a reader to
cross-check two numbers that cannot disagree. Report **R² and cosine**, or state
the identity explicitly.

R² and cosine *do* carry different information, which is why the pair is worth
keeping: R² is scale-sensitive and cosine is not, so a shared space that gets
directions right while losing magnitude scores high on one and low on the other.
Prov-GigaPath and UNI2-h — the two 1536-d encoders — sit lowest on both in every
aligner, on both cohorts. Reconstruction fidelity tracks **dimensionality**, not
model quality: the shared space is 64-d, and 1536 → 64 → 1536 discards more than
512 → 64 → 512 does.

### Alignment quality

| column | definition | reading |
|---|---|---|
| `alignment_error` | mean squared spread of the views' projections of the *same* patch, over the consensus variance (`:101-125`) | 0 is perfect agreement; 1 means cross-model disagreement is as large as the spread between patches, i.e. no usable patch identity |
| `paired_cosine` | mean cosine between two views' projections of the same patch, **mean-centred** (`:131-153`) | centring is load-bearing: uncentred saturates at 1.000 on spaces whose cross-model recall@1 is 0.5, because a large common component dominates every row |
| `shared_cka` | linear CKA between views *inside* the shared space (`:240-257`) | agreement of geometry rather than of individual points |
| `recall@1`, `recall@10`, `mrr` | retrieve each query patch from a database built with a **different** encoder (`:169-206`) | the end-to-end test: does a patch land next to itself across models |

### Collapse detectors

An aligner can score well on everything above and still have destroyed what
makes the embeddings useful. Two independent checks:

| column | definition | reading |
|---|---|---|
| `neighborhood_preservation` | fraction of each patch's k=10 nearest neighbours surviving the projection (`:267-297`) | chance is ≈ k/n. Catches loss of *local* morphological structure |
| `effective_rank` | exp(entropy of the normalised singular-value spectrum) (`:325-348`, Roy & Vetterli 2007) | in [1, latent_dim]. Well below `latent_dim` means the space has partially collapsed |

`latent_dim` is 64 throughout, so `effective_rank` is directly readable as
"dimensions actually used out of 64".

## Results

**TCGA · 20× · 224px** — GPFM, H-optimus-0, Virchow, Virchow2; 2169 slides.

| aligner | recon R² | recon cos | align err | paired cos | shared CKA | recall@1 | nbhd pres | eff rank |
|---|---|---|---|---|---|---|---|---|
| joint_pca | 0.540 | 0.771 | 0.037 | 0.950 | 0.928 | 0.981 | 0.544 | 56.8 |
| gcca | 0.469 | 0.739 | **0.030** | **0.962** | 0.927 | **0.997** | 0.465 | **63.9** |
| mcca | 0.510 | 0.760 | 0.154 | 0.823 | 0.693 | 0.952 | 0.535 | 63.9 |
| procrustes | **0.588** | **0.795** | 0.335 | 0.670 | 0.459 | 0.604 | **0.744** | 57.0 |
| autoencoder | 0.570 | 0.787 | 0.064 | 0.917 | **0.951** | 0.974 | 0.579 | 55.4 |

**CPTAC · 10× · 256px** — the six flagship encoders.

| aligner | recon R² | recon cos | align err | paired cos | shared CKA | recall@1 | nbhd pres | eff rank |
|---|---|---|---|---|---|---|---|---|
| joint_pca | 0.701 | 0.889 | **0.073** | **0.916** | 0.947 | 0.944 | 0.565 | 51.9 |
| gcca | 0.660 | 0.874 | 0.158 | 0.831 | 0.730 | **0.958** | 0.511 | **63.7** |
| mcca | 0.709 | 0.893 | 0.392 | 0.660 | 0.486 | 0.759 | 0.556 | 63.5 |
| procrustes | **0.775** | **0.914** | 0.401 | 0.659 | 0.611 | 0.418 | **0.849** | 53.4 |
| autoencoder | 0.752 | 0.906 | 0.241 | 0.766 | 0.874 | 0.702 | 0.655 | 54.8 |

## What the two tables agree on

**Reconstruction and alignment trade off against each other, and the trade is
not cohort-specific.** Procrustes is best on reconstruction (R² and cosine) and
worst on alignment (error, paired cosine, recall@1) in *both* cohorts, across
disjoint encoder sets. It is the only rigid aligner in the set — it may rotate
and scale a view but not warp it, so it preserves each view's internal geometry
and pays for it in correspondence.

**Procrustes preserves local neighbourhoods best while aligning worst** — 0.744
and 0.849, far above every other aligner in both cohorts. This is the sharpest
form of the trade-off: the aligner that best keeps each model's own structure is
the one that least brings the models together. Optimising reconstruction alone
does not produce a shared space.

**GCCA wins retrieval while losing reconstruction**, and uses nearly the whole
latent space to do it (effective rank 63.9 and 63.7 of 64, against 51.9-57.0 for
the rest). It spends capacity on correspondence rather than round-trip fidelity.

The one disagreement between cohorts is joint_pca vs gcca on alignment error —
gcca is better on TCGA (0.030 vs 0.037), joint_pca on CPTAC (0.073 vs 0.158).
The ranking of those two is not stable; the Procrustes result is.

## Two different recall@1 numbers exist — do not conflate them

`aligner_comparison.csv` and `../retrieval/retrieval_summary.csv` both report
recall@1, from different evaluations. On this group, Procrustes reads **0.604**
in the first and **0.486** in the second; GCCA reads 0.997 and 0.995. The
alignment-stage figure is cross-model retrieval inside `evaluate_aligner` on the
held-out views; the retrieval-stage figure comes from `cross_model_retrieval.py`
with 5000 sampled queries and its own protocol. Quote one, name which.

## Run parameters

500 slides, 50 000 patches, `latent_dim` 64, seed 0, five aligners
(joint_pca, gcca, mcca, procrustes, autoencoder); the autoencoder additionally
uses 4000 max-patches and 80 epochs. `--max-slides 500` means the stage samples
500 slides of the group rather than reading all of them, so these numbers carry
sampling variance that the similarity stage — which reads the full group — does
not.

## Provenance

This folder: TCGA 20×/224px, computed 2026-08-25 15:11. The four encoders here
each hold exactly 2169 slides with **zero** TCGA-RCC among them, so the kidney
withholding does not touch this group and the run date does not matter.

The CPTAC table above is read from the archived flagship run of 2026-08-27
10:11, which is not in this repository. `mos-flagship-lscc` regenerates it; when
that lands, compare the two — the CPTAC group does carry all 134 CPTAC-LSCC
slides in all six encoders, so any change between the archived and regenerated
numbers is the measure of what withholding LSCC did to this stage.
