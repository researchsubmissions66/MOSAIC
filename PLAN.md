# Towards a Universal Latent Space for Computational Pathology Foundation Models

**Status:** Phases I, IV, V, VI, VII, VIII and the magnification ablation implemented and running
**Location:** `~/pfm_latent_space` (host: `alt`)

### Alternative titles

- Do Computational Pathology Foundation Models Learn a Common Morphological Representation?
- Learning a Shared Latent Space Across Computational Pathology Foundation Models
- Foundation Model Interoperability in Computational Pathology via Representation Alignment

---

## 1. Motivation

Recent pathology foundation models (PFMs) — UNI, CONCH, KEEP, MUSK, Virchow, Phikon, GigaPath, and others — differ along every axis that is normally assumed to matter:

- architecture
- pretraining objective
- pretraining dataset
- multimodal supervision

Despite these differences, many reach comparable downstream performance. This raises a fundamental question:

> **Do independently trained pathology foundation models converge to a common latent representation of tissue morphology?**

Existing work compares these models almost exclusively through downstream metrics (AUC, accuracy, F1). Very little is known about the relationships *between their internal representation spaces*. This project sets out to characterize, align, and exploit those spaces.

---

## 2. Central hypothesis

**Independent pathology foundation models learn different coordinate systems over a shared underlying morphological manifold.**

If the hypothesis holds, then:

- representations should be alignable across models,
- semantic neighborhoods should be preserved under alignment,
- features should transfer between models,
- downstream classifiers should be able to operate inside a single shared latent space.

---

## 3. Research questions

| # | Question |
|---|---|
| RQ1 | How similar are representation spaces across pathology foundation models? |
| RQ2 | Do different pretraining objectives produce similar latent geometry? |
| RQ3 | Can multiple PFMs be aligned into a shared latent representation? |
| RQ4 | Does the aligned representation improve downstream learning? |
| RQ5 | Can a downstream model trained on one encoder generalize to another encoder after alignment? |

---

## 4. Datasets

Two cohorts, features extracted with [trident](https://github.com/mahmoodlab/trident):

| Cohort | Store key | Slides | Contents |
|---|---|---|---|
| TCGA | `master_benchmark` | 2169 | BRCA 1126, LUAD 531, LUSC 512 |
| CPTAC | `cptac_benchmark` | 2296 | LUAD 1139, BRCA 654, COAD 369 (LSCC 134, dropped) |

**Key requirement:** every patch must have embeddings from all models compared.
Trident writes one coordinate grid per `(magnification, patch_size)`, so
encoders sharing a grid are row-paired by index and encoders on different grids
are not. That makes `(cohort, magnification, patch_size)` the unit of analysis;
the widest set is `cptac_benchmark/10x_256px` with **6 encoders over 2296
slides**.

---

## 5. Models

Twelve patch encoders are extracted; the registry with dimensions, HF ids and
pretraining objectives is `configs/encoders.yaml`.

| Family | Encoders |
|---|---|
| vision SSL | uni_v2, gigapath, virchow, virchow2, hoptimus0, gpfm, ctranspath |
| vision-language | conch_v1, conch_v15, keep, musk |
| supervised control | resnet50 |

**ResNet50 is the control** — ImageNet-supervised, never saw a slide. Any claim
of a shared morphological manifold must show the pathology models agree with
each other substantially more than with it.

---

## Phase I — Representation similarity

> **Status: implemented.** All seven metrics and the three outputs live in
> `utils/`, driven by `scripts/representation_similarity.py`. See [README.md](README.md).

**Goal:** quantify how similar different PFMs are.

**Methods**

- Linear CKA
- Kernel CKA
- SVCCA
- PWCCA
- Orthogonal Procrustes
- Cosine similarity
- Distance correlation

**Outputs**

- Pairwise similarity matrix (N × N heatmap)
- Hierarchical clustering of models
- MDS / UMAP visualization of the model space

**Expected insights**

- Which models are closest to one another?
- Does the training objective dominate similarity structure?
- Or does the architecture dominate?

---

## Phase IV — Layer-wise alignment

> **Status: implemented.** `utils/layers.py`, driven by
> `scripts/layerwise_alignment.py`.
>
> **This is the one stage that does not read the feature store.** Trident saved
> only final pooled embeddings, so intermediate activations had to be recreated
> by re-running the models with forward hooks — which needs slide images and
> model weights. Weights were already cached locally; slides were not, so
> `scripts/download_slides.py` fetches a few small TCGA slides (~77 MB for four)
> that are **already in the feature store**, and patches are re-cropped at the
> coordinates trident recorded. Using the store's own coordinates keeps the
> analysis on the same tissue as every other phase.
>
> Outputs: an `L_a x L_b` CKA matrix per model pair with the lockstep diagonal
> overlaid, best-match alignment trajectories against relative depth (relative,
> because models differ in block count), a divergence-depth profile, and a
> within-model depth reference. `--pool {cls,mean,cls_mean}` selects how each
> block's tokens are reduced; it materially changes the answer, so sweep it.

Compare *every transformer block*, not only the final embedding.

**Questions**

- Where in the network do models begin to diverge?
- Do early layers encode universal morphology?
- Are late layers model-specific?

**Deliverables**

- Layer-by-layer similarity heatmaps
- Alignment trajectories across depth

---

## Phase V — Shared latent space  *(core methodological contribution)*

> **Status: implemented.** All six approaches live in `utils/alignment/`
> behind one encode/decode interface, with evaluation in
> `utils/alignment_metrics.py` and a driver at `scripts/shared_latent_space.py`.
> See [README.md](README.md). Caveat: the *unsupervised* optimal-transport
> mode is exploratory and does not currently work reliably; supervised OT does.

**Goal:** construct a common latent representation shared across multiple PFMs.

**Candidate approaches**

- Generalized Canonical Correlation Analysis (GCCA)
- Multi-view CCA
- Orthogonal Procrustes
- Joint PCA
- Shared latent autoencoder
- Optimal transport alignment

**Outputs**

- The common embedding space itself
- Per-model projection functions (into and out of the shared space)
- Reconstruction error
- Alignment quality metrics

---

## Phase VI — Cross-model transfer

> **Status: implemented.** `utils/transfer.py`, driven by
> `scripts/cross_model_transfer.py`. Four evaluations in increasing strictness:
> cosine/R² fidelity, retrieval against the target's **real** index in its
> native space, and a linear probe **trained on the target's real embeddings**
> and tested on translated ones. Each encoder's self-round-trip is reported as
> the ceiling, isolating the cost of the shared space from the cost of crossing
> models.
>
> **Scope: every ordered pair within a pairable set.** Transfer needs row-paired
> patches, so it runs within a feature group (one coordinate grid), all ordered
> pairs at once — not a hand-picked list. The driver defaults to exactly that;
> `--pairs src:tgt` narrows it if wanted, and `--list-pairs` enumerates them:
>
> | Set | Ordered pairs | Encoders |
> |---|---|---|
> | `cptac_benchmark/10x_256px` | **30** | conch_v1, ctranspath, gigapath, keep, resnet50, uni_v2 |
> | `{cptac,master}/20x_224px` | 12 | gpfm, hoptimus0, virchow, virchow2 |
> | `master_benchmark/10x_256px` | 20 | conch_v1, ctranspath, gigapath, resnet50, uni_v2 |
> | `*/{5,20}x_256px` | 20 | ctranspath, gigapath, keep, resnet50, uni_v2 |
> | `*/512px` | 2 | conch_v1, conch_v15 |
>
> Cross-*set* pairs (e.g. KEEP@256px → MUSK@384px) are not defined without
> re-extracting one encoder onto the other's grid.

**Question:** can one model's representation be converted into another's?

**Experiments:** all ordered source → target pairs within each pairable set.
(The plan's original CONCH → UNI, KEEP → MUSK, Virchow → CONCH were
illustrative; the first is covered by the 10x/256px set, the latter two span
different grids.)

**Evaluation:** cosine similarity, retrieval, linear-probe accuracy, feature reconstruction.

---

## Phase VII — Cross-model retrieval

> **Status: implemented.** `utils/retrieval.py`, driven by
> `scripts/cross_model_retrieval.py`. Two relevance modes: *identity* (retrieve
> the same patch — mAP collapses to MRR by definition there) and *label*
> (retrieve any patch of the same class, which makes mAP and NDCG distinct).
> The unaligned control is per-model independent PCA to the same dimension.

Build the database with **model A**, issue queries with **model B**.

- Without alignment → baseline retrieval
- With alignment → cross-model retrieval

**Metrics:** Recall@K, mAP, MRR, NDCG.

---

## Phase VIII — Downstream learning

> **Status: implemented.** `utils/mil.py` (ABMIL and TransMIL are the two
> evaluated heads; a mean-pool control exists but is not part of the study),
> `utils/bags.py` (the three input conditions) and `utils/labels.py` (14 tasks),
> driven by `scripts/downstream_mil.py`. Splits are patient-grouped, and the
> aligner for the shared condition is fitted on training slides only.
>
> **Tasks** — all confined to a single cancer type; tissue-of-origin tasks are
> excluded because they saturate and cannot rank encoders:
> `tcga_nsclc` (LUAD/LUSC, 1043 slides), `tcga_brca_subtype` (IDC/ILC, 958),
> `tcga_brca_stage` (1012), `tcga_nsclc_stage` (831),
> plus 9 CPTAC mutation tasks — BRCA (PIK3CA/MAP3K1/GATA3, 377 slides),
> COAD (KRAS/PIK3CA/TP53, 223), LUAD (TP53/STK11/KRAS, 1058).
> **CPTAC-LSCC is dropped from the study entirely** (2026-08-25). Only 134 of
> 1081 slides were ever extracted, leaving 28 patients — too few for the LSCC
> mutation tasks, and the reason `cptac_nsclc` ran 1139:134. That task has been
> removed from the registry too, so the downstream set is **13 tasks**.

Keep the downstream head simple. **Two heads are evaluated — ABMIL and
TransMIL — and every reported bar averages them**, so a result is a statement
about the representation rather than about one classifier. Per-head values stay
in `results.csv`, and they can differ by 0.03-0.04 AUC, so check them before
making any per-encoder claim.

**Conditions**

1. Baseline: single encoder → ABMIL + TransMIL
2. Concatenation of encoders → ABMIL + TransMIL
3. Shared latent space → ABMIL + TransMIL

**Metrics:** AUC, accuracy, F1, balanced accuracy.

**Done:** TransMIL is the second head, so the findings are not an ABMIL
artefact. A mean-pool (no-attention) control is implemented in `utils/mil.py`
but deliberately excluded from the evaluation.

---

## Ablation studies

- **Magnification (5x / 10x / 20x)** — *implemented*, `scripts/magnification_ablation.py`.
  The same experiment repeated at each resolution: same encoders, same slides,
  same seed. Each magnification is an independent replication — the coordinate
  grids differ, so patches are *not* paired across magnifications; what is
  compared is the result (the similarity matrix, the alignment quality), never
  the embeddings.

Latent dimensionality, alignment method, encoder count, training-set size and
retrieval metric are all exposed as CLI flags on the existing drivers
(`--latent-dim`, `--methods`, `--encoders`, `--n-patches`, `--max-slides`), so
sweeping them needs no new code.

---

## Expected contributions

1. The first large-scale representational similarity analysis across computational pathology foundation models.
2. A framework for aligning PFM embeddings into a shared latent space.
3. A systematic study of how pretraining objectives shape morphological representations.
4. A demonstration of cross-model interoperability for retrieval and downstream learning.
5. Evidence supporting — or refuting — the existence of a common latent morphology manifold.

---

## Open items

- ~~Extract the rest of CPTAC-LSCC.~~ **Closed 2026-08-25: LSCC dropped.** The
  slides were reachable — they download from TCIA pathdb, verified end to end on
  `C3L-00081-21.svs` (183 MB, opens in openslide at 20x) — so this was a scope
  decision, not a blocker. `cptac_nsclc` is gone with it.
- **Re-extract MUSK and CONCH v1.5 at 256px.** MUSK is stranded alone at 384px
  and CONCH v1.5 only pairs with CONCH v1 at 512px, so neither can enter the
  main six-encoder comparison.
- **Sweep `--pool` in the layer-wise analysis.** Only CLS pooling has been run;
  `mean` follows the spatial pathway and may change the conclusion.
