# 🤖 AGENT.md

Working notes for AI agents and contributors on the MOSAIC codebase. Read this
alongside [`README.md`](README.md) (install and run) and [`PLAN.md`](PLAN.md)
(full method and research questions).

## 🧭 Orientation

- `scripts/` · entry points: `run_study.py` plus one script per stage.
- `utils/` · metrics, aligners, transfer, retrieval, MIL, feature store.
- `configs/` · encoder registry and downloaded clinical / mutation labels.
- `tests/` · property tests (mathematical invariants, not just shapes).
- `results/` · every computed table the figures are drawn from (see the map below).
- `images/figures/` · the published figures.

## 📏 Ground rules

- Reproduce everything through `scripts/run_study.py`; regenerate every figure
  with `scripts/render_figures.py`. Do not hand-edit result tables.
- **Pairing constraint:** patch encoders are only comparable within one
  `(magnification, patch_size)` grid. Never compare or pair encoders across
  grids; the unit of analysis is `(cohort, magnification, patch_size)`.
- Downstream splits are **patient-grouped**. A slide-level split leaks patients
  across folds and inflates every metric.
- **TCGA-RCC (kidney) is excluded from the study.** `configs/excluded_slides.txt`
  lists the 940 KICH/KIRC/KIRP slides and `FeatureGroup.slides()` filters them,
  so every analysis inherits it and no script opts in. The features stay on
  disk. Reason: kidney was extracted for a different encoder set — of the 12
  registered encoders it exists only for CONCH and CONCH v1.5 (10x/512px) and
  MUSK (10x/384px), and **not at all on the flagship 10x/256px grid** — so it
  could never enter the six-encoder comparison, while its presence made those
  two groups a different cohort from every other TCGA group (3104 vs 2169
  slides). To add slides back, edit that file; do not filter in a script.
- **Two MIL heads, always: ABMIL and TransMIL.** Every reported bar averages
  them, so a result is about the representation and not about one classifier.
  Per-head values live in `results.csv` and can differ by 0.03-0.04 AUC --
  check them before making a per-encoder claim. `utils/mil.py` also implements
  a mean-pool (no-attention) control; it is deliberately **not** part of the
  evaluation.
- Run `python -m pytest tests/ -q` before and after changes.

## 🔁 Reproducing the results

Everything downstream of feature extraction is seeded and deterministic, and
regenerates from the committed `results/*.csv` tables. Only Step 0 needs the raw
slides.

### Step 0 · Feature store (heavy; needs the WSIs and a trident checkout)

Extract patch and slide features into a `feature_root` (set in
`configs/encoders.yaml`). trident writes one coordinate grid per
`(magnification, patch_size)`. Then verify:

```bash
python scripts/scan_features.py --verify 2      # inventory + coordinate-pairing check
```

### Step 1 · Main pipeline (produces the flagship tables)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_study.py --out results/full_run/analysis \
    --preset full --stages inventory similarity magnification alignment transfer retrieval
CUDA_VISIBLE_DEVICES=1 python scripts/run_study.py --out results/full_run/downstream \
    --preset full --stages downstream
```

The `full` preset evaluates **all five aligners** (`joint_pca, gcca, procrustes,
mcca, autoencoder`), which is why the retrieval and alignment tables carry seven
conditions (five aligners plus the two unaligned controls). The standalone stage
scripts default to a **three-aligner** set, so always use `run_study --preset
full` to reproduce the published tables, not the individual scripts with defaults.

### Step 2 · Supplementary runs

```bash
bash scripts/run_all_groups.sh          # per-(cohort, grid) tables  -> results/groups/
bash scripts/run_all_magnifications.sh  # 5x / 10x / 20x ablation     -> results/magnification/
bash scripts/run_all_layerwise.sh       # block-wise CKA (Phase IV)   -> results/layerwise/
python scripts/slide_encoder_study.py   # slide-level encoders        -> results/slide_encoders/
bash scripts/run_musk_baseline.sh       # MUSK standalone downstream  -> results/musk_baseline/
```

### Step 3 · Figures

```bash
python scripts/render_figures.py --out results/figures --format pdf
```

Copies of the PNGs used on the project site live in `images/figures/`.
(`scripts/make_figures.py` is the original reference-style generator; the site
figures come from `render_figures.py`.)

### 🗺️ Command → table → figure map

Every figure regenerates from a committed table in `results/`:

| Figure (`images/figures/`) | Backing table (`results/`) | Produced by |
|---|---|---|
| `fig1_similarity` | `full_run/analysis/similarity/matrices/*.csv` | `run_study --stages similarity` |
| `fig2_retrieval` | `full_run/analysis/retrieval/retrieval_summary.csv` | `run_study --stages retrieval` |
| `fig3_downstream_auc` | `full_run/downstream/downstream/*/results.csv` | `run_study --stages downstream` |
| `fig4_magnification` | `magnification/cptac_benchmark_256px/magnification_summary.csv` | `run_all_magnifications.sh` |
| `fig5_transfer` | `full_run/analysis/transfer/*.csv` | `run_study --stages transfer` |
| `fig6_slide_encoders` | `slide_encoders/*/*.csv` | `slide_encoder_study.py` |
| `fig7_alignment_methods` | `full_run/analysis/alignment/aligner_comparison.csv` | `run_study --stages alignment` |
| `fig8_retrieval_tcga` | `groups/tcga_10x_256/retrieval/retrieval_summary.csv` | `run_all_groups.sh` |
| `fig9_alignment_tcga` | `groups/tcga_10x_256/alignment/aligner_comparison.csv` | `run_all_groups.sh` |
| `fig10_layerwise` | `layerwise/tcga_10x_256_{cls,mean}/matrices/*.csv` | `run_all_layerwise.sh` |

All `results/*.csv` are committed, so `render_figures.py` reproduces every figure
from this repository alone. Only the feature store (Step 0) requires the raw data.

## 📊 Status — 2026-08-27

### ✅ Finished and safe to write up

| result | where | note |
|---|---|---|
| **Downstream MIL, 13 tasks** | `results/full_run/downstream/downstream/` | 3 conditions x 2 MIL heads, all 13 complete. Unaffected by the withholdings below (each task cohort-filters to its own subcohort). |
| **fig3 downstream** | `results/figures/fig3_downstream_auc/*.pdf` | current as of 08-27 16:36. **Use the PDF, not `images/figures/fig3_downstream_auc.png`** — the site PNG is a render behind and shows LUAD KRAS with one bar instead of three. |
| **Per-subcohort similarity, patch level** | `results/full_run/analysis/similarity_by_subcohort/` | 14 of 51 runs, all CPTAC. Carries the pooling finding (below). |
| **TCGA slide-encoder similarity** | `results/slide_encoders/master_benchmark/matrices/` | 6 encoders, 2169 slides, 7 metrics. |
| **Layer-wise (fig10)** | `results/layerwise/tcga_10x_256_{cls,mean}/` | both pooling modes. |
| **TCGA magnification** | `results/magnification/master_benchmark_{224,256}px/` | TCGA groups were never affected. |

### ⏳ Pending — 12 jobs queued

| job | produces |
|---|---|
| `mos-flagship-lscc` | CPTAC similarity / alignment / transfer / retrieval, LSCC withheld |
| `mos-grp-*` (5) | the `results/groups/` trees; `tcga_10x_256` finally includes KEEP |
| `mos-mag-cptac_*` (3) | CPTAC magnification, 224/256/512px |
| `mosaic-slide-align` | both slide-encoder cohorts, all 5 aligners — similarity, alignment, transfer, retrieval |
| `mosaic-subcohort` | remaining 37 subcohort runs incl. slide encoders |
| `mos-figures` | all figures, on dependency |

Nothing CPTAC at patch level is on disk right now; the stale versions were moved to
an off-repo archive directory (649 files, 147 MB) rather than deleted, so
before/after comparison stays possible.

### 🔬 Two findings worth carrying into the write-up

**Pooling inflates the ImageNet control gap.** Linear CKA on the flagship six:

    pooled       0.199        CPTAC-COAD   0.073
    CPTAC-LUAD   0.149        CPTAC-BRCA   0.045

The pooled gap exceeds *every* subcohort's — not an average of them. Pooling barely moves
pathology-to-pathology agreement but drives the ResNet50 column down, because the control
separates tissue types more sharply than the pathology encoders do. Most of the headline gap
is between-tissue variance, and on breast the control is closer to CONCH (0.72) than CONCH is
to GigaPath (0.68). See `results/full_run/analysis/similarity_by_subcohort/README.md`.

**The TCGA slide-encoder CCA degeneracy was not sample size.** It was two unregistered
encoders. `care` and `prism2` exist only for breast slides, so requiring them in the encoder
intersection pinned TCGA to 1126 pure-BRCA slides. Excluding them gives 2169 across
BRCA/LUAD/LUSC, and PWCCA goes from 13-of-28 pairs saturated above 0.98 to 0-of-15.

### ⚠️ Known issues, not yet fixed

- `images/figures/fig3_downstream_auc.png` (the site copy) is stale — LUAD KRAS shows one bar.
  The paper PDF is correct. Regenerate before the site is used as a reference.
- fig3's `Concat` and `MOSAIC` bars are **6 encoders on CPTAC, 5 on TCGA**: KEEP is fitted into
  the CPTAC concatenation and GCCA space, so no figure-level filter reaches it. `Best single`
  *is* symmetric (KEEP excluded; it never wins a task).
- The committed TCGA slide-encoder PDF has clipped y-axis labels — rendered before the
  `col_gap` fix. The code is correct; the figure is one render behind.
- `results/full_run/analysis/magnification/` is an older flat-layout duplicate of the
  `results/magnification/<series>/` trees. Decide which is canonical.
- ~640 files show as deleted in git (the archived staleness). They refill as the jobs land;
  do not commit the deletions in the meantime.

## 🚫 Out of scope — decided, do not re-add without reversing

- **TCGA-RCC (kidney)** and **CPTAC-LSCC** — withheld study-wide via
  `configs/excluded_slides.txt` (1074 slides). Features stay on disk.
- **`cptac_nsclc`** — removed with LSCC. The downstream set is **13 tasks, not 14**;
  `tests/test_downstream.py` locks that.
- **Unregistered encoders** — `plip`, `quiltnet-b16`, `clip_rn50`, `uni_v1`, `virchow2-cls`
  (patch) and `care`, `prism2` (slide). Filtered by `configs/encoders.yaml`.
- **Optimal transport** — registered in `ALIGNER_REGISTRY`, in no preset, deliberately unrun.
- **`mean` MIL head** — implemented, deliberately not part of the evaluation.

## 🟢 Already done

- Result tables for every figure are committed under `results/`, so the figures reproduce
  from this repo without rerunning the pipeline.
- Layer-wise (Phase IV) pooling sweep complete for both `cls` and `mean`.
- CTransPath and GigaPath enabled in the layer-wise analysis.
- CPTAC `10x/256px` feature store populated (6 encoders, 2296 slides on disk, 2162 in scope).
- Both cohorts confirmed to share the same six-encoder set at `10x/256px` — but note the
  TCGA *analyses* still omit KEEP until `mos-grp-tcga_10x_256` lands.
