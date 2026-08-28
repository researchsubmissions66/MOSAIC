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
| `fig4b_magnification_tcga` | `magnification/master_benchmark_256px/magnification_summary.csv` | `run_all_magnifications.sh` |
| `fig4c_magnification_tcga_224px` | `magnification/master_benchmark_224px/magnification_summary.csv` | `run_all_magnifications.sh` |
| `fig5_transfer` | `full_run/analysis/transfer/*.csv` | `run_study --stages transfer` |
| `fig6_slide_encoders` | `slide_encoders/*/*.csv` | `slide_encoder_study.py` |
| `fig7_alignment_methods` | `full_run/analysis/alignment/aligner_comparison.csv` | `run_study --stages alignment` |
| `fig8_retrieval_tcga` | `groups/tcga_10x_256/retrieval/retrieval_summary.csv` | `run_all_groups.sh` |
| `fig9_alignment_tcga` | `groups/tcga_10x_256/alignment/aligner_comparison.csv` | `run_all_groups.sh` |
| `fig10_layerwise` | `layerwise/tcga_10x_256_{cls,mean}/matrices/*.csv` | `run_all_layerwise.sh` |

All `results/*.csv` are committed, so `render_figures.py` reproduces every figure
from this repository alone. Only the feature store (Step 0) requires the raw data.

### 📚 Reference documents

Written to sit next to the data they describe rather than in this file:

| document | covers |
|---|---|
| `results/groups/README.md` | why the group is the unit of analysis; all **18** grids with encoder sets, slide counts and job IDs; how the withholding lands unevenly across them |
| `results/groups/tcga_20x_224/alignment/README.md` | the **11** alignment metrics — definitions, line refs, which failure each catches |
| `results/figures/groups/tcga_20x_224px/README.md` | every number behind that group's four figures, and the two-block metric finding |
| `results/figures/FIGURE_INVENTORY.md` | cohort, grid and encoder set for every figure, original ones included |
| `results/figures/similarity_by_grid/README.md` | the all-grids similarity table |
| `results/full_run/analysis/similarity_by_subcohort/README.md` | the pooling finding |

## 📊 Status — 2026-08-28

### ✅ Finished and safe to write up

| result | where | note |
|---|---|---|
| **CPTAC flagship, all 4 stages** | `results/full_run/analysis/{similarity,alignment,transfer,retrieval}/` | recomputed post-withholding 08-28 06:45-08:11, 6 encoders. fig1/2/5/7 current. |
| **Downstream MIL, 13 tasks** | `results/full_run/downstream/downstream/` | 3 conditions x 2 MIL heads. Unaffected by the withholdings (each task cohort-filters to its own subcohort). |
| **Magnification, both cohorts, all 3 grids** | `results/magnification/` | CPTAC 224/256/512px landed 08-27/28. **fig4 is no longer provisional.** |
| **Slide encoders, both cohorts, all 4 stages** | `results/slide_encoders/{cptac,master}_benchmark/` | 6 encoders, 5 aligners, 2162 / 2169 slides. |
| **11 of 18 groups, all 4 stages** | `results/groups/` | see the table in `results/groups/README.md` |
| **Layer-wise (fig10)** | `results/layerwise/tcga_10x_256_{cls,mean}/` | both pooling modes. |
| **Per-subcohort similarity** | `results/full_run/analysis/similarity_by_subcohort/` | 14 of 15 CPTAC cells; carries the pooling finding. |
| **All figures** | `results/figures/` | 88 re-rendered 08-28 08:52, in the restored rounded-tile style. |

### ⏳ Pending

| job | state | produces |
|---|---|---|
| `mos-grp-tcga_5x_256` · `tcga_20x_256` · `cptac_5x_256` | running | the last three incomplete groups |
| `mos-dn-tcga_20x_224` · `mos-dn-cptac_20x_512` | running | downstream on non-flagship grids -> `results/downstream_other/` |
| `21544821/2/3` | queued | the retrieval/transfer stages that timed out on three CPTAC groups |
| `21544865` | queued | CPTAC 20x/224 downstream, resumable (skips finished tasks) |
| `21502445 mosaic-subcohort` | queued | the last subcohort cell + slide-encoder subcohorts |

**Time limits were the main failure mode overnight**: five jobs hit TIMEOUT, none
crashed. When requeueing a partial group, pass only the missing `--stages` — a
full rerun repeats hours of completed work. Check which stages exist first.

### 🔬 Findings worth carrying into the write-up

**The seven similarity metrics form two blocks that rank encoder pairs in
opposite orders.** On TCGA 20x/224px, `metric_agreement.csv` gives Linear CKA,
Kernel CKA, Cosine RSA and Distance Correlation mutual Spearman **ρ = 1.000** —
four formulas, one ranking. PWCCA anti-correlates with all four (ρ = −0.371) and
with Procrustes (ρ = −0.486). The geometry block calls GPFM↔H-optimus-0 the most
similar pair and Virchow↔Virchow2 the least; PWCCA reverses it. Virchow and
Virchow2 — same lab, same architecture, same dimension, differing only in
pretraining scale — are the *least* similar pair by four metrics. Reporting one
metric family silently picks a conclusion.

**Pooling inflates the ImageNet control gap.** Linear CKA on the flagship six:

    pooled       0.199        CPTAC-COAD   0.073
    CPTAC-LUAD   0.149        CPTAC-BRCA   0.045

The pooled gap exceeds *every* subcohort's — not an average of them. Most of the
headline gap is between-tissue variance: the control separates tissue types more
sharply than the pathology encoders do.

**Reconstruction and alignment trade off, and it replicates across cohorts.**
Procrustes — the only rigid aligner — is best on reconstruction and worst on
alignment on both CPTAC (6 encoders) and TCGA 20x/224 (4 disjoint encoders), and
best on neighbourhood preservation while worst on correspondence. GCCA inverts
the trade and uses nearly all 64 latent dimensions (effective rank 63.9).

**Cross-model transfer is 58% of the self round-trip** (recall@1 0.470 / 0.807 on
TCGA 20x/224). Roughly four tenths of the apparent cross-model penalty is the
64-d bottleneck, not the translation. Reporting cross-model transfer without the
round-trip overstates it.

**The TCGA slide-encoder CCA degeneracy was not sample size.** `care` and
`prism2` exist only for breast slides, so requiring them pinned TCGA to 1126
pure-BRCA slides. Excluding them gives 2169 across BRCA/LUAD/LUSC, and PWCCA goes
from 13-of-28 pairs saturated above 0.98 to 0-of-15.

**The withholding lands very unevenly, which is why kidney went unnoticed.** A
slide enters a group only if *every* encoder on that grid extracted it.
CPTAC-LSCC is in all six flagship encoders, so it affected every CPTAC group.
TCGA-RCC reached exactly **one** of eighteen groups (`tcga_10x_512`, 935 of ~3104
slides): at 10x/256px KEEP holds 939 kidney slides but no other encoder does, so
the intersection already dropped them. Seventeen groups excluded kidney on their
own.

### ⚠️ Known issues, not yet fixed

- `images/figures/fig3_downstream_auc.png` (site copy, 08-27 21:43) is behind the
  PDF (08-28 08:52). Regenerate before the site is used as a reference.
- fig3's `Concat` and `MOSAIC` bars are **6 encoders on CPTAC, 5 on TCGA**: KEEP
  is fitted into the CPTAC concatenation and GCCA space, so no figure-level
  filter reaches it. `Best single` *is* symmetric (KEEP excluded; it never wins).
  Refitting CPTAC without KEEP is a cheap downstream rerun if the comparison
  needs to be like-for-like.
- `results/full_run/analysis/magnification/` is an older flat-layout duplicate of
  the `results/magnification/<series>/` trees. Decide which is canonical.
- ~210 files still show as deleted in git (down from ~640 — the archived
  staleness refilling as jobs land). Do not commit the deletions.
- `scripts/mosaic_*.sh` are gitignored on purpose: they carry the cluster account
  and absolute paths. Keep launchers there, not in tracked scripts.
- `downstream_mil.py` takes **one `--task` per invocation** and writes only that
  invocation's rows to `results.csv`. Loop over tasks in the wrapper with a
  per-task `--out`; a group-level invocation with no task errors out.

## 🚫 Out of scope — decided, do not re-add without reversing

- **TCGA-RCC (kidney)** and **CPTAC-LSCC** — withheld study-wide via
  `configs/excluded_slides.txt` (1074 slides). Features stay on disk.
- **`cptac_nsclc`** — removed with LSCC. The downstream set is **13 tasks, not 14**;
  `tests/test_downstream.py` locks that.
- **Unregistered encoders** — `plip`, `quiltnet-b16`, `clip_rn50`, `uni_v1`, `virchow2-cls`
  (patch) and `care`, `prism2` (slide). Filtered by `configs/encoders.yaml`.
  Note `plip`, `quiltnet-b16` and `clip_rn50` exist at 224px for **TCGA only** —
  registering them would give the cohorts different encoder sets, which is the
  defect that got `care`/`prism2` excluded.
- **Optimal transport** — registered in `ALIGNER_REGISTRY`, in no preset, deliberately unrun.
- **`mean` MIL head** — implemented, deliberately not part of the evaluation.
