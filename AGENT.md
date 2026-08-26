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

## ✅ Tasks to do

Prioritized remaining work to complete coverage and the result set:

1. ~~Complete the CPTAC-LSCC cohort.~~ **Dropped 2026-08-25.** LSCC is out of
   the study: only 134 of 1081 slides ever had features, which made every LSCC
   task unusable or severely imbalanced. The `cptac_nsclc` task (LUAD vs LSCC,
   1139:134) has been removed from the registry and its results deleted, so the
   downstream set is **13 tasks, not 14**. The decision was not forced by
   access — the slides download fine from TCIA pathdb, verified on
   `C3L-00081-21.svs` — it was a scope call. The 134 LSCC feature files remain
   on disk, unused.
2. **Populate the CPTAC `10x/256px` feature store** (`features_{conch_v1,
   ctranspath, gigapath, keep, resnet50, uni_v2}`) before running any
   six-encoder CPTAC analysis or the `concat` / `shared` downstream conditions.
3. **Fill `cptac_luad_kras` concat + shared.** Once task 2 holds, run
   `downstream_mil.py --task cptac_luad_kras --conditions concat shared`.
   Only the `single` condition is currently recorded for this task.
4. **Run Phase VII label-mode retrieval.**
   `cross_model_retrieval.py --mode label --task <task>`. Only `identity` mode
   has been run so far; label mode makes mAP and NDCG distinct and tests
   semantic (same-class) neighbourhood preservation.
5. **Bring the stranded encoders into the flagship comparison** by re-extracting
   at `10x/256px`: MUSK (currently `384px` only), CONCH v1.5 (`512px` only), and
   the Virchow / Virchow2 / H-optimus-0 / GPFM family (currently `224px` only).
6. **Confirm both cohorts share the same six-encoder set** at `10x/256px`
   (add KEEP to the TCGA grid if missing) so CPTAC and TCGA analyses are
   directly comparable.
7. **Regenerate all figures and result tables** (`render_figures.py`) after the
   above.

## 🟢 Already done

- Result tables for every figure are committed under `results/`, so the figures
  reproduce from this repo without rerunning the pipeline.
- Layer-wise (Phase IV) pooling sweep complete for both `cls` and `mean`.
- CTransPath and GigaPath enabled in the layer-wise analysis (five-encoder
  depth study).
- Static publication figures generated for every stage and cohort.
