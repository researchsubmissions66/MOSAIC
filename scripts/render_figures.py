"""Render the polished MOSAIC figure set from results on disk.

Produces a small, cohesive suite in the reference visual language — heatmap rows
with per-panel colourbars, grouped bars with in-bar labels — from the CSVs the
study wrote. Aesthetics are tuned for a compact, publication-grade look.

    python scripts/render_figures.py --out results/figures --format pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paperfigs import (  # noqa: E402
    apply_style,
    clean_label,
    clean_labels,
    grouped_bars,
    heatmap_row,
    save_plot,
)

RESULTS = Path("results")

#: All seven metrics the study computes. Figures previously showed only
#: four (linear/kernel CKA, SVCCA, Procrustes) while the CSVs carried
#: seven -- which omitted PWCCA, the one metric that disagrees with the
#: others (Spearman 0.175 vs Procrustes on the flagship, and negative
#: against CKA on the 224px group). The dissenting metric should be
#: visible, not filtered out.
METRICS = [
    "linear_cka",
    "kernel_cka",
    "svcca",
    "pwcca",
    "procrustes",
    "cosine_rsa",
    "distance_correlation",
]
ALIGN_ORDER = ["unaligned_pca", "procrustes", "autoencoder", "mcca", "joint_pca", "gcca"]
TASK_TITLES = {
    "tcga_nsclc": "TCGA\nNSCLC",
    "tcga_brca_subtype": "TCGA BRCA\nIDC/ILC",
    "tcga_brca_stage": "TCGA BRCA\nStage",
    "tcga_nsclc_stage": "TCGA NSCLC\nStage",
    "cptac_luad_tp53": "LUAD\nTP53",
    "cptac_luad_kras": "LUAD\nKRAS",
    "cptac_luad_stk11": "LUAD\nSTK11",
    "cptac_brca_pik3ca": "BRCA\nPIK3CA",
    "cptac_brca_gata3": "BRCA\nGATA3",
    "cptac_brca_map3k1": "BRCA\nMAP3K1",
    "cptac_coad_tp53": "COAD\nTP53",
    "cptac_coad_kras": "COAD\nKRAS",
    "cptac_coad_pik3ca": "COAD\nPIK3CA",
}
TASK_ORDER = [
    "tcga_nsclc", "tcga_brca_subtype",
    "cptac_luad_tp53", "cptac_luad_kras", "cptac_luad_stk11",
    "cptac_brca_pik3ca", "cptac_brca_gata3", "cptac_brca_map3k1",
    "cptac_coad_tp53", "cptac_coad_kras", "cptac_coad_pik3ca",
    "tcga_brca_stage", "tcga_nsclc_stage",
]


def _load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = clean_labels(df.index)
    df.columns = clean_labels(df.columns)
    return df


def _legend_below(fig, ax, highlight: str | None = None, y: float = -0.02) -> None:
    """Move a grouped-bar axes legend to a clean horizontal row below the plot.

    In-plot legends collide with tall bars; placing it under the axis keeps the
    data unobstructed.
    """
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    leg = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
        ncol=len(labels), frameon=False, fontsize=9.5, handlelength=1.3,
        columnspacing=1.8,
    )
    if highlight:
        for t in leg.get_texts():
            if t.get_text() == highlight:
                t.set_fontweight("bold")


#: Per-figure caption and computation notes. Written to a markdown file that
#: sits beside each figure, so the figure and its provenance travel together.
FIG_META: dict[str, dict[str, str]] = {
    "fig1_similarity": {
        "title": "Representational similarity across pathology foundation models",
        "caption": (
            "Pairwise similarity between the six encoders of the flagship group "
            "(`cptac_benchmark/10x_256px`), one panel per metric. Each panel "
            "carries its own colour scale — the metrics sit at different levels, "
            "and a shared scale would flatten all but the widest-ranging. Only "
            "the lower triangle is shown (the matrices are symmetric). ResNet50, "
            "the ImageNet-supervised control, is consistently the least similar "
            "to the pathology encoders. All seven metrics are shown: PWCCA "
            "ranks the pairs differently from the others (Spearman 0.175 "
            "against Procrustes), so reading any single metric as *the* "
            "similarity overstates how settled the picture is."
        ),
        "source": "results/full_run/analysis/similarity/matrices/*.csv",
        "methods": (
            "- **Encoders (6):** CONCH, CTransPath, Prov-GigaPath, KEEP, "
            "ResNet50, UNI2 — the only patch encoders sharing the 10x/256px "
            "coordinate grid, hence row-paired by patch index.\n"
            "- **Data:** 50,000 patches sampled across ≤500 CPTAC slides "
            "(shared subsample, seed 0); the O(n²) metrics subsample 8,000.\n"
            "- **Metrics (7):** linear CKA (feature-space form), RBF-kernel CKA "
            "(median-heuristic bandwidth), SVCCA (99% variance retained), "
            "PWCCA, normalised orthogonal Procrustes, cosine RSA and "
            "distance correlation. `metric_agreement.csv` holds the Spearman "
            "agreement between them; distance correlation is redundant with "
            "kernel CKA (1.000) while PWCCA is the outlier.\n"
            "- Every matrix is computed on column-centered features; the "
            "diagonal is 1.0 by construction."
        ),
    },
    "fig2_retrieval": {
        "title": "Cross-model retrieval in the shared latent space",
        "caption": (
            "On the flagship CPTAC · 10× · 256px group (6 encoders). Patch-level "
            "retrieval where the database is indexed with one "
            "encoder and queried with another, after both are mapped into a "
            "shared space. Bars are alignment methods (darkest = GCCA); the "
            "unaligned control sits at ~0. Aligned methods lift cross-model "
            "retrieval from chance (~0.001) to Recall@1 ≈ 0.93, and a rigid "
            "rotation (Procrustes) is clearly insufficient."
        ),
        "source": "results/full_run/analysis/retrieval/retrieval_summary.csv",
        "methods": (
            "- **Task:** identity retrieval — the one correct answer for a "
            "query patch is that same patch encoded by the database model. "
            "Chance Recall@1 = 1 / database size.\n"
            "- **Split:** aligners are fit on a training patch split; every "
            "score is computed on held-out patches (30% test).\n"
            "- **Conditions:** GCCA, joint PCA, MCCA, shared autoencoder, and "
            "generalized Procrustes, each at latent dim 64, plus two unaligned "
            "controls (per-model independent PCA / truncation to 64 dims) that "
            "isolate alignment from dimensionality reduction.\n"
            "- **Metrics:** Recall@{1,5,10}, mAP, NDCG. For identity retrieval "
            "mAP equals MRR by definition, so MRR is omitted here.\n"
            "- Values are means over all ordered encoder pairs (same-model "
            "pairs excluded)."
        ),
    },
    "fig3_downstream_auc": {
        "title": "Downstream MIL — single encoder vs. concatenation vs. shared space",
        "caption": (
            "Slide-level prediction under three input representations, split "
            "into the two difficulty regimes. Top: morphological/clinical tasks "
            "saturate near AUC 0.97 and the three representations are "
            "indistinguishable. Bottom: molecular (mutation) tasks are where "
            "representations separate — MOSAIC (the shared space) wins on some "
            "(BRCA GATA3, COAD PIK3CA) and loses on others (LUAD/COAD TP53). The "
            "dashed line is chance (AUC 0.5)."
        ),
        "source": "results/full_run/downstream/downstream/*/results.csv",
        "methods": (
            "- **Classifier held constant across all three bars:** attention-"
            "MIL over a bag of patch embeddings → slide label. Only the *input "
            "representation* changes.\n"
            "  - **Best single** — one encoder's patches; a separate MIL is "
            "trained per encoder and the bar reports the best of the 6.\n"
            "  - **Concat** — all 6 encoders concatenated per patch "
            "(dim ≈ 5000); strictly more information than the shared space.\n"
            "  - **MOSAIC** — the 6 encoders mapped through a GCCA aligner into "
            "a 64-d shared space; the aligner is fit on **training slides "
            "only**.\n"
            "- **MIL heads:** ABMIL and TransMIL, 80 epochs, ≤4000 patches/bag; "
            "each bar averages the two heads (per-head values are in the CSV).\n"
            "- **Splits:** patient-grouped and stratified — no patient appears "
            "in both train and test (TCGA/CPTAC have several slides per "
            "patient).\n"
            "- **Metric:** macro/binary AUC. Every task is imbalanced, so AUC "
            "and balanced accuracy are the metrics to read; BRCA MAP3K1 (8% "
            "positive) is near-degenerate and its sub-chance values are noise."
        ),
    },
    "fig4_magnification": {
        "title": "Representational agreement decreases with magnification",
        "caption": (
            "The same similarity analysis repeated independently at 5×, 10× and "
            "20× (CPTAC, 256px, the five encoders common to all three "
            "magnifications). Every one of the seven metrics declines "
            "monotonically as resolution increases: models agree most on coarse "
            "tissue architecture (5×) and diverge on fine nuclear detail (20×). "
            "Lines and legend are ordered by their value at 20×."
        ),
        "source": "results/magnification/cptac_benchmark_256px/magnification_summary.csv",
        "methods": (
            "- **Each magnification is an independent replication**, not a "
            "paired comparison: trident writes a separate coordinate grid per "
            "magnification, so a 5× patch has no correspondence to a 20× patch. "
            "What is compared across magnifications is the *result* (the "
            "similarity matrix), never the embeddings.\n"
            "- **Held fixed:** the encoder set (intersected across "
            "magnifications → CTransPath, GigaPath, KEEP, ResNet50, UNI2) and "
            "the slide set (only slides present at all three), with one shared "
            "seed.\n"
            "- **Plotted value:** the mean off-diagonal similarity across all "
            "encoder pairs, per metric, at each magnification. 50,000 patches "
            "sampled per magnification; O(n²) metrics subsample 8,000."
        ),
    },
    "fig5_transfer": {
        "title": "Cross-model transfer (source → target)",
        "caption": (
            "Converting one encoder's embeddings into another's through the "
            "shared space. Left: mean cosine of the translated vector to the "
            "target's real embedding. Right: Recall@1 retrieving from the "
            "target's real index in its native space. Rows are the source "
            "encoder, columns the target. Both are directional, so the full "
            "6×6 grid is shown. Note the bright ResNet50 *column*: ResNet50 is "
            "the easiest target to hit (lowest-dimensional, least structured), "
            "which is not the same as it being well reconstructed."
        ),
        "source": "results/full_run/analysis/transfer/matrix_{cosine,retrieval_recall1}.csv",
        "methods": (
            "- **Operation:** encode patches with the source model → GCCA "
            "shared space → decode as the target model. Evaluated on held-out "
            "patches (30% test), aligner fit on the training split.\n"
            "- **Cosine panel:** mean row-wise cosine between the translated "
            "embedding and the target's *real* embedding of the same patch.\n"
            "- **Recall@1 panel:** the translated query is used to retrieve "
            "from a database of the target's untouched real embeddings — the "
            "deployment-realistic test.\n"
            "- A companion linear-probe analysis (in `transfer_summary.csv`, "
            "not plotted here) shows the discriminative content transfers even "
            "where the geometry does not: a probe trained on the target's real "
            "features keeps ~96% of its accuracy on translated ones."
        ),
    },
    "fig7_alignment_methods": {
        "title": "Shared latent space — quality of the alignment methods",
        "caption": (
            "On the flagship CPTAC · 10× · 256px group. The alignment methods "
            "compared on five shared-space quality "
            "metrics (all higher-is-better, rescaled to %). GCCA (darkest) is "
            "strongest on cross-model retrieval and shared-space agreement; the "
            "shared autoencoder and joint PCA trail; rigid generalized "
            "Procrustes is weakest on alignment but best on reconstruction — the "
            "expected trade-off, since a rotation preserves geometry but cannot "
            "warp one model onto another. Note that reconstruction and "
            "alignment pull in opposite directions, which is why both families "
            "of metric are shown together."
        ),
        "source": "results/full_run/analysis/alignment/aligner_comparison.csv",
        "methods": (
            "- **Methods shown (5 of 6):** generalized Procrustes, MCCA, shared "
            "autoencoder, joint PCA, GCCA — all at latent dim 64, fit on a "
            "training patch split and evaluated on held-out patches. "
            "**Optimal transport is implemented but excluded from this run:** "
            "its unsupervised mode is unreliable and supervised OT reduces to "
            "Procrustes, so it would add a redundant bar.\n"
            "- **Metrics:** cross-model Recall@1 (retrieval in the shared "
            "space), reconstruction R² (round-trip fidelity), paired cosine and "
            "shared-space CKA (how well the models agree once aligned), and "
            "k-NN neighbourhood preservation (a collapse detector). Reported "
            "side by side because optimising alignment alone can collapse the "
            "space, and optimising reconstruction alone leaves models "
            "unaligned.\n"
            "- **Excluded from the plot:** `alignment_error` (lower-is-better, "
            "opposite direction) and `effective_rank` (0–64 scale) live in the "
            "CSV but would not share this axis."
        ),
    },
    "fig10_layerwise": {
        "title": "Layer-wise representational alignment (architecture depth)",
        "caption": (
            "Phase IV — how similarity evolves through network depth, not just at "
            "the output, across five encoders. Left: CKA between every block of "
            "UNI2 (24) and GigaPath (40); the dashed line is the lockstep "
            "diagonal, and the bright deep-block region shows the two ViTs "
            "converging. Right: UNI2's best-match CKA to each other encoder "
            "against relative depth. **The result stratifies by architecture:** "
            "the ViT-based pathology models (GigaPath, CONCH) climb to ≈0.95–0.97 "
            "with depth — growing *more* similar to UNI2, against the "
            "universal-early / specific-late intuition — while the hybrid "
            "CTransPath stays low (≈0.45–0.50) and the ResNet50 control rises "
            "then falls. Architecture family, not depth alone, governs where two "
            "networks end up."
        ),
        "source": "results/layerwise/tcga_10x_256_cls/{matrices/*.csv,divergence_profile.csv}",
        "methods": (
            "- **This stage does not read the feature store** — trident saved "
            "only final pooled embeddings, so intermediate activations were "
            "recreated by re-running the models with forward hooks on patches "
            "re-cropped from downloaded slides at trident's own coordinates. The "
            "`patches/` coordinate grid survived even though the `features_*` "
            "were deleted, so the analysis is reproducible from the slides "
            "alone.\n"
            "- **Encoders (5):** UNI2 (24 ViT blocks), GigaPath (40), CONCH (12 "
            "visual blocks — CoCa image tower), CTransPath (12 Swin blocks), "
            "ResNet50 (3 conv stages, global-average pooled). CTransPath and "
            "GigaPath were loaded in trident's own conda env (`timm==0.9.16` + "
            "`timm_ctp`), which the project's base env could not provide.\n"
            "- **Metric:** linear CKA between block activations, CLS-token "
            "pooled (a `mean`-pooled variant is also on disk). 512 patches from "
            "4 TCGA slides.\n"
            "- **Divergence depth** is the *last sustained* crossing below "
            "CKA 0.5, not the first dip — real trajectories are not monotone."
        ),
    },
    "fig8_retrieval_tcga": {
        "title": "Cross-model retrieval in the shared latent space — TCGA",
        "caption": (
            "The TCGA counterpart of the retrieval figure, on the "
            "`master_benchmark/10x_256px` group (5 encoders — the flagship set "
            "minus KEEP, which was not extracted for TCGA at this grid). The "
            "pattern replicates the CPTAC result: alignment lifts cross-model "
            "retrieval from chance to Recall@1 ≈ 0.95 (GCCA), and a rigid "
            "rotation is insufficient."
        ),
        "source": "results/groups/tcga_10x_256/retrieval/retrieval_summary.csv",
        "methods": (
            "- Identical protocol to the CPTAC retrieval figure — identity "
            "retrieval on held-out patches (30% test), aligners fit on the "
            "training split, latent dim 64, two unaligned PCA/truncation "
            "controls.\n"
            "- **Encoders (5):** CONCH, CTransPath, Prov-GigaPath, ResNet50, "
            "UNI2. TCGA's 10×/256px group lacks KEEP, so a direct CPTAC↔TCGA "
            "comparison would restrict to these five shared encoders."
        ),
    },
    "fig9_alignment_tcga": {
        "title": "Shared latent space — alignment methods, TCGA",
        "caption": (
            "The TCGA counterpart of the alignment-quality figure, on "
            "`master_benchmark/10x_256px` (5 encoders). The method ranking "
            "matches CPTAC almost exactly — GCCA best on retrieval and "
            "shared-space CKA, Procrustes best on reconstruction and "
            "neighbourhood preservation but weakest on alignment — which shows "
            "the alignment ↔ reconstruction trade-off is not cohort-specific."
        ),
        "source": "results/groups/tcga_10x_256/alignment/aligner_comparison.csv",
        "methods": (
            "- Identical to the CPTAC alignment figure: five methods "
            "(generalized Procrustes, MCCA, autoencoder, joint PCA, GCCA) at "
            "latent dim 64, on five higher-is-better metrics; optimal transport "
            "excluded (see the CPTAC figure).\n"
            "- **Encoders (5):** CONCH, CTransPath, Prov-GigaPath, ResNet50, UNI2."
        ),
    },
    "fig6_slide_encoders": {
        "title": "Slide-level encoders — all six, across four patch grids",
        "caption": (
            "Linear CKA between the six slide-level encoders, for TCGA (left) "
            "and CPTAC (right). Because slide encoders emit one vector per "
            "slide, the patch-grid pairing constraint does not apply — all six "
            "are directly comparable even though they were built on four "
            "different patch grids, across every slide in each cohort. "
            "Feather↔TITAN is the strongest pair (shared lineage); CHIEF and "
            "Madeleine share a grid yet are only weakly similar, so the grid is "
            "not what drives agreement."
        ),
        "source": "results/slide_encoders/{master_benchmark,cptac_benchmark}/matrices/linear_cka.csv",
        "methods": (
            "- **Encoders (6):** CHIEF, Madeleine (10x/256px), PRISM "
            "(20x/224px), GigaPath-slide (20x/256px), TITAN, Feather "
            "(20x/512px).\n"
            "- **Data:** one embedding per slide, paired by slide id across "
            "2,169 TCGA / 2,296 CPTAC slides (all slides, no subsampling of "
            "slides).\n"
            "- **Metric:** linear CKA on column-centered slide embeddings; "
            "lower triangle shown.\n"
            "- **Caveat:** n ≈ 2,200 slides against 512–1280 dims is comfortable "
            "for CKA but near the floor for the CCA family, which is why only "
            "CKA is shown for the slide-level analysis."
        ),
    },
}


def _incomplete_downstream_tasks() -> list[str]:
    """Tasks whose results.csv is missing an input condition.

    The fig3 caption used to assert in a hardcoded string that LUAD KRAS could
    never be completed because its features were deleted. That was wrong -- the
    features are present -- and a static sentence would go stale the moment the
    missing rows were computed. Derive it from the tables instead, so the
    caption stops mentioning a task as soon as it is filled in.
    """
    root = Path(__file__).resolve().parents[1]
    want = {"single", "concat", "shared"}
    missing = []
    for f in sorted(root.glob("results/full_run/downstream/downstream/*/results.csv")):
        try:
            have = set(pd.read_csv(f)["condition"].unique())
        except Exception:
            continue
        if want - have:
            missing.append(f.parent.name)
    return missing


def emit(fig, out: Path, key: str, fmt: str) -> None:
    """Save a figure into its own directory alongside a provenance markdown.

    Layout: ``<out>/<key>/<key>.<fmt>`` plus ``<out>/<key>/<key>.md``.
    """
    fdir = out / key
    fdir.mkdir(parents=True, exist_ok=True)
    save_plot(fig, fdir / f"{key}.{fmt}")

    meta = FIG_META.get(key)
    if meta is not None:
        caption = meta["caption"]
        if key.startswith("fig3_downstream"):
            gaps = _incomplete_downstream_tasks()
            if gaps:
                names = ", ".join(clean_label(t) for t in gaps)
                caption += (
                    f" {names} shows fewer than three bars: those input "
                    "conditions have not been computed yet. The features are "
                    "present, so this is a pending run, not a missing result."
                )
        md = fdir / f"{key}.md"
        md.write_text(
            f"# {meta['title']}\n\n"
            f"![{key}]({key}.png)\n\n"
            f"**Caption.** {caption}\n\n"
            f"## How it was computed\n\n{meta['methods']}\n\n"
            f"## Source data\n\n`{meta['source']}`\n\n"
            f"## Files\n\n"
            f"- `{key}.png` — raster preview\n"
            f"- `{key}.pdf` — vector, for the paper\n"
            f"- `{key}.md` — this file\n\n"
            f"Regenerate with `python scripts/render_figures.py "
            f"--out results/figures --format pdf`.\n"
        )
    print(f"  {key}/{key}.{fmt}")


# ---------------------------------------------------------------------------
# Fig 1 — representational similarity (flagship: 6 encoders, CPTAC 10x/256px)
# ---------------------------------------------------------------------------


def fig_similarity(out: Path, fmt: str) -> None:
    mdir = RESULTS / "full_run/analysis/similarity/matrices"
    mats = {
        clean_label(m): _load_matrix(mdir / f"{m}.csv")
        for m in METRICS
        if (mdir / f"{m}.csv").exists()
    }
    if not mats:
        print("  skip similarity"); return
    fig = heatmap_row(
        mats, value_fmt="{:.2f}", mask="lower", ylab="Encoder",
        cbar_label="Similarity", rotate_xticks=40,
        panel_size=(3.5, 3.9), label_size=7.5, base_size=10,
    )
    emit(fig, out, "fig1_similarity", fmt)


# ---------------------------------------------------------------------------
# Fig 2 — cross-model retrieval: alignment lifts it from chance
# ---------------------------------------------------------------------------


def fig_retrieval(out: Path, fmt: str) -> None:
    f = RESULTS / "full_run/analysis/retrieval/retrieval_summary.csv"
    if not f.exists():
        print("  skip retrieval"); return
    df = pd.read_csv(f, index_col=0)
    metrics = ["recall@1", "recall@5", "recall@10", "map", "ndcg"]
    conds = [c for c in ALIGN_ORDER if c in df.index]

    rows = []
    for c in conds:
        for m in metrics:
            rows.append({"metric": clean_label(m), "condition": clean_label(c),
                         "value": df.loc[c, m] * 100})
    long = pd.DataFrame(rows)

    fig, ax = grouped_bars(
        long, x="metric", y="value", group="condition",
        group_order=[clean_label(c) for c in conds], highlight="GCCA",
        ramp="purple", value_fmt="{:.0f}", ylab="Score (%)",
        figsize=(9.6, 5.0), label_size=7,
    )
    ax.set_ylim(0, 108)
    ax.set_title("Cross-model retrieval in the shared space  ·  CPTAC · 10× · 256px",
                 fontweight="bold", fontsize=12.5, pad=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend_below(fig, ax, highlight="GCCA", y=0.0)
    emit(fig, out, "fig2_retrieval", fmt)


# ---------------------------------------------------------------------------
# Fig 3 — downstream: best single encoder vs concat vs shared, per task
# ---------------------------------------------------------------------------


# The two difficulty regimes, stacked as separate panels.
CLINICAL_TASKS = [
    "tcga_nsclc", "tcga_brca_subtype",
    "tcga_brca_stage", "tcga_nsclc_stage",
]
MUTATION_TASKS = [
    "cptac_luad_tp53", "cptac_luad_kras", "cptac_luad_stk11",
    "cptac_brca_pik3ca", "cptac_brca_gata3", "cptac_brca_map3k1",
    "cptac_coad_tp53", "cptac_coad_kras", "cptac_coad_pik3ca",
]
# The 3-series blue ramp grouped_bars uses, so the shared legend matches.
_DS_COLOURS = ["#a9cce3", "#6aa5cd", "#2e6f9e"]
_DS_SERIES = ["Best single", "Concat", "MOSAIC"]


def fig_downstream(out: Path, fmt: str, metric: str = "auc") -> None:
    from matplotlib.patches import Patch

    root = RESULTS / "full_run/downstream/downstream"
    files = sorted(root.glob("*/results.csv"))
    if not files:
        print("  skip downstream"); return

    frames = [pd.read_csv(f).assign(task=f.parent.name) for f in files]
    data = pd.concat(frames, ignore_index=True)
    data["v"] = data[metric] * 100

    # Average over MIL heads; per task keep the BEST single encoder, plus
    # concat and shared — a clean 3-way that makes the representation the point.
    rows = []
    for task, g in data.groupby("task"):
        single = g[g.condition == "single"].groupby("encoders")["v"].mean()
        rows += [
            {"task": task, "series": "Best single",
             "value": single.max() if len(single) else np.nan},
            {"task": task, "series": "Concat",
             "value": g[g.condition == "concat"]["v"].mean()},
            {"task": task, "series": "MOSAIC",
             "value": g[g.condition == "shared"]["v"].mean()},
        ]
    agg = pd.DataFrame(rows).dropna(subset=["value"])
    present = set(agg["task"])

    apply_style(11)
    fig, axes = plt.subplots(
        2, 1, figsize=(11.0, 8.2),
        gridspec_kw={"height_ratios": [len([t for t in CLINICAL_TASKS if t in present]),
                                       len([t for t in MUTATION_TASKS if t in present])],
                     "hspace": 0.60},
    )

    panels = [
        (axes[0], CLINICAL_TASKS, "Morphological & clinical",
         "subtyping and staging — visible in tissue architecture"),
        (axes[1], MUTATION_TASKS, "Molecular  (mutation prediction)",
         "weak morphological signal — where representations separate"),
    ]
    for ax, task_list, title, subtitle in panels:
        tasks = [t for t in task_list if t in present]
        sub = agg[agg["task"].isin(tasks)].copy()
        sub["task_label"] = sub["task"].map(lambda t: TASK_TITLES.get(t, clean_label(t)))
        order = [TASK_TITLES.get(t, clean_label(t)) for t in tasks]

        grouped_bars(
            sub, x="task_label", y="value", group="series",
            group_order=_DS_SERIES, highlight="MOSAIC", x_order=order,
            ramp="blue", value_fmt="{:.0f}", ylab=f"{clean_label(metric)} (%)",
            label_size=7.5, separators=True, ax=ax,
        )
        if ax.get_legend():
            ax.get_legend().remove()
        ax.axhline(50, color="#888888", lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(ax.get_xlim()[1], 50, " chance", va="center", ha="left",
                fontsize=7.5, color="#888888")
        ax.set_ylim(0, 108)
        # Title on the upper line, italic subtitle just beneath it — both above
        # the axes, clearly separated so neither touches the bars.
        ax.text(0, 1.16, title, transform=ax.transAxes, fontweight="bold",
                fontsize=12, va="bottom")
        ax.text(0, 1.05, subtitle, transform=ax.transAxes, fontsize=8.5,
                style="italic", color="#666666", va="bottom")
        ax.tick_params(axis="x", labelsize=8.5)

    fig.tight_layout(rect=(0, 0, 1, 0.88))

    # Title on the top line, the shared legend on a separate line beneath it,
    # both clear of the panels.
    fig.suptitle("Downstream MIL — single encoder vs. concatenation vs. shared space",
                 fontsize=13.5, fontweight="bold", y=0.985)
    handles = [Patch(facecolor=c, label=s) for c, s in zip(_DS_COLOURS, _DS_SERIES)]
    leg = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                     fontsize=10, bbox_to_anchor=(0.5, 0.94), handlelength=1.3,
                     columnspacing=1.8)
    for t in leg.get_texts():
        if t.get_text() == "MOSAIC":
            t.set_fontweight("bold")
    emit(fig, out, f"fig3_downstream_{metric}", fmt)


# ---------------------------------------------------------------------------
# Fig 4 — magnification: agreement falls as resolution rises
# ---------------------------------------------------------------------------


def fig_magnification(out: Path, fmt: str) -> None:
    f = RESULTS / "magnification/cptac_benchmark_256px/magnification_summary.csv"
    if not f.exists():
        print("  skip magnification"); return
    df = pd.read_csv(f)
    metrics = ["linear_cka", "kernel_cka", "svcca", "pwcca",
               "procrustes", "cosine_rsa", "distance_correlation"]
    metrics = [m for m in metrics if m in df.columns]

    # A curated qualitative palette — distinct, harmonious, print-safe.
    palette = ["#2e6f9e", "#1f7a7a", "#3f9e57", "#d4a13a",
               "#dd7b32", "#c0392b", "#7a51a8"]
    mags = df["magnification"].to_numpy()

    # Order lines (and the legend) by their value at the highest magnification,
    # so the legend labels line up top-to-bottom with the line endpoints.
    order = sorted(metrics, key=lambda m: df[m].iloc[-1], reverse=True)
    colours = {m: palette[i % len(palette)] for i, m in enumerate(order)}
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    apply_style(11)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, m in enumerate(order):
        ax.plot(mags, df[m], marker=markers[i % len(markers)], markersize=6.5,
                lw=2.4, color=colours[m], markeredgecolor="white",
                markeredgewidth=0.8, label=clean_label(m), zorder=3,
                clip_on=False, solid_capstyle="round")

    ax.set_xscale("log", base=2)
    ax.set_xticks(mags)
    ax.set_xticklabels([f"{int(v)}×" for v in mags])
    ax.set_xlim(mags.min() * 0.92, mags.max() * 1.08)
    ax.minorticks_off()
    ax.set_xlabel("Magnification", fontweight="bold")
    ax.set_ylabel("Cross-model similarity", fontweight="bold")
    ax.set_title("Agreement decreases with magnification",
                 fontweight="bold", fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.28, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Legend outside, to the right, in line-endpoint order — no overlap.
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
        fontsize=9, handlelength=1.6, borderaxespad=0, labelspacing=0.7,
    )
    fig.tight_layout()
    emit(fig, out, "fig4_magnification", fmt)


# ---------------------------------------------------------------------------
# Fig 5 — cross-model transfer: the probe transfers though geometry does not
# ---------------------------------------------------------------------------


def fig_transfer(out: Path, fmt: str) -> None:
    f = RESULTS / "full_run/analysis/transfer"
    cos = f / "matrix_cosine.csv"
    rec = f / "matrix_retrieval_recall1.csv"
    mats = {}
    if cos.exists():
        mats["Cosine to real target"] = _load_matrix(cos)
    if rec.exists():
        mats["Retrieval Recall@1"] = _load_matrix(rec)
    if not mats:
        print("  skip transfer"); return
    fig = heatmap_row(
        mats, value_fmt="{:.2f}", ylab="Source encoder", xlab="Target encoder",
        cbar_label="", rotate_xticks=40, panel_size=(4.6, 4.8), label_size=8.0,
    )
    fig.suptitle("Cross-model transfer  (source → target)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    emit(fig, out, "fig5_transfer", fmt)


# ---------------------------------------------------------------------------
# Fig 6 — slide encoders: all six compared, across four grids
# ---------------------------------------------------------------------------


def fig_slide(out: Path, fmt: str) -> None:
    base = RESULTS / "slide_encoders"
    mats = {}
    for cohort, title in [("master_benchmark", "TCGA"), ("cptac_benchmark", "CPTAC")]:
        p = base / cohort / "matrices/linear_cka.csv"
        if p.exists():
            mats[title] = _load_matrix(p)
    if not mats:
        print("  skip slide"); return
    fig = heatmap_row(
        mats, value_fmt="{:.2f}", mask="lower", ylab="Slide encoder",
        cbar_label="Linear CKA", rotate_xticks=40,
        panel_size=(4.4, 4.6), label_size=8.0,
    )
    fig.suptitle("Slide-level encoders — all six, across four patch grids",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    emit(fig, out, "fig6_slide_encoders", fmt)


# ---------------------------------------------------------------------------
# Fig 7 — the alignment methods, compared on shared-space quality
# ---------------------------------------------------------------------------

# Higher-is-better metrics on a common [0, 1] scale, so they share one axis.
_ALIGN_METRICS = {
    "recall@1": "Cross-model\nRecall@1",
    "reconstruction_r2": "Reconstruction\nR²",
    "paired_cosine": "Paired\ncosine",
    "shared_cka": "Shared-space\nCKA",
    "neighborhood_preservation": "Neighbourhood\npreservation",
}
_ALIGN_ORDER = ["procrustes", "mcca", "autoencoder", "joint_pca", "gcca"]


def fig_alignment(out: Path, fmt: str) -> None:
    f = RESULTS / "full_run/analysis/alignment/aligner_comparison.csv"
    if not f.exists():
        print("  skip alignment"); return
    df = pd.read_csv(f, index_col=0)
    methods = [m for m in _ALIGN_ORDER if m in df.index]

    rows = []
    for m in methods:
        for col, label in _ALIGN_METRICS.items():
            if col in df.columns:
                rows.append({"metric": label, "method": clean_label(m),
                             "value": df.loc[m, col] * 100})
    long = pd.DataFrame(rows)

    fig, ax = grouped_bars(
        long, x="metric", y="value", group="method",
        group_order=[clean_label(m) for m in methods], highlight="GCCA",
        ramp="green", value_fmt="{:.0f}", ylab="Score (%)",
        figsize=(10.5, 5.1), label_size=6.5,
    )
    ax.set_ylim(0, 108)
    ax.set_title("Shared latent space — alignment methods  ·  CPTAC · 10× · 256px",
                 fontweight="bold", fontsize=12.5, pad=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend_below(fig, ax, highlight="GCCA", y=0.0)
    emit(fig, out, "fig7_alignment_methods", fmt)


# ---------------------------------------------------------------------------
# Every patch-encoder group and every magnification series — so all 12 patch
# encoders get comparative figures, not only the flagship six.
# ---------------------------------------------------------------------------

#: label -> (result base dir, human title). The flagship lives under full_run.
GROUP_DIRS = {
    "cptac_10x_256px": (RESULTS / "full_run/analysis",
                        "CPTAC · 10× · 256px  (flagship, 6 encoders)"),
    "tcga_10x_256px": (RESULTS / "groups/tcga_10x_256", "TCGA · 10× · 256px"),
    "cptac_20x_256px": (RESULTS / "groups/cptac_20x_256", "CPTAC · 20× · 256px"),
    "cptac_20x_224px": (RESULTS / "groups/cptac_20x_224",
                        "CPTAC · 20× · 224px  (Virchow / GPFM / H-optimus)"),
    "tcga_20x_224px": (RESULTS / "groups/tcga_20x_224",
                       "TCGA · 20× · 224px  (Virchow / GPFM / H-optimus)"),
    "cptac_20x_512px": (RESULTS / "groups/cptac_20x_512",
                        "CPTAC · 20× · 512px  (CONCH v1 / v1.5)"),
}


def _similarity_panel(mdir: Path, title: str, path: Path, fmt: str) -> bool:
    """Render a similarity heatmap row for one group, sized to its encoders."""
    mats = {clean_label(m): _load_matrix(mdir / f"{m}.csv")
            for m in METRICS if (mdir / f"{m}.csv").exists()}
    if not mats:
        return False
    n = next(iter(mats.values())).shape[0]
    fig = heatmap_row(
        mats, value_fmt="{:.2f}", mask="lower", ylab="Encoder",
        cbar_label="Similarity", rotate_xticks=40,
        panel_size=(max(2.8, 0.62 * n + 1.0), max(3.0, 0.62 * n + 1.2)),
        label_size=8.0,
    )
    fig.suptitle(title, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, path)
    return True


def _retrieval_panel(f: Path, title: str, path: Path, fmt: str) -> bool:
    if not f.exists():
        return False
    df = pd.read_csv(f, index_col=0)
    metrics = ["recall@1", "recall@5", "recall@10", "map", "ndcg"]
    conds = [c for c in ALIGN_ORDER if c in df.index]
    if not conds:
        return False
    rows = [{"metric": clean_label(m), "condition": clean_label(c),
             "value": df.loc[c, m] * 100}
            for c in conds for m in metrics if m in df.columns]
    fig, ax = grouped_bars(
        pd.DataFrame(rows), x="metric", y="value", group="condition",
        group_order=[clean_label(c) for c in conds], highlight="GCCA",
        ramp="purple", value_fmt="{:.0f}", ylab="Score (%)",
        figsize=(9.6, 5.0), label_size=7,
    )
    ax.set_ylim(0, 108)
    ax.set_title(f"Cross-model retrieval — {title}", fontweight="bold",
                 fontsize=12.5, pad=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend_below(fig, ax, highlight="GCCA", y=0.0)
    save_plot(fig, path)
    return True


def _alignment_panel(f: Path, title: str, path: Path, fmt: str) -> bool:
    if not f.exists():
        return False
    df = pd.read_csv(f, index_col=0)
    methods = [m for m in _ALIGN_ORDER if m in df.index]
    if not methods:
        return False
    rows = [{"metric": label, "method": clean_label(m), "value": df.loc[m, col] * 100}
            for m in methods for col, label in _ALIGN_METRICS.items()
            if col in df.columns]
    fig, ax = grouped_bars(
        pd.DataFrame(rows), x="metric", y="value", group="method",
        group_order=[clean_label(m) for m in methods], highlight="GCCA",
        ramp="green", value_fmt="{:.0f}", ylab="Score (%)",
        figsize=(10.5, 5.1), label_size=6.5,
    )
    ax.set_ylim(0, 108)
    ax.set_title(f"Alignment methods — {title}", fontweight="bold",
                 fontsize=12.5, pad=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend_below(fig, ax, highlight="GCCA", y=0.0)
    save_plot(fig, path)
    return True


def _transfer_panel(tdir: Path, title: str, path: Path, fmt: str) -> bool:
    """Source→target transfer heatmaps (cosine + Recall@1) for one group."""
    mats = {}
    if (tdir / "matrix_cosine.csv").exists():
        mats["Cosine to real target"] = _load_matrix(tdir / "matrix_cosine.csv")
    if (tdir / "matrix_retrieval_recall1.csv").exists():
        mats["Retrieval Recall@1"] = _load_matrix(tdir / "matrix_retrieval_recall1.csv")
    if not mats:
        return False
    n = next(iter(mats.values())).shape[0]
    fig = heatmap_row(
        mats, value_fmt="{:.2f}", ylab="Source encoder", xlab="Target encoder",
        cbar_label="", rotate_xticks=40,
        panel_size=(max(3.2, 0.66 * n + 1.1), max(3.4, 0.66 * n + 1.3)),
        label_size=8.0,
    )
    fig.suptitle(f"Cross-model transfer — {title}", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, path)
    return True


def figs_all_groups(out: Path, fmt: str) -> None:
    """Similarity, retrieval, alignment and transfer figures for every group."""
    root = out / "groups"
    for label, (base, title) in GROUP_DIRS.items():
        gdir = root / label
        gdir.mkdir(parents=True, exist_ok=True)
        did = []
        if _similarity_panel(base / "similarity/matrices", title,
                             gdir / f"similarity.{fmt}", fmt):
            did.append("similarity")
        if _retrieval_panel(base / "retrieval/retrieval_summary.csv", title,
                           gdir / f"retrieval.{fmt}", fmt):
            did.append("retrieval")
        if _alignment_panel(base / "alignment/aligner_comparison.csv", title,
                          gdir / f"alignment.{fmt}", fmt):
            did.append("alignment")
        if _transfer_panel(base / "transfer", title,
                           gdir / f"transfer.{fmt}", fmt):
            did.append("transfer")
        print(f"  groups/{label}: {', '.join(did) or 'no data'}")


def figs_all_magnification(out: Path, fmt: str) -> None:
    """A line figure per magnification series."""
    root = out / "magnification"
    for f in sorted(RESULTS.glob("magnification/*/magnification_summary.csv")):
        series = f.parent.name
        sdir = root / series
        sdir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(f)
        metrics = [m for m in ["linear_cka", "kernel_cka", "svcca", "pwcca",
                               "procrustes", "cosine_rsa", "distance_correlation"]
                   if m in df.columns]
        palette = ["#2e6f9e", "#1f7a7a", "#3f9e57", "#d4a13a",
                   "#dd7b32", "#c0392b", "#7a51a8"]
        markers = ["o", "s", "^", "D", "v", "P", "X"]
        order = sorted(metrics, key=lambda m: df[m].iloc[-1], reverse=True)
        mags = df["magnification"].to_numpy()

        apply_style(11)
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for i, m in enumerate(order):
            ax.plot(mags, df[m], marker=markers[i % len(markers)], markersize=6.5,
                    lw=2.4, color=palette[i % len(palette)], markeredgecolor="white",
                    markeredgewidth=0.8, label=clean_label(m), clip_on=False)
        ax.set_xscale("log", base=2)
        ax.set_xticks(mags)
        ax.set_xticklabels([f"{int(v)}×" for v in mags])
        ax.set_xlim(mags.min() * 0.92, mags.max() * 1.08)
        ax.minorticks_off()
        ax.set_xlabel("Magnification", fontweight="bold")
        ax.set_ylabel("Cross-model similarity", fontweight="bold")
        ax.set_title(f"Agreement vs magnification — {series}",
                     fontweight="bold", fontsize=12.5, pad=10)
        ax.grid(axis="y", alpha=0.28, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                  fontsize=9, handlelength=1.6, labelspacing=0.7)
        fig.tight_layout()
        save_plot(fig, sdir / f"magnification.{fmt}")
        print(f"  magnification/{series}")


# Layer-wise encoders: canonical order, display names and trajectory colours.
_LW_ENC = {
    "uni_v2": "UNI2", "gigapath": "GigaPath", "conch_v1": "CONCH",
    "ctranspath": "CTransPath", "resnet50": "ResNet50",
}
_LW_ORDER = ["uni_v2", "gigapath", "conch_v1", "ctranspath", "resnet50"]
_LW_COLOR = {
    "uni_v2": "#6a4fb3", "gigapath": "#2e6f9e", "conch_v1": "#1f7a7a",
    "ctranspath": "#d2691e", "resnet50": "#c0392b",
}


def _lw_pair(mdir: Path, a: str, b: str) -> pd.DataFrame | None:
    """Block×block CKA matrix oriented with ``a`` on the rows, ``b`` on cols.

    Only one direction of each pair is stored on disk; the reverse is just the
    transpose, so ``a``'s depth-trajectory into ``b`` is always the row-wise max.
    """
    fa, fb = mdir / f"{a}__{b}.csv", mdir / f"{b}__{a}.csv"
    if fa.exists():
        return pd.read_csv(fa, index_col=0)
    if fb.exists():
        return pd.read_csv(fb, index_col=0).T
    return None


def _build_layerwise_fig(mdir: Path, anchor: str, pool: str,
                         legend: str = "best", partner: str | None = None):
    """Two-panel layer-wise figure anchored on ``anchor``. Returns fig or None.

    Left: block×block CKA of the anchor against its most-convergent partner.
    Right: the anchor's best-match CKA to every other encoder, vs depth.
    """
    others = [e for e in _LW_ORDER if e != anchor]
    mats = {b: _lw_pair(mdir, anchor, b) for b in others}
    mats = {b: M for b, M in mats.items() if M is not None}
    if not mats:
        return None
    # per anchor-block, best match in b (row-wise max of the block×block matrix)
    trajs = {b: (np.linspace(0.0, 1.0, M.shape[0]), M.values.max(axis=1))
             for b, M in mats.items()}
    # heatmap partner: an explicit choice, else the most convergent partner —
    # skipping degenerate <6-block partners (ResNet50) so the panel stays informative
    if partner not in mats:
        cand = {b: M for b, M in mats.items() if M.shape[1] >= 6} or mats
        partner = max(cand, key=lambda b: trajs[b][1].max())
    Mh = mats[partner]

    apply_style(11)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2),
                             gridspec_kw={"width_ratios": [1.0, 1.3], "wspace": 0.32})

    # --- Panel A: block × block CKA, anchor vs its closest partner --------
    ax = axes[0]
    im = ax.imshow(Mh.values, cmap="magma", origin="lower", aspect="auto",
                   vmin=0.4, vmax=1.0)
    na, nb = Mh.shape
    ax.plot([0, nb - 1], [0, na - 1], color="white", lw=1.0, ls="--",
            alpha=0.7)  # lockstep diagonal (relative depth)
    ax.set_xlabel(f"{_LW_ENC[partner]} block  (depth →)", fontweight="bold")
    ax.set_ylabel(f"{_LW_ENC[anchor]} block  (depth →)", fontweight="bold")
    ax.set_title("Block-by-block CKA", fontweight="bold", fontsize=12.5, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Linear CKA", fontweight="bold", fontsize=9)
    cb.outline.set_visible(False)

    # --- Panel B: anchor's best-match trajectory to every other encoder --
    ax = axes[1]
    for b in others:
        if b not in trajs:
            continue
        depth, best = trajs[b]
        ax.plot(depth, best, marker="o", markersize=4.5, lw=2.2,
                color=_LW_COLOR[b], markeredgecolor="white", markeredgewidth=0.6,
                label=f"{_LW_ENC[anchor]} → {_LW_ENC[b]}")
    ax.set_xlabel(f"Relative depth in {_LW_ENC[anchor]}", fontweight="bold")
    ax.set_ylabel("Best-match CKA in the other model", fontweight="bold")
    ax.set_ylim(0.35, 1.02)
    ax.set_title(f"{_LW_ENC[anchor]}'s alignment across depth", fontweight="bold",
                 fontsize=12.5, pad=8)
    ax.grid(alpha=0.28, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend == "tuned":
        # hand-placed in the clean band between the CTransPath and ResNet50
        # curves (only valid for the UNI2 / CLS headline arrangement)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.235), fontsize=8.5,
                  frameon=False, labelspacing=0.32, handlelength=1.6,
                  borderaxespad=0.4)
    else:
        # curves rearrange per anchor, so let matplotlib dodge the data
        ax.legend(loc="best", fontsize=8.5, frameon=False, labelspacing=0.32,
                  handlelength=1.6, borderaxespad=0.4)

    fig.suptitle(
        f"Layer-wise representational alignment  ·  {pool.upper()} pooling  ·  "
        f"TCGA  ·  {_LW_ENC[anchor]} baseline",
        fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def fig_layerwise(out: Path, fmt: str, pool: str = "cls") -> None:
    """Headline Phase-IV figure: UNI2 baseline (block heatmap + trajectories)."""
    mdir = RESULTS / f"layerwise/tcga_10x_256_{pool}/matrices"
    if not mdir.exists():
        print("  skip layerwise"); return
    # pin the heatmap partner to GigaPath (the 24×40 panel the caption describes)
    fig = _build_layerwise_fig(mdir, "uni_v2", pool, legend="tuned",
                               partner="gigapath")
    if fig is None:
        print("  skip layerwise"); return
    key = "fig10_layerwise" if pool == "cls" else "fig10b_layerwise_mean"
    emit(fig, out, key, fmt)


def figs_all_layerwise(out: Path, fmt: str) -> None:
    """One layer-wise figure per (baseline encoder × pooling) variation."""
    root = out / "layerwise"
    for pool in ("cls", "mean"):
        mdir = RESULTS / f"layerwise/tcga_10x_256_{pool}/matrices"
        if not mdir.exists():
            continue
        for anchor in _LW_ORDER:
            fig = _build_layerwise_fig(mdir, anchor, pool, legend="best")
            if fig is None:
                continue
            key = f"{anchor}_{pool}"
            sdir = root / key
            sdir.mkdir(parents=True, exist_ok=True)
            for ext in {fmt, "png"}:
                save_plot(fig, sdir / f"layerwise_{key}.{ext}")
            plt.close(fig)
            print(f"  layerwise/{key} ({_LW_ENC[anchor]} baseline, {pool})")


def _retrieval_grouped_fig(df, title):
    """Build a retrieval grouped-bar figure (shared by CPTAC and TCGA)."""
    metrics = ["recall@1", "recall@5", "recall@10", "map", "ndcg"]
    conds = [c for c in ALIGN_ORDER if c in df.index]
    rows = [{"metric": clean_label(m), "condition": clean_label(c),
             "value": df.loc[c, m] * 100}
            for c in conds for m in metrics if m in df.columns]
    fig, ax = grouped_bars(
        pd.DataFrame(rows), x="metric", y="value", group="condition",
        group_order=[clean_label(c) for c in conds], highlight="GCCA",
        ramp="purple", value_fmt="{:.0f}", ylab="Score (%)",
        figsize=(9.6, 5.0), label_size=7,
    )
    ax.set_ylim(0, 108)
    ax.set_title(title, fontweight="bold", fontsize=12.5, pad=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend_below(fig, ax, highlight="GCCA", y=0.0)
    return fig


def _alignment_grouped_fig(df, title):
    """Build an alignment-methods grouped-bar figure (CPTAC and TCGA)."""
    methods = [m for m in _ALIGN_ORDER if m in df.index]
    rows = [{"metric": label, "method": clean_label(m), "value": df.loc[m, col] * 100}
            for m in methods for col, label in _ALIGN_METRICS.items()
            if col in df.columns]
    fig, ax = grouped_bars(
        pd.DataFrame(rows), x="metric", y="value", group="method",
        group_order=[clean_label(m) for m in methods], highlight="GCCA",
        ramp="green", value_fmt="{:.0f}", ylab="Score (%)",
        figsize=(10.5, 5.1), label_size=6.5,
    )
    ax.set_ylim(0, 108)
    ax.set_title(title, fontweight="bold", fontsize=12.5, pad=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legend_below(fig, ax, highlight="GCCA", y=0.0)
    return fig


def fig_retrieval_tcga(out: Path, fmt: str) -> None:
    f = RESULTS / "groups/tcga_10x_256/retrieval/retrieval_summary.csv"
    if not f.exists():
        print("  skip retrieval_tcga"); return
    fig = _retrieval_grouped_fig(
        pd.read_csv(f, index_col=0),
        "Cross-model retrieval in the shared space  ·  TCGA · 10× · 256px")
    emit(fig, out, "fig8_retrieval_tcga", fmt)


def fig_alignment_tcga(out: Path, fmt: str) -> None:
    f = RESULTS / "groups/tcga_10x_256/alignment/aligner_comparison.csv"
    if not f.exists():
        print("  skip alignment_tcga"); return
    fig = _alignment_grouped_fig(
        pd.read_csv(f, index_col=0),
        "Shared latent space — alignment methods  ·  TCGA · 10× · 256px")
    emit(fig, out, "fig9_alignment_tcga", fmt)


# ---------------------------------------------------------------------------
# Similarity, one figure per (patch grid, cohort, magnification)
# ---------------------------------------------------------------------------

#: Which encoders live on each patch grid. Row-pairing confines a similarity
#: matrix to one (cohort, magnification, patch_size) group, so these are
#: separate results rather than panels of one figure -- a 224px patch and a
#: 256px patch cover different tissue and share no row index.
GRID_NOTE = {
    "224px": "GPFM, H-optimus-0, Virchow, Virchow2 (Virchow only at 20x)",
    "256px": "CONCH / CONCH v1.5, CTransPath, Prov-GigaPath, KEEP, ResNet50, UNI2-h",
    "384px": "MUSK alone -- similarity undefined at n=1",
    "512px": "CONCH vs CONCH v1.5",
}


def _similarity_dirs() -> dict:
    """Locate every similarity matrix directory, keyed by grid/cohort/mag.

    Two layouts hold these: ``results/groups/<tag>/similarity`` and the
    ``results/full_run/analysis/similarity_<px>/<group>`` folders. Where both
    describe the same group the richer one wins, since one carries six encoders
    and the other five for TCGA 10x/256px.
    """
    root = Path(__file__).resolve().parents[1]
    found: dict = {}
    for f in sorted(root.glob("results/**/matrices/linear_cka.csv")):
        mdir = f.parent
        tag = mdir.parent.name
        if tag == "similarity":                       # results/groups/<tag>/similarity
            tag = mdir.parent.parent.name
        if mdir.parent.name == "similarity" and tag == "analysis":
            # the flagship run: results/full_run/analysis/similarity/matrices
            key = ("256px", "cptac", "10x")
            n = len(next(csv.reader(open(f)))[1:])
            if key not in found or n > found[key][0]:
                found[key] = (n, mdir)
            continue
        m = re.search(r"(cptac|master|tcga)\w*?_(\d+)x_(\d+)(?:px)?", tag)
        if not m:
            continue
        cohort = "cptac" if m.group(1) == "cptac" else "tcga"
        key = (f"{m.group(3)}px", cohort, f"{m.group(2)}x")
        n = len(next(csv.reader(open(f)))[1:])
        if key not in found or n > found[key][0]:
            found[key] = (n, mdir)
    return found


def figs_similarity_by_grid(out: Path, fmt: str) -> None:
    """One similarity figure per grid/cohort/magnification, filed by grid."""
    found = _similarity_dirs()
    if not found:
        print("  skip similarity-by-grid: nothing found")
        return

    index: dict = {}
    for (px, cohort, mag), (n, mdir) in sorted(found.items()):
        mats = {clean_label(m): _load_matrix(mdir / f"{m}.csv")
                for m in METRICS if (mdir / f"{m}.csv").exists()}
        if not mats:
            continue
        wide = max(3.0, min(4.4, 26.0 / len(mats)))
        fig = heatmap_row(
            mats, value_fmt="{:.2f}", mask="lower", ylab="Encoder",
            cbar_label="Similarity", rotate_xticks=40,
            panel_size=(wide, wide * 1.1), label_size=7.5, base_size=10,
            suptitle=f"{cohort.upper()} · {mag} · {px} · {n} encoders",
        )
        name = f"{cohort}_{mag}_{px}"
        save_plot(fig, out / "similarity_by_grid" / px / f"{name}.{fmt}")
        plt.close(fig)
        index.setdefault(px, []).append((name, cohort, mag, n, mdir))
        print(f"  similarity_by_grid/{px}/{name}.{fmt}  ({n} encoders, {len(mats)} metrics)")

    lines = ["# Similarity by patch grid", "",
             "One figure per `(patch grid, cohort, magnification)`. These are separate",
             "results, not panels of one figure: similarity needs row-paired patches, and",
             "trident writes one coordinate grid per `(magnification, patch_size)`, so",
             "encoders on different grids share no row index and were never compared.",
             "", "All seven metrics are shown in every figure.", ""]
    for px in sorted(index):
        lines += [f"## {px}", "", f"_{GRID_NOTE.get(px, '')}_", "",
                  "| figure | cohort | magnification | encoders | source |",
                  "|---|---|---|---|---|"]
        for name, cohort, mag, n, mdir in sorted(index[px], key=lambda r: (r[1], int(r[2][:-1]))):
            rel = mdir.relative_to(root) if mdir.is_absolute() else mdir
            lines.append(f"| `{px}/{name}.{fmt}` | {cohort.upper()} | {mag} | {n} | `{rel}` |")
        lines.append("")
    lines += ["## Not represented", "",
              "- **384px** — MUSK is the only encoder on it, so similarity is undefined.",
              "- **5x/256px** — covered by the magnification series rather than a",
              "  standalone matrix; see `results/figures/magnification/`.", ""]
    (out / "similarity_by_grid" / "README.md").write_text("\n".join(lines))
    print(f"  similarity_by_grid/README.md  ({sum(len(v) for v in index.values())} figures indexed)")


def table_similarity_all(out: Path, fmt: str) -> None:
    """One table holding every similarity result, so all grids are comparable.

    Each row is a (grid, cohort, magnification) group; each metric column is the
    mean off-diagonal similarity for that group. Levels are only comparable
    within a column -- the metrics sit at different scales -- and only loosely
    across rows, since the groups hold different encoders.
    """
    found = _similarity_dirs()
    if not found:
        return
    rows = []
    for (px, cohort, mag), (n, mdir) in sorted(
        found.items(), key=lambda kv: (kv[0][0], kv[0][1], int(kv[0][2][:-1]))
    ):
        rec = {"grid": px, "cohort": cohort.upper(), "mag": mag,
               "encoders": n, "pairs": n * (n - 1) // 2}
        for m in METRICS:
            f = mdir / f"{m}.csv"
            if not f.exists():
                rec[m] = float("nan"); continue
            d = pd.read_csv(f, index_col=0).values
            off = d[~np.eye(d.shape[0], dtype=bool)]
            rec[m] = float(np.mean(off))
        rows.append(rec)
    df = pd.DataFrame(rows)
    dest = out / "similarity_by_grid"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "similarity_all_grids.csv", index=False)

    hdr = ["grid", "cohort", "mag", "encoders", "pairs"] + [clean_label(m) for m in METRICS]
    lines = ["# Every similarity result in one table", "",
             "Mean **off-diagonal** similarity per group.", "",
             "- **encoders** — how many encoders share that group's coordinate grid;",
             "  the similarity matrix is `encoders x encoders`.",
             "- **pairs** — distinct encoder pairs, `encoders x (encoders - 1) / 2`.",
             "  This is what each mean is actually averaged over, and it is why a",
             "  2-encoder row is a single number rather than a distribution.",
             "- The unit diagonal is excluded from every mean.", "",
             "Read down a column, not across a row: the seven metrics sit at different",
             "levels, and the groups hold different encoder sets, so a 2-encoder 512px",
             "row is not on the same footing as a 6-encoder 256px one.", "",
             "| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        npairs = r["encoders"] * (r["encoders"] - 1) // 2
        cells = [r["grid"], r["cohort"], r["mag"], str(r["encoders"]), str(npairs)]
        cells += ["—" if pd.isna(r[m]) else f"{r[m]:.3f}" for m in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Source: `similarity_all_grids.csv` in this folder.", ""]
    (dest / "similarity_all_grids.md").write_text("\n".join(lines))
    print(f"  similarity_by_grid/similarity_all_grids.{{csv,md}}  ({len(rows)} groups)")


def figs_slide_encoder_similarity(out: Path, fmt: str) -> None:
    """Similarity for the slide-level encoders, all seven metrics, per cohort.

    Slide encoders emit one vector per slide, so the patch-grid pairing
    constraint does not apply and every one of them is comparable at once --
    they are filed apart from the patch grids for exactly that reason. The
    trade-off is sample size: n is the slide count, not a patch count, against
    dimensions of 512-1280, which is close to the floor for the CCA family.
    """
    root = Path(__file__).resolve().parents[1]
    made = []
    for mdir in sorted(root.glob("results/slide_encoders/*/matrices")):
        cohort = mdir.parent.name
        label = "CPTAC" if "cptac" in cohort else "TCGA"
        mats = {clean_label(m): _load_matrix(mdir / f"{m}.csv")
                for m in METRICS if (mdir / f"{m}.csv").exists()}
        if not mats:
            continue
        n = next(iter(mats.values())).shape[0]
        # Wrap to a 4-wide grid rather than one 27in row: the panels keep the
        # same generous size the two-panel figures use, so the encoder labels
        # stay readable at eight encoders.
        fig = heatmap_row(
            mats, value_fmt="{:.2f}", mask="lower", ylab="Slide encoder",
            cbar_label="Similarity", rotate_xticks=40, n_cols=4,
            panel_size=(4.6, 4.9), label_size=8.5, base_size=11,
            suptitle=f"Slide encoders · {label} · {n} encoders",
        )
        dest = out / "slide_encoders" / f"{label.lower()}_slide_encoders.{fmt}"
        save_plot(fig, dest)
        plt.close(fig)
        made.append((label, n, dest, mdir))
        print(f"  slide_encoders/{dest.name}  ({n} encoders, {len(mats)} metrics)")

    if made:
        lines = ["# Slide-encoder similarity", "",
                 "All seven metrics, one figure per cohort.", "",
                 "Slide encoders emit one vector per slide, so the patch-grid pairing",
                 "constraint does not apply and all of them are comparable at once --",
                 "unlike the patch encoders, which split across four grids.", "",
                 "**Sample-size caveat.** n here is the number of *slides*, not patches,",
                 "against dimensions of 512-1280. That is ample for CKA and RSA but close",
                 "to the floor for SVCCA and PWCCA, which saturate as n approaches d.", "",
                 "| figure | cohort | encoders | source |", "|---|---|---|---|"]
        for label, n, dest, mdir in made:
            rel = mdir.relative_to(root) if mdir.is_absolute() else mdir
            lines.append(f"| `{dest.name}` | {label} | {n} | `{rel}` |")
        lines.append("")
        (out / "slide_encoders" / "README.md").write_text("\n".join(lines))
        print("  slide_encoders/README.md")


def figs_similarity_by_subcohort(out: Path, fmt: str, metric: str = "linear_cka") -> None:
    """Subcohort similarity: full per-subcohort figures plus tissue comparisons.

    Two shapes, because they answer different questions:

    ``<grid>/<group>_<subcohort>.<fmt>``
        All seven metrics for one (group, subcohort) -- the complete record.

    ``compare/<group>_<metric>.<fmt>``
        One panel per subcohort, same metric, same encoders. This is the figure
        the split exists for: encoders and magnification are held fixed and only
        tissue varies, so a difference between panels is a tissue effect rather
        than the composition artefact that pooling produces.
    """
    root = Path(__file__).resolve().parents[1]
    base = root / "results/full_run/analysis/similarity_by_subcohort"
    if not base.is_dir():
        print("  skip similarity-by-subcohort: nothing on disk")
        return

    by_group: dict[str, list[tuple[str, Path]]] = {}
    for mdir in sorted(base.glob("*/*/matrices")):
        group, sub = mdir.parent.parent.name, mdir.parent.name
        by_group.setdefault(group, []).append((sub, mdir))

    n_full = 0
    for group, entries in sorted(by_group.items()):
        m = re.search(r"_(\d+x)_(\d+px)$", group)
        grid = m.group(2) if m else "other"
        for sub, mdir in sorted(entries):
            mats = {clean_label(k): _load_matrix(mdir / f"{k}.csv")
                    for k in METRICS if (mdir / f"{k}.csv").exists()}
            if not mats:
                continue
            n = next(iter(mats.values())).shape[0]
            wide = max(3.0, min(4.4, 26.0 / len(mats)))
            fig = heatmap_row(
                mats, value_fmt="{:.2f}", mask="lower", ylab="Encoder",
                cbar_label="Similarity", rotate_xticks=40,
                panel_size=(wide, wide * 1.1), label_size=7.5, base_size=10,
                suptitle=f"{sub.replace('_', '-')} · {group} · {n} encoders",
            )
            save_plot(fig, out / "similarity_by_subcohort" / grid / f"{group}_{sub}.{fmt}")
            plt.close(fig)
            n_full += 1

        # tissue comparison: same metric, same encoders, one panel per subcohort
        panels = {}
        for sub, mdir in sorted(entries):
            f = mdir / f"{metric}.csv"
            if f.exists():
                panels[sub.replace("_", "-")] = _load_matrix(f)
        if len(panels) >= 2:
            n = next(iter(panels.values())).shape[0]
            fig = heatmap_row(
                panels, value_fmt="{:.2f}", mask="lower", ylab="Encoder",
                cbar_label=clean_label(metric), shared_limits=True,
                rotate_xticks=40, panel_size=(4.0, 4.4), label_size=8.0, base_size=11,
                suptitle=f"{clean_label(metric)} by subcohort · {group} · "
                         f"{n} encoders (shared colour scale)",
            )
            save_plot(fig, out / "similarity_by_subcohort" / "compare"
                      / f"{group}_{metric}.{fmt}")
            plt.close(fig)
            print(f"  similarity_by_subcohort/compare/{group}_{metric}.{fmt}  "
                  f"({len(panels)} subcohorts)")
    print(f"  similarity_by_subcohort: {n_full} per-subcohort figures")


#: The ImageNet-supervised control. Every "pathology encoders agree with each
#: other more than with the control" claim is measured against this row.
CONTROL_ENCODER = "ResNet50 (ImageNet)"


def fig_control_gap(out: Path, fmt: str, metric: str = "linear_cka") -> None:
    """How far the pathology encoders sit from the ImageNet control, per subcohort.

    This is the sanity check the shared-manifold argument rests on, and pooling
    inflates it: the pooled gap is larger than any individual subcohort's gap,
    because the control separates tissue types more sharply than the pathology
    encoders do, so mixing tissue pushes its geometry away from theirs. Plotting
    pooled and per-subcohort side by side is the only way to see that.
    """
    root = Path(__file__).resolve().parents[1]
    base = root / "results/full_run/analysis/similarity_by_subcohort"
    pooled = root / f"results/full_run/analysis/similarity/matrices/{metric}.csv"

    def gap(path: Path):
        d = pd.read_csv(path, index_col=0)
        if CONTROL_ENCODER not in d.index:
            return None
        enc = [e for e in d.index if e != CONTROL_ENCODER]
        pp = np.mean([d.loc[a, b] for i, a in enumerate(enc) for b in enc[i + 1:]])
        vc = np.mean([d.loc[CONTROL_ENCODER, e] for e in enc])
        return pp, vc

    rows = []
    if pooled.exists() and (g := gap(pooled)):
        rows.append(("Pooled\n(all tissue)", *g))
    for mdir in sorted(base.glob("cptac_benchmark_10x_256px/*/matrices")):
        f = mdir / f"{metric}.csv"
        if f.exists() and (g := gap(f)):
            rows.append((mdir.parent.name.replace("_", "-"), *g))
    if len(rows) < 2:
        print("  skip control-gap: needs pooled plus at least one subcohort")
        return

    long = pd.DataFrame(
        [{"subcohort": n, "series": s, "value": v * 100.0}
         for n, pp, vc in rows
         for s, v in (("Pathology ↔ pathology", pp), ("vs ImageNet control", vc))]
    )
    fig, ax = grouped_bars(
        long, x="subcohort", y="value", group="series",
        highlight="vs ImageNet control", ramp="blue",
        group_order=["Pathology ↔ pathology", "vs ImageNet control"],
        x_order=[r[0] for r in rows],
        ylab=f"Mean {clean_label(metric)} (%)",
        figsize=(2.0 * len(rows) + 3.0, 4.8), legend_ncol=2,
    )
    for i, (_, pp, vc) in enumerate(rows):
        ax.annotate(f"gap {pp - vc:+.3f}", xy=(i, max(pp, vc) * 100 + 3),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 100)
    emit(fig, out, f"fig_control_gap_{metric}", fmt)
    plt.close(fig)
    for n, pp, vc in rows:
        print(f"      {n.splitlines()[0]:14s} patho={pp:.3f} control={vc:.3f} gap={pp - vc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the MOSAIC figure set.")
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    parser.add_argument("--format", default="pdf")
    parser.add_argument("--headline-only", action="store_true",
                        help="skip the per-group and per-series figures")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Rendering figures -> {args.out}\n")
    fig_similarity(args.out, args.format)
    fig_retrieval(args.out, args.format)
    fig_downstream(args.out, args.format, "auc")
    fig_magnification(args.out, args.format)
    fig_transfer(args.out, args.format)
    fig_slide(args.out, args.format)
    fig_alignment(args.out, args.format)
    fig_retrieval_tcga(args.out, args.format)
    fig_alignment_tcga(args.out, args.format)
    fig_layerwise(args.out, args.format)
    if not args.headline_only:
        print("\nPer-group figures:")
        figs_all_groups(args.out, args.format)
        print("\nSimilarity by patch grid:")
        figs_similarity_by_grid(args.out, args.format)
        table_similarity_all(args.out, args.format)
        figs_slide_encoder_similarity(args.out, args.format)
        figs_similarity_by_subcohort(args.out, args.format)
        fig_control_gap(args.out, args.format)
        print("\nPer-series magnification figures:")
        figs_all_magnification(args.out, args.format)
        print("\nPer-baseline layer-wise figures:")
        figs_all_layerwise(args.out, args.format)
    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()
