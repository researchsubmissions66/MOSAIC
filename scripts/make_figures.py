"""Render the publication figures from a completed study run.

Reads the CSVs written by ``run_study.py`` and emits figures in the reference
visual language: heatmap rows with per-panel colourbars, and grouped bar panels
with in-bar value labels.

Examples
--------
    python scripts/make_figures.py --run results/full_run --out results/figures
    python scripts/make_figures.py --run results/full_run --out results/figures --format pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paperfigs import (  # noqa: E402
    clean_label,
    clean_labels,
    grouped_bars,
    heatmap_row,
    save_plot,
)

#: Task names that read better with an explicit line break in a bar chart.
TASK_TITLES = {
    "tcga_brca_subtype": "TCGA BRCA\nIDC/ILC",
    "tcga_brca_stage": "TCGA BRCA\nStage",
    "tcga_nsclc_stage": "TCGA NSCLC\nStage",
}


def _find(run: Path, *relative: str) -> Path | None:
    """Locate a stage directory across the two possible run layouts.

    A single ``run_study.py`` invocation writes ``<run>/<stage>/``; splitting
    the run across GPUs produces ``<run>/analysis/<stage>/`` and
    ``<run>/downstream/<stage>/``. Both are searched so figures render either
    way.
    """
    for prefix in ((), ("analysis",), ("downstream",)):
        candidate = run.joinpath(*prefix, *relative)
        if candidate.exists():
            return candidate
    return None


def similarity_figure(run: Path, out: Path, fmt: str, metrics: list[str]) -> None:
    """Heatmap row of encoder-by-encoder similarity, one panel per metric."""
    mdir = _find(run, "similarity", "matrices")
    if mdir is None:
        print("  skip similarity: no similarity/matrices found")
        return

    mats = {}
    for metric in metrics:
        f = mdir / f"{metric}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col=0)
        df.index = clean_labels(df.index)
        df.columns = clean_labels(df.columns)
        mats[clean_label(metric)] = df

    if not mats:
        print("  skip similarity: no metric CSVs")
        return

    fig = heatmap_row(
        mats,
        value_fmt="{:.2f}",
        mask="lower",
        ylab="Encoder",
        cbar_label="Similarity",
        rotate_xticks=45,
    )
    save_plot(fig, out / f"similarity_row.{fmt}")
    print(f"  similarity_row.{fmt}  ({len(mats)} panels)")


def magnification_figure(run: Path, out: Path, fmt: str, metric: str) -> None:
    """Heatmap row of the same similarity matrix at each magnification."""
    mdir = _find(run, "magnification", "matrices")
    if mdir is None:
        print("  skip magnification: no magnification/matrices found")
        return

    mats = {}
    for f in sorted(mdir.glob(f"{metric}_*x.csv")):
        mag = f.stem.split("_")[-1]
        df = pd.read_csv(f, index_col=0)
        df.index = clean_labels(df.index)
        df.columns = clean_labels(df.columns)
        mats[mag] = df

    if not mats:
        print("  skip magnification: no matrices")
        return

    # Shared limits here: the whole point is that the *level* shifts with
    # magnification, which independent scales would hide.
    fig = heatmap_row(
        mats,
        value_fmt="{:.2f}",
        mask="lower",
        ylab="Encoder",
        cbar_label=clean_label(metric),
        shared_limits=True,
        rotate_xticks=45,
    )
    save_plot(fig, out / f"magnification_{metric}.{fmt}")
    print(f"  magnification_{metric}.{fmt}  ({len(mats)} magnifications)")


def downstream_figure(run: Path, out: Path, fmt: str, metric: str) -> None:
    """Grouped bars: every encoder, concat and the shared space, per task."""
    root = _find(run, "downstream") or run
    files = sorted(root.glob("*/results.csv")) or sorted(root.glob("*/*/results.csv"))
    if not files:
        print(f"  skip downstream: no results under {root}")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["task"] = f.parent.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # One bar per input condition; single-encoder rows keep their encoder name.
    data["series"] = np.where(
        data["condition"] == "single",
        data["encoders"],
        data["condition"].map({"concat": "Concat", "shared": "MOSAIC"}),
    )
    # Registry keys must never reach a figure label.
    data["series"] = data["series"].map(clean_label)
    data["value"] = data[metric] * 100.0
    data["task_label"] = data["task"].map(
        lambda t: TASK_TITLES.get(t, clean_label(t))
    )

    # Average over MIL heads so the bar is about the representation, not the
    # classifier; the per-head split lives in the CSV.
    agg = (
        data.groupby(["task_label", "series"], as_index=False)["value"].mean()
    )

    order = sorted(s for s in agg["series"].unique() if s not in ("Concat", "MOSAIC"))
    order += ["Concat", "MOSAIC"]

    fig, _ = grouped_bars(
        agg,
        x="task_label",
        y="value",
        group="series",
        highlight="MOSAIC",
        group_order=order,
        ramp="blue",
        ylab=f"{clean_label(metric)} (%)",
        figsize=(1.9 * agg["task_label"].nunique() + 3.0, 4.8),
        legend_ncol=3,
    )
    save_plot(fig, out / f"downstream_{metric}.{fmt}")
    print(f"  downstream_{metric}.{fmt}  ({agg['task_label'].nunique()} tasks)")


def retrieval_figure(run: Path, out: Path, fmt: str) -> None:
    """Grouped bars comparing aligned and unaligned cross-model retrieval."""
    f = _find(run, "retrieval", "retrieval_summary.csv")
    if f is None:
        print("  skip retrieval: no retrieval_summary.csv found")
        return

    df = pd.read_csv(f, index_col=0).reset_index()
    df = df.rename(columns={df.columns[0]: "condition"})

    keep = [c for c in ("recall@1", "recall@5", "recall@10", "map", "mrr", "ndcg")
            if c in df.columns]
    long = df.melt(
        id_vars="condition", value_vars=keep, var_name="metric", value_name="value"
    )
    long["value"] *= 100.0
    long["metric"] = long["metric"].map(clean_label)
    long["condition"] = long["condition"].map(clean_label)

    fig, _ = grouped_bars(
        long,
        x="metric",
        y="value",
        group="condition",
        highlight="GCCA",
        ramp="purple",
        ylab="Score (%)",
        figsize=(1.7 * len(keep) + 3.0, 4.6),
        legend_ncol=2,
    )
    save_plot(fig, out / f"retrieval.{fmt}")
    print(f"  retrieval.{fmt}")


def main() -> None:
    """Render every figure the run has data for."""
    parser = argparse.ArgumentParser(
        description="Render MOSAIC publication figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", type=Path, required=True, help="study output directory")
    parser.add_argument("--out", type=Path, required=True, help="figure directory")
    parser.add_argument("--format", default="pdf", help="figure format (pdf or png)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["linear_cka", "kernel_cka", "svcca", "procrustes"],
        help="similarity metrics to panel",
    )
    parser.add_argument(
        "--primary-metric", default="linear_cka", help="metric for the magnification row"
    )
    parser.add_argument(
        "--downstream-metric", default="auc", help="auc, balanced_accuracy, f1 ..."
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Rendering figures from {args.run} -> {args.out}\n")

    similarity_figure(args.run, args.out, args.format, args.metrics)
    magnification_figure(args.run, args.out, args.format, args.primary_metric)
    downstream_figure(args.run, args.out, args.format, args.downstream_metric)
    retrieval_figure(args.run, args.out, args.format)

    print(f"\nDone. Figures in {args.out}")


if __name__ == "__main__":
    main()
