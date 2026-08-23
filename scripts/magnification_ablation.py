"""Magnification ablation: repeat the similarity and alignment runs at 5x, 10x, 20x.

Runs the identical experiment at every magnification in a series — same
encoders, same slides, same metrics, same seed — and reports whether the
conclusions depend on the scale the tissue was looked at.

The encoder set is intersected across magnifications automatically, so a model
that was only extracted at one magnification (CONCH at 10x, in the CPTAC
store) is excluded rather than silently making one panel wider than the others.

Examples
--------
    python scripts/magnification_ablation.py --out results/ablation_mag
    python scripts/magnification_ablation.py --series cptac_benchmark/224px \\
        --out results/ablation_224 --n-patches 20000
    python scripts/magnification_ablation.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ablation import (  # noqa: E402
    magnification_stability,
    magnification_summary,
    rank_shift_report,
    run_alignment_ablation,
    run_similarity_ablation,
    similarity_trends,
)
from utils.features import FeatureStore  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_magnification_panel,
    plot_magnification_trends,
    plot_similarity_heatmap,
    save_figure,
)


def main() -> None:
    """Run the magnification ablation and write results."""
    parser = argparse.ArgumentParser(
        description="Repeat Phase I / Phase V across magnifications.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--series",
        default="best",
        help="series key, e.g. 'cptac_benchmark/256px', or 'best'",
    )
    parser.add_argument(
        "--list", action="store_true", help="list available series and exit"
    )
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument(
        "--n-patches", type=int, default=20000, help="patches per magnification"
    )
    parser.add_argument(
        "--max-slides", type=int, default=200, help="slides read per magnification"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="subsample for the O(n^2) metrics",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None, help="metrics (default: all seven)"
    )
    parser.add_argument(
        "--primary-metric",
        default="linear_cka",
        help="metric used for the stability and rank-shift reports",
    )
    parser.add_argument(
        "--alignment",
        nargs="*",
        default=["joint_pca", "gcca", "procrustes"],
        help="Phase V methods to also ablate; pass with no values to skip",
    )
    parser.add_argument("--latent-dim", type=int, default=64, help="shared space size")
    parser.add_argument("--seed", type=int, default=0, help="shared seed")
    parser.add_argument("--format", default="png", help="figure format")
    args = parser.parse_args()

    store = FeatureStore()

    if args.list:
        print("Available magnification series:\n")
        for s in store.magnification_series():
            mags = ", ".join(f"{m:g}x" for m in s.magnifications)
            print(f"  {s.key:28s} [{mags}]  {len(s.encoders)} encoders: {s.encoders}")
        return

    if args.out is None:
        parser.error("--out is required unless --list is given")

    series = (
        store.best_series()
        if args.series == "best"
        else _find_series(store, args.series)
    )
    print(series)
    print(f"shared slides: {len(series.shared_slides())}\n")

    results, views = run_similarity_ablation(
        series,
        metrics=args.metrics,
        n_patches=args.n_patches,
        max_slides=args.max_slides,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # Relabel with display names for figures and reports.
    names = store.display_names(series.encoders)
    for mats in results.values():
        for S in mats.values():
            S.index = names
            S.columns = names

    out = args.out
    (out / "matrices").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    for mag, mats in results.items():
        for metric, S in mats.items():
            S.to_csv(out / "matrices" / f"{metric}_{mag:g}x.csv")

    trends = similarity_trends(results)
    trends.to_csv(out / "matrices" / "similarity_trends.csv", index=False)

    summary = magnification_summary(results)
    summary.to_csv(out / "magnification_summary.csv")
    print("\n=== Mean off-diagonal similarity by magnification ===")
    print(summary.round(4).to_string())

    primary = args.primary_metric
    if primary in next(iter(results.values())):
        stability = magnification_stability(results, metric=primary)
        stability.to_csv(out / "magnification_stability.csv")
        print(
            f"\n=== Structure stability ({primary}, Spearman over model pairs) ===\n"
            "Does the ranking of model pairs survive a change of magnification?"
        )
        print(stability.round(4).to_string())

        shifts = rank_shift_report(results, metric=primary)
        shifts.to_csv(out / "rank_shifts.csv")
        print(f"\n=== Most magnification-sensitive pairs ({primary}) ===")
        print(shifts.round(4).head(10).to_string())

        fig = plot_magnification_panel(
            results, metric=primary, suptitle=f"{series.key}: {primary}"
        )
        save_figure(fig, out / "figures" / f"panel_{primary}.{args.format}")

        fig, _ = plot_similarity_heatmap(
            stability,
            title=f"Similarity-structure agreement across magnification ({primary})",
            cbar_label="Spearman",
            vmin=0.0,
            vmax=1.0,
        )
        save_figure(fig, out / "figures" / f"stability_{primary}.{args.format}")

    for metric in sorted(trends["metric"].unique()):
        fig, _ = plot_magnification_trends(
            trends, metric=metric, title=f"{series.key}: {metric}"
        )
        save_figure(fig, out / "figures" / f"trends_{metric}.{args.format}")

    if args.alignment:
        print("\n=== Phase V across magnification ===")
        align = run_alignment_ablation(
            views,
            methods=args.alignment,
            latent_dim=args.latent_dim,
            seed=args.seed,
            retrieval_samples=2000,
        )
        align.to_csv(out / "alignment_by_magnification.csv")
        cols = [
            "reconstruction_r2",
            "alignment_error",
            "paired_cosine",
            "recall@1",
            "neighborhood_preservation",
            "effective_rank",
        ]
        print(align[[c for c in cols if c in align.columns]].round(4).to_string())

    print(f"\nWrote ablation results to {out}")


def _find_series(store: FeatureStore, key: str):
    """Look up a magnification series by key, with a helpful error."""
    for s in store.magnification_series():
        if s.key == key:
            return s
    available = [s.key for s in store.magnification_series()]
    raise SystemExit(f"unknown series {key!r}; available: {available}")


if __name__ == "__main__":
    main()
