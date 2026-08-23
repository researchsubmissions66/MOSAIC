"""Run the full analysis suite on slide-level encoders.

Slide encoders emit one vector per slide, so the patch-grid pairing constraint
does not apply: encoders built on different ``(magnification, patch_size)``
grids still describe the same slides and are directly comparable. All six are
therefore analysed together.

There is no magnification sweep here — unlike the patch encoders, each slide
encoder exists at exactly one grid (CHIEF at 10x/256px, PRISM at 20x/224px,
TITAN and Feather at 20x/512px, and so on), so magnification is confounded with
encoder identity and cannot be varied independently. What *is* swept is the
cohort.

Stages: similarity (7 metrics), shared latent space, cross-model transfer, and
cross-model retrieval.

Examples
--------
    python scripts/slide_encoder_study.py --out results/slide_encoders
    python scripts/slide_encoder_study.py --out results/slide --cohorts master_benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.alignment import build_aligner, split_views  # noqa: E402
from utils.alignment_metrics import compare_aligners  # noqa: E402
from utils.features import FeatureStore  # noqa: E402
from utils.pairwise import compute_all_similarity_matrices  # noqa: E402
from utils.paperfigs import clean_labels, heatmap_row, save_plot  # noqa: E402
from utils.retrieval import compare_retrieval, retrieval_summary  # noqa: E402
from utils.transfer import evaluate_transfer, transfer_summary  # noqa: E402


def run_cohort(store: FeatureStore, cohort: str, args) -> None:
    """Run every slide-level analysis for one cohort."""
    sset = store.slide_encoders(cohort)
    print(f"\n{'=' * 70}\n{sset}\ngrids: {sset.grids}\n{'=' * 70}")

    views, slides = sset.load(
        encoders=args.encoders, max_slides=args.max_slides, verbose=True
    )
    print({k: v.shape for k, v in views.items()})

    out = args.out / cohort
    (out / "matrices").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pd.Series(slides, name="slide_id").to_csv(out / "slides.csv", index=False)

    # --- similarity -----------------------------------------------------
    print("\n--- similarity ---")
    mats = compute_all_similarity_matrices(
        views, max_samples=args.max_samples, seed=args.seed, verbose=False
    )
    for metric, M in mats.items():
        M.to_csv(out / "matrices" / f"{metric}.csv")
    print(mats["linear_cka"].round(3).to_string())

    panels = {}
    for metric in ["linear_cka", "kernel_cka", "svcca", "procrustes"]:
        if metric in mats:
            M = mats[metric].copy()
            M.index = clean_labels(M.index)
            M.columns = clean_labels(M.columns)
            panels[metric.replace("_", " ").title()] = M
    if panels:
        fig = heatmap_row(
            panels, value_fmt="{:.2f}", mask="lower", ylab="Slide encoder",
            cbar_label="Similarity", rotate_xticks=45,
        )
        save_plot(fig, out / "figures" / f"similarity_row.{args.format}")

    # --- alignment ------------------------------------------------------
    print("\n--- shared latent space ---")
    train, test, _, test_idx = split_views(views, test_size=args.test_size, seed=args.seed)
    aligners = {}
    for method in args.methods:
        print(f"  fitting {method} ...", end=" ", flush=True)
        aligners[method] = build_aligner(
            method, latent_dim=args.latent_dim, random_state=args.seed
        ).fit(train)
        print("done")

    summary, _ = compare_aligners(aligners, test, retrieval_samples=args.query_samples)
    summary.to_csv(out / "alignment_comparison.csv")
    cols = [c for c in ["reconstruction_r2", "alignment_error", "paired_cosine",
                        "recall@1", "neighborhood_preservation", "effective_rank"]
            if c in summary.columns]
    print(summary[cols].round(4).to_string())

    # --- transfer -------------------------------------------------------
    print("\n--- cross-model transfer ---")
    best = args.methods[-1]
    table = evaluate_transfer(
        aligners[best], test, max_samples=args.query_samples, seed=args.seed
    )
    table.to_csv(out / "transfer_pairs.csv", index=False)
    tsum = transfer_summary(table)
    tsum.to_csv(out / "transfer_summary.csv")
    keep = [c for c in tsum.columns if not c.startswith(("retrieval_n", "n_"))]
    print(tsum[keep].round(4).to_string())

    # --- retrieval ------------------------------------------------------
    print("\n--- cross-model retrieval ---")
    rtable = compare_retrieval(
        test, aligners, baseline_dim=args.latent_dim,
        max_samples=args.query_samples, seed=args.seed,
    )
    rtable.to_csv(out / "retrieval_pairs.csv", index=False)
    rsum = retrieval_summary(rtable)
    rsum.to_csv(out / "retrieval_summary.csv")
    print(rsum[[c for c in rsum.columns if not c.startswith("n_")]].round(4).to_string())

    print(f"\nWrote {cohort} results to {out}")


def main() -> None:
    """Run the slide-encoder study across cohorts."""
    parser = argparse.ArgumentParser(
        description="Full analysis suite on slide-level encoders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--cohorts", nargs="+", default=["master_benchmark", "cptac_benchmark"],
        help="cohorts to run",
    )
    parser.add_argument("--encoders", nargs="+", default=None, help="restrict encoders")
    parser.add_argument(
        "--max-slides", type=int, default=None, help="cap slides (default: all)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000,
        help="subsample for the O(n^2) metrics",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["joint_pca", "procrustes", "gcca"],
        help="aligners; the last is used for the transfer stage",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=32,
        help="shared space size. Keep well below the slide count: n is ~2200 "
             "slides here, not 50k patches",
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="held-out fraction")
    parser.add_argument("--query-samples", type=int, default=600, help="retrieval db size")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--format", default="png", help="figure format")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    store = FeatureStore()
    for cohort in args.cohorts:
        run_cohort(store, cohort, args)

    print(f"\nAll cohorts done -> {args.out}")


if __name__ == "__main__":
    main()
