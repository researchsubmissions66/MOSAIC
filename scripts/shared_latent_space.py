"""Build and evaluate shared latent spaces across foundation models (Phase V).

Fits every requested aligner on a training split of row-paired embeddings,
evaluates all of them on the same held-out patches, and writes a comparison
table, per-method reports, and figures.

Examples
--------
Real embeddings::

    python scripts/shared_latent_space.py --emb-dir /path/to/embeddings \\
        --out results/phase5 --latent-dim 64

Synthetic smoke test (no data needed)::

    python scripts/shared_latent_space.py --demo --out /tmp/phase5_demo --latent-dim 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.alignment import available_aligners, build_aligner, split_views  # noqa: E402
from utils.alignment_metrics import (  # noqa: E402
    compare_aligners,
    cross_model_transfer_scores,
)
from utils.visualization import (  # noqa: E402
    plot_similarity_heatmap,
    save_figure,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from representation_similarity import load_embedding_dir, load_from_feature_store  # noqa: E402

#: Per-method overrides applied on top of the shared latent_dim.
METHOD_KWARGS: dict[str, dict] = {
    "mcca": {"pca_dim": 128},
    "autoencoder": {"hidden_dims": (512, 256), "epochs": 150},
    "optimal_transport": {"n_restarts": 3},
}


def make_demo_views(
    n_patches: int = 4000,
    latent_k: int = 24,
    n_clusters: int = 20,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Synthesise models sharing a clustered latent manifold.

    The latent factor is a Gaussian mixture rather than a single Gaussian,
    which matters: real patch embeddings cluster by tissue type, and that
    cluster structure is what makes rotations identifiable. A single isotropic
    Gaussian is rotationally symmetric, so unsupervised alignment is provably
    impossible on it — a smoke test built that way would make the OT method
    look broken when it is merely being asked something unanswerable.

    Parameters
    ----------
    n_patches : int, default 4000
        Number of synthetic patches.
    latent_k : int, default 24
        True shared latent dimensionality.
    n_clusters : int, default 20
        Number of mixture components ("tissue types").
    seed : int, default 0
        RNG seed.

    Returns
    -------
    dict of str to numpy.ndarray
        Four models: three linear views of the shared latent (one noisy), one
        nonlinear view, plus an unrelated control.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, latent_k)) * 3.0
    labels = rng.integers(0, n_clusters, size=n_patches)
    latent = centers[labels] + rng.normal(size=(n_patches, latent_k)) * 0.5

    def view(dim: int, noise: float) -> np.ndarray:
        W = rng.normal(size=(latent_k, dim)) / np.sqrt(latent_k)
        return latent @ W + noise * rng.normal(size=(n_patches, dim))

    return {
        "modelA": view(384, 0.10),
        "modelB": view(512, 0.15),
        "modelC_noisy": view(768, 0.60),
        "modelD_nonlin": np.tanh(1.5 * view(256, 0.10)),
        "modelE_unrelated": rng.normal(size=(n_patches, 320)),
    }


def main() -> None:
    """Parse arguments, fit and evaluate aligners, write outputs."""
    parser = argparse.ArgumentParser(
        description="Phase V shared-latent-space construction and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--emb-dir", type=Path, help="directory of row-paired embeddings")
    src.add_argument(
        "--group",
        type=str,
        help="feature-store group, e.g. 'cptac_benchmark/10x_256px', or 'best'",
    )
    src.add_argument("--demo", action="store_true", help="use synthetic data")
    parser.add_argument(
        "--encoders", nargs="+", default=None, help="restrict --group to these encoders"
    )
    parser.add_argument(
        "--n-patches", type=int, default=50000, help="patches to sample for --group"
    )
    parser.add_argument(
        "--max-slides", type=int, default=400, help="slides to sample from for --group"
    )

    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--latent-dim", type=int, default=64, help="shared space size")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["joint_pca", "gcca", "mcca", "procrustes"],
        help=f"aligners to run; choose from {available_aligners()}",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="held-out patch fraction"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="cap on training patches (subsampled)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--save-aligners", action="store_true", help="persist fitted aligners"
    )
    parser.add_argument("--format", default="png", help="figure format")
    args = parser.parse_args()

    print("Loading representations...")
    if args.demo:
        views = make_demo_views()
    elif args.group:
        views, _ = load_from_feature_store(
            args.group,
            encoders=args.encoders,
            n_patches=args.n_patches,
            max_slides=args.max_slides,
            seed=args.seed,
        )
    else:
        views = load_embedding_dir(args.emb_dir)
    n = next(iter(views.values())).shape[0]
    print(f"{len(views)} models, {n} patches")

    train, test, _, _ = split_views(views, test_size=args.test_size, seed=args.seed)
    if args.max_train is not None:
        n_train = next(iter(train.values())).shape[0]
        if n_train > args.max_train:
            idx = np.random.default_rng(args.seed).choice(
                n_train, size=args.max_train, replace=False
            )
            train = {k: v[np.sort(idx)] for k, v in train.items()}
    n_train = next(iter(train.values())).shape[0]
    n_test = next(iter(test.values())).shape[0]
    print(f"train={n_train} patches, test={n_test} patches\n")

    aligners = {}
    for method in args.methods:
        kwargs = {"latent_dim": args.latent_dim, "random_state": args.seed}
        kwargs.update(METHOD_KWARGS.get(method, {}))
        print(f"Fitting {method}...", end=" ", flush=True)
        t0 = time.perf_counter()
        aligners[method] = build_aligner(method, **kwargs).fit(train)
        print(f"done ({time.perf_counter() - t0:.1f}s)")

    print("\nEvaluating on held-out patches...")
    summary, reports = compare_aligners(aligners, test)

    out = args.out
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    summary.to_csv(out / "aligner_comparison.csv")
    print("\n=== Shared latent space comparison (held-out) ===")
    print(summary.round(4).to_string())

    for method, report in reports.items():
        d = out / "reports" / method
        d.mkdir(parents=True, exist_ok=True)
        report["reconstruction"].to_csv(d / "reconstruction.csv")
        report["paired_cosine"].to_csv(d / "paired_cosine.csv")
        report["shared_cka"].to_csv(d / "shared_cka.csv")
        report["retrieval"].to_csv(d / "retrieval.csv", index=False)
        report["neighborhood"].to_csv(d / "neighborhood.csv")

        transfer = cross_model_transfer_scores(aligners[method], test)
        transfer.to_csv(d / "cross_model_transfer.csv", index=False)

        fig, _ = plot_similarity_heatmap(
            report["shared_cka"],
            title=f"{method}: linear CKA in the shared space",
            vmin=0.0,
            vmax=1.0,
        )
        save_figure(fig, out / "figures" / f"{method}_shared_cka.{args.format}")

        fig, _ = plot_similarity_heatmap(
            report["paired_cosine"],
            title=f"{method}: paired cosine in the shared space",
            vmin=0.0,
            vmax=1.0,
        )
        save_figure(fig, out / "figures" / f"{method}_paired_cosine.{args.format}")

        if args.save_aligners:
            aligners[method].save(out / "reports" / method / "aligner.joblib")

    _write_diagnostics(aligners, out)

    print(f"\nWrote comparison, per-method reports and figures to {out}")


def _write_diagnostics(aligners: dict, out: Path) -> None:
    """Persist the method-specific diagnostics worth reporting in the paper."""
    for method, aligner in aligners.items():
        d = out / "reports" / method

        if hasattr(aligner, "eigenvalues_"):
            pd.DataFrame(
                {
                    "component": np.arange(len(aligner.eigenvalues_)),
                    "eigenvalue": aligner.eigenvalues_,
                    "view_agreement": getattr(
                        aligner,
                        "view_agreement_",
                        getattr(aligner, "correlations_", np.nan),
                    ),
                }
            ).to_csv(d / "spectrum.csv", index=False)

        if hasattr(aligner, "view_residuals_"):
            pd.Series(aligner.view_residuals_, name="residual").to_csv(
                d / "view_residuals.csv"
            )
        if hasattr(aligner, "encoder_r2_"):
            pd.DataFrame(
                {
                    "encoder_r2": pd.Series(aligner.encoder_r2_),
                    "loading_mass": pd.Series(aligner.view_loadings_),
                }
            ).to_csv(d / "view_contributions.csv")
        if hasattr(aligner, "matching_accuracy_"):
            pd.Series(aligner.matching_accuracy_, name="matching_accuracy").to_csv(
                d / "ot_matching_accuracy.csv"
            )
        if hasattr(aligner, "history_"):
            pd.DataFrame(
                {k: v for k, v in aligner.history_.items() if v}
            ).to_csv(d / "training_history.csv", index=False)


if __name__ == "__main__":
    main()
