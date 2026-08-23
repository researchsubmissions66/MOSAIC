"""Representational similarity across pathology foundation models (Phase I).

Loads one embedding matrix per model, computes every registered similarity
metric over a shared patch subsample, and writes the matrices and figures to an
output directory.

Embeddings are expected as one file per model, all row-paired (patch i is the
same patch in every file, in the same order). ``.npy``, ``.npz`` and ``.pt``
are supported; the file stem becomes the model name.

Examples
--------
Real embeddings::

    python scripts/representation_similarity.py --emb-dir /path/to/embeddings \\
        --out results/phase1 --max-samples 5000

Synthetic smoke test (no data needed)::

    python scripts/representation_similarity.py --demo --out /tmp/phase1_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (  # noqa: E402
    compute_all_similarity_matrices,
    plot_clustered_heatmap,
    plot_metric_panel,
    plot_model_space,
    save_figure,
    stack_similarity_matrices,
)

SUPPORTED_SUFFIXES = {".npy", ".npz", ".pt", ".pth"}


def load_embedding(path: Path, key: str | None = None) -> np.ndarray:
    """Load one model's embedding matrix from disk.

    Parameters
    ----------
    path : pathlib.Path
        File to load. ``.npy`` / ``.npz`` are read with numpy, ``.pt`` /
        ``.pth`` with torch.
    key : str, optional
        Array key inside an ``.npz`` archive or a dict-valued ``.pt`` file.
        Defaults to the first entry.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_patches, n_features)``.
    """
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            arr = data[key or list(data.files)[0]]
    elif path.suffix in (".pt", ".pth"):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            obj = obj[key or next(iter(obj))]
        arr = obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)
    else:
        raise ValueError(f"unsupported embedding format: {path.suffix}")

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected 2-D embeddings, got shape {arr.shape}")
    return arr


def load_embedding_dir(emb_dir: Path) -> dict[str, np.ndarray]:
    """Load every supported embedding file in a directory, keyed by file stem.

    Parameters
    ----------
    emb_dir : pathlib.Path
        Directory containing one file per model.

    Returns
    -------
    dict of str to numpy.ndarray
        ``{model_name: embedding_matrix}``.
    """
    files = sorted(p for p in emb_dir.iterdir() if p.suffix in SUPPORTED_SUFFIXES)
    if not files:
        raise FileNotFoundError(
            f"no embedding files ({', '.join(sorted(SUPPORTED_SUFFIXES))}) in {emb_dir}"
        )

    reps: dict[str, np.ndarray] = {}
    for path in files:
        reps[path.stem] = load_embedding(path)
        print(f"  loaded {path.stem:>14s}  shape={reps[path.stem].shape}")
    return reps


def make_demo_representations(
    n_patches: int = 2000, seed: int = 0
) -> dict[str, np.ndarray]:
    """Synthesise five 'models' with a planted similarity structure.

    Three models are noisy linear views of a shared latent morphology manifold
    (with decreasing fidelity), one is a nonlinear view of it, and one is
    unrelated. Any correct metric should recover that grouping — this is the
    smoke test for the whole pipeline before real embeddings exist.

    Parameters
    ----------
    n_patches : int, default 2000
        Number of synthetic patches.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    dict of str to numpy.ndarray
        ``{model_name: embedding_matrix}``.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_patches, 16))

    def view(dim: int, noise: float) -> np.ndarray:
        W = rng.normal(size=(16, dim)) / np.sqrt(16)
        return latent @ W + noise * rng.normal(size=(n_patches, dim))

    return {
        "modelA_clean": view(384, 0.05),
        "modelB_clean": view(512, 0.10),
        "modelC_noisy": view(768, 0.60),
        "modelD_nonlin": np.tanh(2.0 * view(256, 0.10)),
        "modelE_unrelated": rng.normal(size=(n_patches, 640)),
    }


def load_from_feature_store(
    group_key: str,
    encoders: list[str] | None = None,
    n_patches: int = 20_000,
    max_slides: int | None = 200,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Sample row-paired patch features from the trident feature store.

    Parameters
    ----------
    group_key : str
        Group identifier, e.g. ``'cptac_benchmark/10x_256px'``, or ``'best'``
        to pick whichever group offers the most encoders.
    encoders : list of str, optional
        Restrict to these encoders.
    n_patches : int, default 20000
        Patches to sample.
    max_slides : int or None, default 200
        Cap on slides read.
    seed : int, default 0
        Sampling seed.

    Returns
    -------
    tuple
        ``({encoder: matrix}, display_names)``.
    """
    from utils.features import FeatureStore

    store = FeatureStore()
    group = store.best_group() if group_key == "best" else store.group(group_key)
    print(f"  group {group.key}: {sorted(group.encoders)}")

    reps = group.sample_patches(
        n_patches=n_patches,
        encoders=encoders,
        max_slides=max_slides,
        seed=seed,
        verbose=True,
    )
    names = list(reps)
    return reps, store.display_names(names)


def main() -> None:
    """Parse arguments, run Phase I, and write matrices and figures."""
    parser = argparse.ArgumentParser(
        description="Phase I representational similarity analysis across PFMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--emb-dir",
        type=Path,
        help="directory with one row-paired embedding file per model",
    )
    src.add_argument(
        "--group",
        type=str,
        help="feature-store group to load, e.g. 'cptac_benchmark/10x_256px'; "
        "use 'best' for the group with the most encoders "
        "(see scripts/scan_features.py)",
    )
    src.add_argument(
        "--demo",
        action="store_true",
        help="run on synthetic data with a known structure instead",
    )
    parser.add_argument(
        "--encoders", nargs="+", default=None, help="restrict --group to these encoders"
    )
    parser.add_argument(
        "--n-patches", type=int, default=20000, help="patches to sample for --group"
    )
    parser.add_argument(
        "--max-slides", type=int, default=200, help="slides to sample from for --group"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output directory for CSVs and figures"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="patches to subsample (shared across models); 0 disables",
    )
    parser.add_argument("--seed", type=int, default=0, help="subsampling seed")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="subset of metrics to run (default: all seven Phase I metrics)",
    )
    parser.add_argument(
        "--distance-mode",
        default="one_minus",
        choices=["one_minus", "angular", "sqrt_one_minus"],
        help="similarity-to-distance conversion for clustering and MDS",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="skip the UMAP panel (recommended when comparing few models)",
    )
    parser.add_argument("--format", default="png", help="figure format (png or pdf)")
    args = parser.parse_args()

    print("Loading representations...")
    labels = None
    if args.demo:
        reps = make_demo_representations()
    elif args.group:
        reps, labels = load_from_feature_store(
            args.group,
            encoders=args.encoders,
            n_patches=args.n_patches,
            max_slides=args.max_slides,
            seed=args.seed,
        )
    else:
        reps = load_embedding_dir(args.emb_dir)
    n_patches = next(iter(reps.values())).shape[0]
    print(f"{len(reps)} models, {n_patches} patches\n")

    max_samples = args.max_samples or None
    matrices = compute_all_similarity_matrices(
        reps,
        metrics=args.metrics,
        max_samples=max_samples,
        seed=args.seed,
        verbose=True,
    )

    if labels is not None:
        # Relabel with human-readable model names for the figures.
        for S in matrices.values():
            S.index = labels
            S.columns = labels

    out = args.out
    (out / "matrices").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    for metric, S in matrices.items():
        S.to_csv(out / "matrices" / f"{metric}.csv")

        fig = plot_model_space(
            S,
            distance_mode=args.distance_mode,
            include_umap=not args.no_umap,
            suptitle=metric,
        )
        save_figure(fig, out / "figures" / f"{metric}_model_space.{args.format}")

        grid = plot_clustered_heatmap(S, title=metric, vmin=0.0, vmax=1.0)
        save_figure(grid.figure, out / "figures" / f"{metric}_clustered.{args.format}")

    long = stack_similarity_matrices(matrices)
    long.to_csv(out / "matrices" / "pairwise_long.csv", index=False)

    panel = plot_metric_panel(matrices)
    save_figure(panel, out / "figures" / f"all_metrics_panel.{args.format}")

    # Do the metrics rank model pairs the same way? A Phase I result in itself.
    metric_cols = [c for c in long.columns if c not in ("model_a", "model_b")]
    if len(long) >= 3:
        agreement = long[metric_cols].corr(method="spearman")
        agreement.to_csv(out / "matrices" / "metric_agreement.csv")
        print("\nCross-metric agreement (Spearman over model pairs):")
        print(agreement.round(3).to_string())

    print(f"\nWrote {len(matrices)} matrices and figures to {out}")


if __name__ == "__main__":
    main()
