"""Cross-model retrieval, with and without alignment (Phase VII).

Indexes a patch database with each model and queries it with every other,
comparing aligned shared spaces against unaligned controls of the same
dimensionality.

Examples
--------
    python scripts/cross_model_retrieval.py --group best --out results/phase7
    python scripts/cross_model_retrieval.py --group best --out results/phase7 --mode label \\
        --task cptac_cancer_type
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.alignment import build_aligner, split_views  # noqa: E402
from utils.features import FeatureStore  # noqa: E402
from utils.retrieval import compare_retrieval, retrieval_summary  # noqa: E402
from utils.visualization import plot_similarity_heatmap, save_figure  # noqa: E402


def main() -> None:
    """Run cross-model retrieval and write the comparison."""
    parser = argparse.ArgumentParser(
        description="Phase VII: cross-model retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--group", default="best", help="feature-store group")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--encoders", nargs="+", default=None, help="restrict encoders")
    parser.add_argument("--n-patches", type=int, default=20000, help="patches sampled")
    parser.add_argument("--max-slides", type=int, default=200, help="slides read")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["joint_pca", "gcca", "procrustes"],
        help="aligners to evaluate",
    )
    parser.add_argument("--latent-dim", type=int, default=64, help="shared space size")
    parser.add_argument(
        "--mode",
        default="identity",
        choices=["identity", "label"],
        help="relevance: the same patch, or any patch of the same class",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="task supplying slide labels for --mode label",
    )
    parser.add_argument(
        "--query-samples",
        type=int,
        default=2000,
        help="database size; recall depends on it, so hold it fixed across runs",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10], help="cutoffs")
    parser.add_argument("--test-size", type=float, default=0.3, help="held-out fraction")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--format", default="png", help="figure format")
    args = parser.parse_args()

    store = FeatureStore()
    group = store.best_group() if args.group == "best" else store.group(args.group)
    print(f"{group}\n")

    encoders = args.encoders or sorted(group.encoders)
    labels = None

    if args.mode == "label":
        if not args.task:
            parser.error("--mode label requires --task")
        views, labels = _sample_with_labels(
            group, encoders, args.task, args.n_patches, args.max_slides, args.seed
        )
    else:
        views = group.sample_patches(
            n_patches=args.n_patches,
            encoders=encoders,
            max_slides=args.max_slides,
            seed=args.seed,
            verbose=True,
        )

    # Aligners are fitted on train and every retrieval score is computed on the
    # held-out split, so no aligner has seen the patches it is scored on.
    train, test, _, test_idx = split_views(views, test_size=args.test_size, seed=args.seed)
    test_labels = labels[test_idx] if labels is not None else None
    print(f"\ntrain={next(iter(train.values())).shape[0]} "
          f"test={next(iter(test.values())).shape[0]} patches")

    aligners = {}
    for method in args.methods:
        print(f"Fitting {method} ...", end=" ", flush=True)
        aligners[method] = build_aligner(
            method, latent_dim=args.latent_dim, random_state=args.seed
        ).fit(train)
        print("done")

    print(f"\nScoring retrieval (mode={args.mode}) ...")
    table = compare_retrieval(
        test,
        aligners,
        ks=tuple(args.ks),
        labels=test_labels,
        mode=args.mode,
        baseline_dim=args.latent_dim,
        max_samples=args.query_samples,
        seed=args.seed,
    )

    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "retrieval_pairs.csv", index=False)

    summary = retrieval_summary(table)
    summary.to_csv(out / "retrieval_summary.csv")
    print("\n=== Cross-model retrieval (held-out, same-model pairs excluded) ===")
    cols = [c for c in summary.columns if not c.startswith("n_")]
    print(summary[cols].round(4).to_string())

    chance = 1.0 / min(args.query_samples, next(iter(test.values())).shape[0])
    if args.mode == "identity":
        print(f"\nchance recall@1 = {chance:.5f}")

    for condition in table["condition"].unique():
        sub = table[table["condition"] == condition]
        mat = sub.pivot(index="query", columns="database", values=f"recall@{args.ks[0]}")
        mat.to_csv(out / f"recall_matrix_{condition}.csv")
        fig, _ = plot_similarity_heatmap(
            mat,
            title=f"{condition}: recall@{args.ks[0]}",
            cbar_label=f"recall@{args.ks[0]}",
            vmin=0.0,
            vmax=1.0,
        )
        save_figure(fig, out / "figures" / f"recall_{condition}.{args.format}")

    print(f"\nWrote retrieval results to {out}")


def _sample_with_labels(
    group, encoders, task_name, n_patches, max_slides, seed
) -> tuple[dict, np.ndarray]:
    """Sample patches together with their slide's class label.

    Patches inherit the slide label, so label retrieval asks whether a patch
    retrieves patches from slides of the same class.
    """
    from utils.labels import get_task

    slides = group.slides(encoders)
    task = get_task(task_name, slide_ids=slides)
    if task.empty:
        raise SystemExit(f"task {task_name!r} has no slides in this feature group")

    rng = np.random.default_rng(seed)
    chosen = task
    if max_slides is not None and len(task) > max_slides:
        idx = rng.choice(len(task), size=max_slides, replace=False)
        chosen = task.iloc[np.sort(idx)]

    per_slide = max(1, int(np.ceil(n_patches / len(chosen))))
    chunks: dict[str, list] = {e: [] for e in encoders}
    label_chunks = []

    for row in chosen.itertuples(index=False):
        try:
            feats = group.load_slide(row.slide_id, encoders=encoders)
        except (ValueError, OSError):
            continue
        n = feats[encoders[0]].shape[0]
        take = min(per_slide, n)
        sel = np.sort(rng.choice(n, size=take, replace=False))
        for e in encoders:
            chunks[e].append(feats[e][sel].astype(np.float32, copy=False))
        label_chunks.append(np.full(take, row.label, dtype=object))

    views = {e: np.vstack(v) for e, v in chunks.items()}
    print(f"  {views[encoders[0]].shape[0]} patches from {len(label_chunks)} slides")
    return views, np.concatenate(label_chunks)


if __name__ == "__main__":
    main()
