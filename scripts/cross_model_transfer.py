"""Cross-model transfer: convert one encoder's embeddings into another's (Phase VI).

Encodes patches with a source model, decodes them as the target model, and asks
whether the result is usable where the target's real embeddings are usable —
by fidelity, by retrieval against the target's real index, and by a linear
probe trained on the target's real embeddings.

Only encoders sharing a coordinate grid can be transferred between, since the
aligner is fitted on row-paired patches. ``--list-pairs`` reports what is
possible.

Examples
--------
    python scripts/cross_model_transfer.py --list-pairs
    python scripts/cross_model_transfer.py --group best --out results/transfer
    python scripts/cross_model_transfer.py --group best --out results/transfer \\
        --pairs conch_v1:uni_v2 uni_v2:conch_v1 --task cptac_luad_tp53
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.alignment import build_aligner, split_views  # noqa: E402
from utils.features import FeatureStore  # noqa: E402
from utils.transfer import (  # noqa: E402
    evaluate_transfer,
    transfer_summary,
    transferable_pairs,
)
from utils.visualization import plot_similarity_heatmap, save_figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_model_retrieval import _sample_with_labels  # noqa: E402


def main() -> None:
    """Run the cross-model transfer evaluation."""
    parser = argparse.ArgumentParser(
        description="Phase VI: cross-model representation transfer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list-pairs", action="store_true", help="list transferable pairs and exit"
    )
    parser.add_argument("--group", default="best", help="feature-store group")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--encoders", nargs="+", default=None, help="restrict encoders")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="explicit 'source:target' pairs; default is every ordered pair",
    )
    parser.add_argument("--n-patches", type=int, default=20000, help="patches sampled")
    parser.add_argument("--max-slides", type=int, default=200, help="slides read")
    parser.add_argument(
        "--aligner", default="gcca", help="aligner providing the translation"
    )
    parser.add_argument("--latent-dim", type=int, default=64, help="shared space size")
    parser.add_argument(
        "--task",
        default=None,
        help="task supplying patch labels for the linear probe (slide label "
        "propagated to its patches); omitted, no probe is run",
    )
    parser.add_argument(
        "--query-samples", type=int, default=2000, help="retrieval database size"
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="held-out fraction")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--format", default="png", help="figure format")
    args = parser.parse_args()

    store = FeatureStore()

    if args.list_pairs:
        table = transferable_pairs(store)
        print("Transferable (source, target) pairs — same coordinate grid only:\n")
        for key, grp in table.groupby("group"):
            encoders = sorted(set(grp["source"]))
            print(f"  {key:30s} {len(grp):3d} ordered pairs over {encoders}")
        print(
            "\nPairs whose encoders live on different grids (e.g. keep@256px -> "
            "musk@384px, virchow@224px -> conch_v1@256px) are NOT listed: they "
            "require re-extracting one encoder onto the other's grid."
        )
        return

    if args.out is None:
        parser.error("--out is required unless --list-pairs is given")

    group = store.best_group() if args.group == "best" else store.group(args.group)
    encoders = args.encoders or sorted(group.encoders)
    print(f"{group}\n")

    pairs = None
    if args.pairs:
        pairs = []
        for spec in args.pairs:
            if ":" not in spec:
                parser.error(f"pair {spec!r} must be 'source:target'")
            src, tgt = spec.split(":", 1)
            missing = [e for e in (src, tgt) if e not in group.encoders]
            if missing:
                parser.error(
                    f"{missing} not in {group.key}; available: {sorted(group.encoders)}. "
                    "Encoders on different grids cannot be transferred between."
                )
            pairs.append((src, tgt))

    labels = None
    if args.task:
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

    train, test, _, test_idx = split_views(views, test_size=args.test_size, seed=args.seed)
    test_labels = labels[test_idx] if labels is not None else None
    print(f"\ntrain={next(iter(train.values())).shape[0]} "
          f"test={next(iter(test.values())).shape[0]} patches")

    print(f"Fitting {args.aligner} on the training split ...", end=" ", flush=True)
    aligner = build_aligner(
        args.aligner, latent_dim=args.latent_dim, random_state=args.seed
    ).fit(train)
    print("done\n")

    print("Evaluating transfer ...")
    table = evaluate_transfer(
        aligner,
        test,
        labels=test_labels,
        pairs=pairs,
        max_samples=args.query_samples,
        seed=args.seed,
        verbose=True,
    )

    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "transfer_pairs.csv", index=False)

    summary = transfer_summary(table)
    summary.to_csv(out / "transfer_summary.csv")

    cross = table[~table["self"]]
    show = [
        c
        for c in ["source", "target", "cosine", "r2", "retrieval_recall@1",
                  "retrieval_mrr", "probe_true", "probe_translated", "probe_ratio"]
        if c in table.columns
    ]
    print("\n=== Cross-model transfer (held-out patches) ===")
    print(cross[show].round(4).to_string(index=False))

    print("\n=== Summary: crossing models vs round-tripping the shared space ===")
    keep = [c for c in summary.columns if not c.startswith(("retrieval_n", "n_"))]
    print(summary[keep].round(4).to_string())

    for metric, label in [
        ("cosine", "cosine to the target's real embedding"),
        ("retrieval_recall@1", "recall@1 against the target's real index"),
    ]:
        if metric not in table.columns:
            continue
        mat = table.pivot(index="source", columns="target", values=metric)
        mat.to_csv(out / f"matrix_{metric.replace('@', '')}.csv")
        fig, _ = plot_similarity_heatmap(
            mat, title=f"{args.aligner}: {label}", cbar_label=metric, vmin=0.0, vmax=1.0
        )
        save_figure(
            fig, out / "figures" / f"transfer_{metric.replace('@', '')}.{args.format}"
        )

    print(f"\nWrote transfer results to {out}")


if __name__ == "__main__":
    main()
