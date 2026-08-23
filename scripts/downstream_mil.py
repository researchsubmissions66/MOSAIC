"""Slide-level downstream MIL across input conditions (Phase VIII).

Trains a MIL head on each of the three conditions and reports AUC, accuracy,
macro F1 and balanced accuracy on a patient-disjoint test split:

1. ``single``  — each encoder on its own
2. ``concat``  — all encoders concatenated per patch
3. ``shared``  — the fitted shared latent space

Examples
--------
    python scripts/downstream_mil.py --task tcga_nsclc --group master_benchmark/10x_256px \\
        --out results/phase8
    python scripts/downstream_mil.py --task tcga_brca_subtype --group master_benchmark/10x_256px \\
        --out results/phase8_brca --mil abmil transmil
    python scripts/downstream_mil.py --list-tasks
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.alignment import build_aligner  # noqa: E402
from utils.bags import bag_summary, build_bags  # noqa: E402
from utils.features import FeatureStore  # noqa: E402
from utils.labels import (  # noqa: E402
    TASK_REGISTRY,
    available_tasks,
    get_task,
    grouped_split,
)
from utils.mil import MILConfig, evaluate_mil, train_mil  # noqa: E402


def main() -> None:
    """Run the Phase VIII comparison."""
    parser = argparse.ArgumentParser(
        description="Phase VIII: downstream MIL across input conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list-tasks", action="store_true", help="list tasks and exit")
    parser.add_argument("--task", default=None, help="downstream task name")
    parser.add_argument("--group", default="best", help="feature-store group")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--encoders", nargs="+", default=None, help="restrict encoders")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["single", "concat", "shared"],
        help="input conditions to run",
    )
    parser.add_argument(
        "--mil",
        nargs="+",
        default=["abmil"],
        help="MIL heads; add 'transmil' or 'mean' to check ABMIL is not special",
    )
    parser.add_argument(
        "--aligner", default="gcca", help="aligner used for the shared condition"
    )
    parser.add_argument("--latent-dim", type=int, default=64, help="shared space size")
    parser.add_argument(
        "--max-patches", type=int, default=2000, help="patches kept per slide"
    )
    parser.add_argument("--epochs", type=int, default=50, help="max MIL epochs")
    parser.add_argument("--test-size", type=float, default=0.25, help="held-out patients")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--align-patches",
        type=int,
        default=20000,
        help="patches used to fit the aligner (training slides only)",
    )
    args = parser.parse_args()

    if args.list_tasks:
        print("Available downstream tasks:\n")
        for name in available_tasks():
            t = TASK_REGISTRY[name]
            print(f"  {name}")
            print(f"      cohort : {t.store_cohort}")
            print(f"      classes: {list(t.classes)}")
            print(f"      {t.description}")
            if t.notes:
                print(f"      note   : {t.notes}")
            print()
        return

    if not args.task or args.out is None:
        parser.error("--task and --out are required unless --list-tasks is given")

    store = FeatureStore()
    task = TASK_REGISTRY[args.task]
    group = (
        store.best_group(cohort=task.store_cohort)
        if args.group == "best"
        else store.group(args.group)
    )
    if group.cohort != task.store_cohort:
        parser.error(
            f"task {args.task!r} lives in {task.store_cohort} but --group is "
            f"{group.cohort}"
        )

    encoders = args.encoders or sorted(group.encoders)
    print(f"{group}")
    print(f"task: {args.task} — {task.description}")
    if task.notes:
        print(f"note: {task.notes}")

    labels = get_task(args.task, slide_ids=group.slides(encoders))
    print(f"\n{len(labels)} slides, {labels['patient_id'].nunique()} patients")
    print(labels["label"].value_counts().to_string())

    train_idx, test_idx = grouped_split(
        labels, test_size=args.test_size, seed=args.seed
    )
    train_slides = set(labels.iloc[train_idx]["slide_id"])
    print(f"split: {len(train_idx)} train / {len(test_idx)} test slides (patient-disjoint)")

    aligner = None
    if "shared" in args.conditions:
        print(f"\nFitting {args.aligner} on TRAINING slides only ...", end=" ", flush=True)
        t0 = time.perf_counter()
        fit_views = group.sample_patches(
            n_patches=args.align_patches,
            encoders=encoders,
            slides=sorted(train_slides),
            max_slides=200,
            seed=args.seed,
        )
        aligner = build_aligner(
            args.aligner, latent_dim=args.latent_dim, random_state=args.seed
        ).fit(fit_views)
        print(f"done ({time.perf_counter() - t0:.1f}s)")

    rows = []
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        enc_sets = (
            [[e] for e in encoders] if condition == "single" else [list(encoders)]
        )
        for enc_set in enc_sets:
            tag = enc_set[0] if condition == "single" else "all"
            print(f"\n=== {condition} [{tag}] ===")
            t0 = time.perf_counter()
            bags, y, groups, class_names = build_bags(
                group,
                labels,
                condition=condition,
                encoders=enc_set,
                aligner=aligner,
                max_patches=args.max_patches,
                seed=args.seed,
                verbose=True,
            )
            print(f"  loaded {len(bags)} bags, dim={bags[0].shape[1]} "
                  f"({time.perf_counter() - t0:.1f}s)")

            # build_bags may drop slides, so recompute the split on what loaded.
            loaded = pd.DataFrame({"slide_id": _loaded_ids(labels, group, enc_set)})
            tr = np.array([i for i, s in enumerate(loaded["slide_id"]) if s in train_slides])
            te = np.array([i for i, s in enumerate(loaded["slide_id"]) if s not in train_slides])
            if len(tr) == 0 or len(te) == 0:
                print("  skipping: empty split")
                continue

            for mil in args.mil:
                cfg = MILConfig(
                    model=mil, epochs=args.epochs, seed=args.seed, max_patches=args.max_patches
                )
                t1 = time.perf_counter()
                model, _ = train_mil(
                    [bags[i] for i in tr],
                    y[tr],
                    config=cfg,
                    groups=groups[tr],
                )
                scores = evaluate_mil(
                    model, [bags[i] for i in te], y[te], class_names=class_names
                )
                rows.append(
                    {
                        "condition": condition,
                        "encoders": tag,
                        "mil": mil,
                        "dim": bags[0].shape[1],
                        "auc": scores["auc"],
                        "accuracy": scores["accuracy"],
                        "f1": scores["f1"],
                        "balanced_accuracy": scores["balanced_accuracy"],
                        "n_test": scores["n_test"],
                        "seconds": round(time.perf_counter() - t1, 1),
                    }
                )
                print(
                    f"  {mil:9s} auc={scores['auc']:.4f} bacc={scores['balanced_accuracy']:.4f} "
                    f"f1={scores['f1']:.4f} acc={scores['accuracy']:.4f}"
                )

            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
            bag_summary(bags, y, class_names).to_csv(
                out / f"bags_{condition}_{tag}.csv", index=False
            )

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    print("\n=== Phase VIII results (patient-disjoint test split) ===")
    print(results.round(4).to_string(index=False))
    print(f"\nWrote results to {out}")


def _loaded_ids(labels: pd.DataFrame, group, encoders) -> list[str]:
    """Slide ids in the order build_bags emits them."""
    available = set(group.slides(list(encoders)))
    return [s for s in labels["slide_id"] if s in available]


if __name__ == "__main__":
    main()
