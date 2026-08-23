"""Inventory the feature store: which encoders are available, where, and paired.

Writes ``configs/feature_inventory.json`` (machine-readable, consumed by nothing
but useful as a record of what the analysis was run against) and prints a table
sorted by how many models each group lets you compare.

Optionally verifies that coordinate grids really are identical within a group —
the assumption every downstream phase depends on.

Examples
--------
    python scripts/scan_features.py
    python scripts/scan_features.py --verify 3
    python scripts/scan_features.py --feature-root /path/to/.../trident_features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.features import FeatureStore  # noqa: E402


def main() -> None:
    """Scan the feature store and report what is available."""
    parser = argparse.ArgumentParser(
        description="Inventory trident-extracted features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="encoder registry")
    parser.add_argument(
        "--feature-root", type=Path, default=None, help="override the store root"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "feature_inventory.json",
        help="where to write the inventory JSON",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        help="verify coordinate alignment on this many slides per group",
    )
    parser.add_argument(
        "--no-slides",
        action="store_true",
        help="skip shared-slide counting (much faster)",
    )
    parser.add_argument(
        "--show-unknown",
        action="store_true",
        help="also report encoders missing from the registry",
    )
    args = parser.parse_args()

    store = FeatureStore(
        config=args.config,
        feature_root=args.feature_root,
        known_only=not args.show_unknown,
    )
    print(store, "\n")

    print("=== Registered encoders ===")
    print(store.describe_encoders().to_string(), "\n")

    print("=== Feature groups (row-pairable sets) ===")
    df = store.inventory(count_slides=not args.no_slides)
    print(df.to_string(index=False), "\n")

    best = store.best_group()
    print(f"Widest comparison: {best.key} with {best.n_encoders} encoders")
    print(f"  {sorted(best.encoders)}\n")

    verified: dict[str, object] = {}
    if args.verify:
        print(f"=== Verifying coordinate alignment ({args.verify} slides/group) ===")
        for key, group in sorted(store.groups.items()):
            if group.n_encoders < 2:
                continue
            slides = group.slides()[: args.verify]
            if not slides:
                verified[key] = "no shared slides"
                print(f"{key:36s} no shared slides")
                continue
            results = [group.verify_alignment(s) for s in slides]
            ok = all(results)
            verified[key] = ok
            print(
                f"{key:36s} {'OK' if ok else 'MISALIGNED'} "
                f"({sum(results)}/{len(results)} slides)"
            )
        print()

    payload = {
        "feature_root": str(store.feature_root),
        "groups": json.loads(df.to_json(orient="records")),
        "widest_group": best.key,
        "coordinate_verification": verified,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote inventory to {args.out}")


if __name__ == "__main__":
    main()
