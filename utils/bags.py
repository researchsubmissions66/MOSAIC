"""Building slide-level bags for Phase VIII from the feature store.

Turns ``(feature group, task)`` into the ``(bags, labels, groups)`` triple that
:func:`utils.mil.train_mil` consumes, under each of the three input conditions
the plan compares:

``single``
    One encoder's patch embeddings.
``concat``
    Every encoder's embeddings concatenated per patch. The dimensionality is
    the sum, so this is the condition with strictly the most information —
    and therefore the one the shared space has to beat to justify itself.
``shared``
    Patches mapped through a fitted aligner into the shared latent space,
    either per-encoder or as the cross-encoder consensus.

Bags are loaded slide by slide and can be capped with ``max_patches``: a single
slide can hold 20k patches at 20x, and a full cohort at full depth is hundreds
of gigabytes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["build_bags", "encode_bag", "bag_summary"]


def encode_bag(
    per_encoder: dict[str, np.ndarray],
    condition: str,
    encoders: Sequence[str],
    aligner=None,
    consensus: bool = True,
) -> np.ndarray:
    """Combine one slide's per-encoder patch features into a single bag matrix.

    Parameters
    ----------
    per_encoder : dict of str to numpy.ndarray
        ``{encoder: (n_patches, dim)}`` for one slide, row-paired.
    condition : {'single', 'concat', 'shared'}
        Input condition.
    encoders : sequence of str
        Encoders to use. For ``'single'`` only the first is used.
    aligner : BaseAligner, optional
        Required for ``'shared'``.
    consensus : bool, default True
        For ``'shared'``, average the encoders' projections into one embedding
        per patch. If False, concatenate them instead, which keeps more
        information but gives up the "one shared representation" claim.

    Returns
    -------
    numpy.ndarray
        Bag matrix of shape ``(n_patches, dim)``.
    """
    if condition == "single":
        return per_encoder[encoders[0]]

    if condition == "concat":
        return np.hstack([per_encoder[e] for e in encoders])

    if condition == "shared":
        if aligner is None:
            raise ValueError("condition='shared' requires a fitted aligner")
        zs = [aligner.transform_view(e, per_encoder[e]) for e in encoders]
        stacked = np.stack(zs, axis=0)
        return stacked.mean(axis=0) if consensus else np.hstack(zs)

    raise ValueError(
        f"unknown condition {condition!r}; expected 'single', 'concat' or 'shared'"
    )


def build_bags(
    group,
    labels: pd.DataFrame,
    condition: str = "single",
    encoders: Sequence[str] | None = None,
    aligner=None,
    consensus: bool = True,
    max_patches: int | None = 2000,
    dtype=np.float32,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Load slide bags for a task under one input condition.

    Parameters
    ----------
    group : FeatureGroup
        Feature group to read from.
    labels : pandas.DataFrame
        Table with ``slide_id``, ``label`` and ``patient_id``, as returned by
        :func:`utils.labels.get_task`.
    condition : {'single', 'concat', 'shared'}, default 'single'
        Input condition.
    encoders : sequence of str, optional
        Encoders to use. Defaults to all in the group.
    aligner : BaseAligner, optional
        Fitted aligner, required for ``'shared'``.
    consensus : bool, default True
        Average rather than concatenate the shared-space projections.
    max_patches : int or None, default 2000
        Cap on patches per slide, sampled without replacement. ``None`` keeps
        everything, which is far heavier: bags reach ~20k patches at 20x.
    dtype : numpy dtype, default ``np.float32``
        Bag dtype.
    seed : int, default 0
        Patch-subsampling seed.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    tuple
        ``(bags, y, groups, class_names)`` where ``y`` holds integer labels
        aligned to ``class_names`` and ``groups`` holds patient ids for
        leakage-free splitting. Slides that fail to load are skipped and
        excluded from all three arrays.
    """
    names = list(encoders) if encoders else sorted(group.encoders)
    rng = np.random.default_rng(seed)

    class_names = sorted(labels["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    available = set(group.slides(names))
    rows = labels[labels["slide_id"].isin(available)]
    if rows.empty:
        raise ValueError("no task slides are present in this feature group")
    if verbose:
        missing = len(labels) - len(rows)
        print(f"  {len(rows)} slides usable ({missing} not in the feature group)")

    bags, y, groups = [], [], []
    for i, row in enumerate(rows.itertuples(index=False)):
        try:
            per_encoder = group.load_slide(row.slide_id, encoders=names)
        except (ValueError, OSError) as exc:
            if verbose:
                print(f"  skipping {row.slide_id}: {exc}")
            continue

        n = per_encoder[names[0]].shape[0]
        if max_patches is not None and n > max_patches:
            sel = np.sort(rng.choice(n, size=max_patches, replace=False))
            per_encoder = {k: v[sel] for k, v in per_encoder.items()}

        bag = encode_bag(per_encoder, condition, names, aligner, consensus)
        bags.append(np.ascontiguousarray(bag, dtype=dtype))
        y.append(class_to_idx[row.label])
        groups.append(row.patient_id)

        if verbose and (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(rows)} slides")

    if not bags:
        raise ValueError("no slides could be loaded")
    return bags, np.array(y, dtype=np.int64), np.array(groups), class_names


def bag_summary(
    bags: Sequence[np.ndarray], y: np.ndarray, class_names: Sequence[str]
) -> pd.DataFrame:
    """Describe a loaded bag set.

    Parameters
    ----------
    bags : sequence of numpy.ndarray
        Slide bags.
    y : numpy.ndarray
        Integer labels.
    class_names : sequence of str
        Class names.

    Returns
    -------
    pandas.DataFrame
        One row per class with slide count and patch statistics.
    """
    sizes = np.array([b.shape[0] for b in bags])
    rows = []
    for i, name in enumerate(class_names):
        mask = y == i
        rows.append(
            {
                "class": name,
                "n_slides": int(mask.sum()),
                "mean_patches": float(sizes[mask].mean()) if mask.any() else 0.0,
                "min_patches": int(sizes[mask].min()) if mask.any() else 0,
                "max_patches": int(sizes[mask].max()) if mask.any() else 0,
            }
        )
    return pd.DataFrame(rows)
