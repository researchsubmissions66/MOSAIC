"""Driver for building N x N similarity matrices across foundation models.

Phase I's deliverable is a model-by-model similarity matrix per metric. This
module owns the bookkeeping around that: a registry of the seven metrics, a
consistent subsampling policy (the same patch indices for every model, so rows
stay paired), symmetry exploitation, and conversion of similarities into the
distance matrices consumed by clustering, MDS and UMAP.

Typical use
-----------
>>> reps = {"UNI": uni_emb, "CONCH": conch_emb, "Virchow": virchow_emb}
>>> S = compute_similarity_matrix(reps, "linear_cka")            # doctest: +SKIP
>>> mats = compute_all_similarity_matrices(reps, max_samples=5000)  # doctest: +SKIP
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cca import mean_cca_correlation, pwcca, svcca
from .cka import kernel_cka, linear_cka
from .cosine import cosine_rsa_similarity
from .distance_correlation import distance_correlation
from .preprocessing import as_matrix, subsample_indices
from .procrustes import orthogonal_procrustes_similarity

__all__ = [
    "METRIC_REGISTRY",
    "SYMMETRIC_METRICS",
    "QUADRATIC_METRICS",
    "available_metrics",
    "get_metric",
    "compute_similarity",
    "compute_similarity_matrix",
    "compute_all_similarity_matrices",
    "similarity_to_distance",
    "stack_similarity_matrices",
]


#: Name -> callable mapping for every Phase I metric. Each callable takes two
#: row-paired representation matrices and returns a scalar similarity, with all
#: remaining arguments supplied as keywords.
METRIC_REGISTRY: dict[str, Callable[..., float]] = {
    "linear_cka": linear_cka,
    "kernel_cka": kernel_cka,
    "svcca": svcca,
    "pwcca": pwcca,
    "procrustes": orthogonal_procrustes_similarity,
    "cosine_rsa": cosine_rsa_similarity,
    "distance_correlation": distance_correlation,
    "mean_cca": mean_cca_correlation,
}

#: Metrics that are symmetric in their arguments, so only the upper triangle
#: needs computing. ``pwcca`` qualifies only because its default
#: ``symmetric=True`` averages both directions.
SYMMETRIC_METRICS: frozenset[str] = frozenset(
    {
        "linear_cka",
        "kernel_cka",
        "svcca",
        "procrustes",
        "cosine_rsa",
        "distance_correlation",
        "mean_cca",
        "pwcca",
    }
)

#: Metrics whose memory grows as O(n^2) in the number of patches. The driver
#: warns if these are run without subsampling.
QUADRATIC_METRICS: frozenset[str] = frozenset(
    {"kernel_cka", "cosine_rsa", "distance_correlation"}
)


def available_metrics() -> list[str]:
    """List the registered metric names.

    Returns
    -------
    list of str
        Sorted metric names accepted by :func:`compute_similarity` and
        :func:`compute_similarity_matrix`.
    """
    return sorted(METRIC_REGISTRY)


def get_metric(name: str) -> Callable[..., float]:
    """Look up a metric callable by name.

    Parameters
    ----------
    name : str
        One of :func:`available_metrics`.

    Returns
    -------
    callable
        The metric function.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    try:
        return METRIC_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown metric {name!r}; available: {available_metrics()}"
        ) from None


def compute_similarity(X, Y, metric: str = "linear_cka", **kwargs) -> float:
    """Compute a single similarity value between two representations.

    Parameters
    ----------
    X, Y : array-like
        Row-paired representation matrices.
    metric : str, default 'linear_cka'
        One of :func:`available_metrics`.
    **kwargs
        Passed through to the underlying metric (e.g. ``threshold`` for
        ``kernel_cka``, ``var_threshold`` for ``svcca``).

    Returns
    -------
    float
        Similarity value.
    """
    return float(get_metric(metric)(X, Y, **kwargs))


def _prepare_representations(
    representations: Mapping[str, object],
    max_samples: int | None,
    seed: int | None,
    dtype,
) -> tuple[list[str], list[np.ndarray]]:
    """Coerce, validate and jointly subsample a dict of model representations.

    Applies one shared index set across all models so rows remain paired.
    """
    names = list(representations)
    if len(names) < 2:
        raise ValueError(
            f"need at least 2 models to build a similarity matrix, got {len(names)}"
        )

    mats = [as_matrix(representations[name], dtype=dtype) for name in names]

    n_rows = {m.shape[0] for m in mats}
    if len(n_rows) != 1:
        detail = ", ".join(f"{n}={m.shape[0]}" for n, m in zip(names, mats))
        raise ValueError(
            "all models must encode the same patches in the same order; "
            f"got differing sample counts: {detail}"
        )

    n = mats[0].shape[0]
    idx = subsample_indices(n, max_samples, seed=seed)
    if idx is not None:
        mats = [m[idx] for m in mats]
    return names, mats


def compute_similarity_matrix(
    representations: Mapping[str, object],
    metric: str = "linear_cka",
    max_samples: int | None = None,
    seed: int | None = 0,
    dtype=np.float64,
    verbose: bool = False,
    **metric_kwargs,
) -> pd.DataFrame:
    """Build the N x N model-by-model similarity matrix for one metric.

    Parameters
    ----------
    representations : mapping of str to array-like
        ``{model_name: embedding_matrix}``. Every matrix must have the same
        number of rows, in the same patch order; feature dimensions may differ.
    metric : str, default 'linear_cka'
        One of :func:`available_metrics`.
    max_samples : int or None, default None
        Subsample this many patches (shared across all models) before
        computing. Strongly recommended for the O(n^2) metrics — see
        :data:`QUADRATIC_METRICS`.
    seed : int or None, default 0
        RNG seed for subsampling. Fixing it keeps matrices comparable across
        metrics, which matters because the metrics are later compared to one
        another.
    dtype : numpy dtype, default ``np.float64``
        Working precision.
    verbose : bool, default False
        Print per-pair progress and timing.
    **metric_kwargs
        Forwarded to the metric.

    Returns
    -------
    pandas.DataFrame
        Square DataFrame indexed and columned by model name. The diagonal is
        computed rather than assumed, so it doubles as a sanity check: it
        should be 1.0 (or very close) for every metric here.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> reps = {"a": rng.normal(size=(200, 16)), "b": rng.normal(size=(200, 24))}
    >>> S = compute_similarity_matrix(reps, "linear_cka")
    >>> S.shape
    (2, 2)
    """
    names, mats = _prepare_representations(representations, max_samples, seed, dtype)
    fn = get_metric(metric)
    symmetric = metric in SYMMETRIC_METRICS

    if metric in QUADRATIC_METRICS and mats[0].shape[0] > 10_000:
        warnings.warn(
            f"{metric!r} is O(n^2) and you passed {mats[0].shape[0]} samples; "
            "consider max_samples=5000.",
            RuntimeWarning,
            stacklevel=2,
        )

    n_models = len(names)
    S = np.full((n_models, n_models), np.nan)

    for i in range(n_models):
        for j in range(n_models):
            if symmetric and j < i:
                S[i, j] = S[j, i]
                continue
            t0 = time.perf_counter()
            S[i, j] = float(fn(mats[i], mats[j], **metric_kwargs))
            if verbose:
                dt = time.perf_counter() - t0
                print(
                    f"[{metric}] {names[i]:>12s} vs {names[j]:<12s} "
                    f"= {S[i, j]:.4f}  ({dt:.2f}s)"
                )

    return pd.DataFrame(S, index=names, columns=names)


def compute_all_similarity_matrices(
    representations: Mapping[str, object],
    metrics: Sequence[str] | None = None,
    max_samples: int | None = None,
    seed: int | None = 0,
    verbose: bool = False,
    metric_kwargs: Mapping[str, Mapping] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build one similarity matrix per metric over the same patch subsample.

    Because the subsample is drawn once with a fixed seed and reused, the
    resulting matrices are directly comparable — any disagreement between
    metrics reflects the metrics, not different data.

    Parameters
    ----------
    representations : mapping of str to array-like
        ``{model_name: embedding_matrix}``, row-paired across models.
    metrics : sequence of str or None, default None
        Metrics to run. ``None`` runs the seven Phase I metrics (i.e. every
        registered metric except the ``mean_cca`` baseline).
    max_samples : int or None, default None
        Shared patch subsample size.
    seed : int or None, default 0
        Subsampling seed.
    verbose : bool, default False
        Print progress.
    metric_kwargs : mapping of str to mapping, optional
        Per-metric keyword overrides, e.g.
        ``{"kernel_cka": {"threshold": 0.4}, "svcca": {"var_threshold": 0.95}}``.

    Returns
    -------
    dict of str to pandas.DataFrame
        ``{metric_name: similarity_matrix}``.
    """
    if metrics is None:
        metrics = [
            "linear_cka",
            "kernel_cka",
            "svcca",
            "pwcca",
            "procrustes",
            "cosine_rsa",
            "distance_correlation",
        ]
    metric_kwargs = metric_kwargs or {}

    results: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        if verbose:
            print(f"=== {metric} ===")
        results[metric] = compute_similarity_matrix(
            representations,
            metric=metric,
            max_samples=max_samples,
            seed=seed,
            verbose=verbose,
            **dict(metric_kwargs.get(metric, {})),
        )
    return results


def similarity_to_distance(
    S: pd.DataFrame | np.ndarray,
    mode: str = "one_minus",
) -> pd.DataFrame | np.ndarray:
    """Convert a similarity matrix into a distance matrix.

    Clustering, MDS and UMAP all consume distances. This enforces the
    properties those routines assume: symmetry, a zero diagonal, and
    non-negativity.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity matrix.
    mode : {'one_minus', 'angular', 'sqrt_one_minus'}, default 'one_minus'
        * ``'one_minus'`` — ``1 - S``. Simple and monotone; adequate for
          clustering and ordination, but not a metric.
        * ``'angular'`` — ``arccos(S) / pi``, a true metric for
          correlation-like similarities in ``[-1, 1]``.
        * ``'sqrt_one_minus'`` — ``sqrt(2 - 2S)``, matching the normalised
          Procrustes distance. Use this when the similarity came from
          ``procrustes`` so that the geometry in the MDS plot is faithful.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Distance matrix of the same type and labels as the input.
    """
    is_df = isinstance(S, pd.DataFrame)
    arr = np.asarray(S.values if is_df else S, dtype=np.float64)

    if mode == "one_minus":
        D = 1.0 - arr
    elif mode == "angular":
        D = np.arccos(np.clip(arr, -1.0, 1.0)) / np.pi
    elif mode == "sqrt_one_minus":
        D = np.sqrt(np.maximum(2.0 - 2.0 * arr, 0.0))
    else:
        raise ValueError(
            f"unknown mode {mode!r}; expected 'one_minus', 'angular' or "
            "'sqrt_one_minus'"
        )

    D = (D + D.T) / 2.0  # enforce exact symmetry for scipy
    np.fill_diagonal(D, 0.0)
    np.maximum(D, 0.0, out=D)

    if is_df:
        return pd.DataFrame(D, index=S.index, columns=S.columns)
    return D


def stack_similarity_matrices(
    matrices: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Flatten several similarity matrices into one long-format table.

    Handy for cross-metric comparison: correlating the columns of the result
    answers "do these metrics rank model pairs the same way", which is a Phase I
    finding in its own right.

    Parameters
    ----------
    matrices : mapping of str to pandas.DataFrame
        ``{metric_name: similarity_matrix}``, all over the same models.

    Returns
    -------
    pandas.DataFrame
        One row per unordered model pair, with columns ``model_a``,
        ``model_b`` and one column per metric.
    """
    if not matrices:
        raise ValueError("no matrices given")

    names = list(next(iter(matrices.values())).index)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    rows = []
    for a, b in pairs:
        row = {"model_a": a, "model_b": b}
        for metric, M in matrices.items():
            row[metric] = float(M.loc[a, b])
        rows.append(row)
    return pd.DataFrame(rows)
