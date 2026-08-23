"""Shared preprocessing helpers for representational similarity analysis.

Every similarity metric in this package operates on *paired* representation
matrices: two matrices ``X`` (n x d1) and ``Y`` (n x d2) whose rows correspond
to the **same** n image patches, encoded by two different foundation models.
The feature dimensions d1 and d2 may differ.

The helpers here handle the boring-but-critical parts that all metrics share:
type coercion (torch -> numpy), pairing checks, column centering, Frobenius
normalisation, and consistent row subsampling.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "as_matrix",
    "check_paired",
    "center_columns",
    "zscore_columns",
    "normalize_frobenius",
    "drop_constant_columns",
    "subsample_indices",
    "prepare_pair",
]


def as_matrix(X, dtype=np.float64, copy: bool = True) -> np.ndarray:
    """Coerce a representation container into a 2-D numpy array.

    Accepts numpy arrays, torch tensors (CPU or GPU), or anything exposing
    ``__array__``. Torch tensors are detached and moved to CPU automatically.

    Parameters
    ----------
    X : array-like or torch.Tensor
        Representation matrix of shape ``(n_samples, n_features)``.
    dtype : numpy dtype, default ``np.float64``
        Target dtype. Float64 is the default because the CCA and Procrustes
        routines involve SVDs of ill-conditioned matrices where float32 loses
        meaningful precision.
    copy : bool, default True
        If False, skip the copy when the input is already a suitable array.
        A copy still happens when one is unavoidable (e.g. converting stored
        float32 features to float64) — this is "avoid copying when possible",
        not NumPy 2's "never copy", which would raise here.

    Returns
    -------
    numpy.ndarray
        C-contiguous array of shape ``(n_samples, n_features)``.

    Raises
    ------
    ValueError
        If the input is not 2-D or contains non-finite values.
    """
    if hasattr(X, "detach"):  # torch.Tensor (avoids a hard torch dependency)
        X = X.detach().cpu().numpy()

    # np.asarray copies only when required; np.array(copy=False) would raise
    # under NumPy 2 whenever a dtype conversion forces one.
    X = np.array(X, dtype=dtype, order="C") if copy else np.asarray(X, dtype=dtype, order="C")

    if X.ndim != 2:
        raise ValueError(
            f"expected a 2-D (n_samples, n_features) matrix, got shape {X.shape}"
        )
    if not np.all(np.isfinite(X)):
        n_bad = int((~np.isfinite(X)).sum())
        raise ValueError(f"representation contains {n_bad} non-finite value(s)")
    return X


def check_paired(X: np.ndarray, Y: np.ndarray) -> None:
    """Validate that two representation matrices are row-paired and usable.

    Parameters
    ----------
    X, Y : numpy.ndarray
        Representation matrices of shape ``(n, d1)`` and ``(n, d2)``.

    Raises
    ------
    ValueError
        If the sample counts differ, or if either matrix has fewer samples
        than features in a way that makes the similarity estimate degenerate.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "representations must be row-paired (same patches, same order): "
            f"got {X.shape[0]} and {Y.shape[0]} samples"
        )
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 samples, got {n}")

    d_max = max(X.shape[1], Y.shape[1])
    if n <= d_max:
        warnings.warn(
            f"n_samples ({n}) <= n_features ({d_max}). Similarity estimates "
            "will be strongly biased upward (CCA-family metrics saturate at "
            "1.0 in this regime). Use at least ~10x more samples than the "
            "largest feature dimension.",
            RuntimeWarning,
            stacklevel=3,
        )


def center_columns(X: np.ndarray, copy: bool = True) -> np.ndarray:
    """Subtract the per-feature (column) mean.

    Column centering is required by every metric here: CKA, CCA, Procrustes and
    distance correlation are all defined on mean-centered representations, and
    skipping it lets a shared offset inflate the similarity.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of shape ``(n_samples, n_features)``.
    copy : bool, default True
        If False, centre in place.

    Returns
    -------
    numpy.ndarray
        Column-centered matrix of the same shape.
    """
    if copy:
        X = X.copy()
    X -= X.mean(axis=0, keepdims=True)
    return X


def zscore_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Center and scale each feature to unit variance.

    Useful when feature dimensions live on wildly different scales (e.g. before
    joint PCA in Phase V). Note that most Phase I metrics are already invariant
    to isotropic scaling, so this is *not* applied by default.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of shape ``(n_samples, n_features)``.
    eps : float, default 1e-12
        Floor on the standard deviation, guarding constant columns.

    Returns
    -------
    numpy.ndarray
        Standardised matrix of the same shape.
    """
    X = center_columns(X)
    X /= np.maximum(X.std(axis=0, keepdims=True), eps)
    return X


def normalize_frobenius(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rescale a matrix to unit Frobenius norm.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of shape ``(n_samples, n_features)``.
    eps : float, default 1e-12
        Floor on the norm, guarding all-zero matrices.

    Returns
    -------
    numpy.ndarray
        Matrix scaled so that ``||X||_F == 1``.
    """
    return X / max(float(np.linalg.norm(X, "fro")), eps)


def drop_constant_columns(X: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Remove features with (near-)zero variance.

    Dead units are common in foundation-model embeddings and make the CCA
    whitening step rank-deficient. Dropping them is numerically equivalent to
    the rank truncation the CCA code performs anyway, but is cheaper.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of shape ``(n_samples, n_features)``.
    tol : float, default 1e-10
        Standard-deviation threshold below which a column is dropped.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_samples, n_kept_features)``.
    """
    keep = X.std(axis=0) > tol
    if not keep.any():
        raise ValueError("all feature columns are constant")
    return X[:, keep]


def subsample_indices(
    n_samples: int,
    max_samples: int | None,
    seed: int | None = 0,
) -> np.ndarray | None:
    """Draw a fixed random subset of row indices.

    Several metrics are O(n^2) in memory (kernel CKA, distance correlation), so
    large patch collections must be subsampled. Crucially, the **same** indices
    must be applied to every model's representation, otherwise the rows stop
    being paired. Generate the indices once and reuse them.

    Parameters
    ----------
    n_samples : int
        Total number of available samples.
    max_samples : int or None
        Maximum number of samples to keep. ``None`` (or a value >= n_samples)
        returns ``None``, meaning "use everything".
    seed : int or None, default 0
        Seed for the RNG, for reproducibility.

    Returns
    -------
    numpy.ndarray or None
        Sorted array of selected row indices, or ``None`` if no subsampling is
        needed. Indices are sorted so that any downstream ordering (e.g. organ
        grouping) is preserved.
    """
    if max_samples is None or n_samples <= max_samples:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_samples, size=max_samples, replace=False)
    idx.sort()
    return idx


def prepare_pair(
    X,
    Y,
    center: bool = True,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce, validate and optionally center a pair of representations.

    This is the standard entry point used at the top of every metric.

    Parameters
    ----------
    X, Y : array-like or torch.Tensor
        Row-paired representation matrices, shapes ``(n, d1)`` and ``(n, d2)``.
    center : bool, default True
        Whether to column-center both matrices.
    dtype : numpy dtype, default ``np.float64``
        Working precision.

    Returns
    -------
    tuple of numpy.ndarray
        The prepared ``(X, Y)`` pair.
    """
    X = as_matrix(X, dtype=dtype)
    Y = as_matrix(Y, dtype=dtype)
    check_paired(X, Y)
    if center:
        X = center_columns(X, copy=False)
        Y = center_columns(Y, copy=False)
    return X, Y


def _warn_quadratic_memory(n: int, name: str, threshold: int = 20_000) -> None:
    """Warn when an O(n^2) metric is about to allocate a very large matrix."""
    if n > threshold:
        gb = (n * n * 8) / 1024**3
        warnings.warn(
            f"{name} builds {n}x{n} matrices (~{gb:.1f} GB each). Consider "
            "passing max_samples to subsample first.",
            RuntimeWarning,
            stacklevel=3,
        )
