"""Cosine-based representational similarity analysis (RSA).

The classical RSA recipe, and the most directly interpretable metric in the
suite: build a representational similarity matrix (RSM) *within* each model —
patch i vs patch j under that model's own geometry — then ask how well the two
RSMs agree. Because the RSM only records relative angles between patches, the
comparison never needs the two models to share a feature dimension, and it is
automatically invariant to rotation and isotropic scaling.

Cosine (rather than Euclidean) RSMs are used throughout because retrieval in
computational pathology is almost always cosine-based, so this measures
agreement in the geometry that downstream use actually depends on.

This module also provides :func:`mean_cosine_similarity`, which is *not* an
RSA measure: it compares two representations row by row and therefore requires
them to already live in the same space. That makes it the natural scoring
function for Phases V and VI, after alignment.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .preprocessing import _warn_quadratic_memory, as_matrix, prepare_pair

__all__ = [
    "l2_normalize",
    "cosine_rsm",
    "upper_triangle",
    "rsa_similarity",
    "cosine_rsa_similarity",
    "mean_cosine_similarity",
]


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit L2 norm.

    Parameters
    ----------
    X : numpy.ndarray
        Matrix of shape ``(n_samples, n_features)``.
    eps : float, default 1e-12
        Floor on the row norm, guarding all-zero rows.

    Returns
    -------
    numpy.ndarray
        Row-normalised matrix of the same shape.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def cosine_rsm(X, center: bool = True) -> np.ndarray:
    """Build the cosine representational similarity matrix of one model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A single model's representation of the patch set.
    center : bool, default True
        Column-center before normalising. With centering the entries are
        Pearson correlations between patches; without, they are raw cosines.
        Centering is the default so that a shared feature offset (common in
        transformer embeddings, where a few dimensions carry a large constant)
        cannot inflate every similarity towards 1.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix of shape ``(n_samples, n_samples)`` with entries in
        ``[-1, 1]`` and a unit diagonal.

    Notes
    -----
    Memory is O(n^2). Subsample to a few thousand patches per comparison.
    """
    X = as_matrix(X)
    _warn_quadratic_memory(X.shape[0], "cosine_rsm")
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    Xn = l2_normalize(X)
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def upper_triangle(M: np.ndarray, k: int = 1) -> np.ndarray:
    """Extract the strict upper triangle of a square matrix as a flat vector.

    The diagonal is excluded because every RSM has a unit diagonal by
    construction; including it would add n identical values to both sides of
    the correlation and inflate the agreement.

    Parameters
    ----------
    M : numpy.ndarray
        Square matrix of shape ``(n, n)``.
    k : int, default 1
        Diagonal offset, passed to :func:`numpy.triu_indices`.

    Returns
    -------
    numpy.ndarray
        Flat vector of length ``n(n-1)/2``.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {M.shape}")
    iu = np.triu_indices(M.shape[0], k=k)
    return M[iu]


def rsa_similarity(
    rsm_x: np.ndarray,
    rsm_y: np.ndarray,
    correlation: str = "spearman",
) -> float:
    """Correlate two precomputed representational similarity matrices.

    Use this when the RSMs are already built (e.g. cached across many pairwise
    comparisons, or produced with a non-cosine kernel).

    Parameters
    ----------
    rsm_x, rsm_y : numpy.ndarray
        Square RSMs of shape ``(n, n)`` over the *same* patches in the same
        order.
    correlation : {'spearman', 'pearson', 'kendall'}, default 'spearman'
        Correlation applied to the flattened upper triangles.

    Returns
    -------
    float
        Correlation coefficient in ``[-1, 1]``.
    """
    if rsm_x.shape != rsm_y.shape:
        raise ValueError(f"RSM shapes differ: {rsm_x.shape} vs {rsm_y.shape}")

    a = upper_triangle(rsm_x)
    b = upper_triangle(rsm_y)

    correlation = correlation.lower()
    if correlation == "pearson":
        r = stats.pearsonr(a, b)[0]
    elif correlation == "spearman":
        r = stats.spearmanr(a, b)[0]
    elif correlation == "kendall":
        r = stats.kendalltau(a, b)[0]
    else:
        raise ValueError(
            f"unknown correlation {correlation!r}; expected 'spearman', "
            "'pearson' or 'kendall'"
        )
    return float(r)


def cosine_rsa_similarity(
    X,
    Y,
    correlation: str = "spearman",
    center: bool = True,
) -> float:
    """Cosine-RSA similarity between two paired representations.

    Builds a cosine RSM for each model and correlates them. This is the
    "cosine similarity" entry of the Phase I metric suite, expressed as a
    proper representation-level comparison rather than a per-pair cosine.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Representation from the first model.
    Y : array-like of shape (n_samples, d2)
        Representation from the second model, rows paired with X.
    correlation : {'spearman', 'pearson', 'kendall'}, default 'spearman'
        Correlation between the two flattened RSMs. Spearman is the default
        because it only assumes the *ordering* of patch similarities is
        comparable across models — the weakest assumption that still answers
        "do these models agree about which patches look alike".
    center : bool, default True
        Column-center each representation before building its RSM.

    Returns
    -------
    float
        Correlation in ``[-1, 1]``; 1.0 for representations identical up to
        rotation and scaling.

    Notes
    -----
    Memory is O(n^2) and Spearman additionally ranks two length-n(n-1)/2
    vectors, so keep n in the low thousands per comparison.
    """
    X, Y = prepare_pair(X, Y, center=center)
    rsm_x = cosine_rsm(X, center=False)
    rsm_y = cosine_rsm(Y, center=False)
    return rsa_similarity(rsm_x, rsm_y, correlation=correlation)


def mean_cosine_similarity(X, Y, center: bool = False) -> float:
    """Mean row-wise cosine similarity between two *aligned* representations.

    Unlike every other metric in this package, this one compares corresponding
    rows directly, so X and Y must share a feature space and dimension. It is
    meaningless for two raw foundation models and is included for the
    post-alignment evaluations in Phases V and VI, where "did patch i land in
    the right place" is exactly the question.

    Parameters
    ----------
    X, Y : array-like of shape (n_samples, n_features)
        Row-paired representations in a *common* space.
    center : bool, default False
        Column-center before computing cosines. Off by default: after
        alignment the absolute position of the embedding is meaningful.

    Returns
    -------
    float
        Mean cosine similarity across rows, in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the feature dimensions differ.
    """
    X, Y = prepare_pair(X, Y, center=center)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "mean_cosine_similarity compares rows directly and needs a shared "
            f"feature space; got {X.shape[1]} vs {Y.shape[1]} dimensions. Use "
            "cosine_rsa_similarity for unaligned models."
        )
    Xn = l2_normalize(X)
    Yn = l2_normalize(Y)
    return float(np.mean(np.sum(Xn * Yn, axis=1)))
