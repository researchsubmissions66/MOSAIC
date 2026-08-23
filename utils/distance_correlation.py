"""Distance correlation between representation spaces.

Distance correlation (Szekely, Rizzo & Bakirov, 2007) is zero **iff** the two
representations are statistically independent — including nonlinear and
higher-order dependence that CKA's linear kernel and CCA's linear projections
both miss. In this suite it is the strictest test of the central hypothesis: if
two pathology foundation models genuinely encode the same morphology, dCor
should be high; if it is near zero, no alignment of any kind will succeed.

Unlike CKA it is not invariant to anisotropic rescaling of the feature space,
which is a feature rather than a bug — it means dCor responds to distortions of
the metric structure that CKA is blind to.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .preprocessing import _warn_quadratic_memory, prepare_pair

__all__ = [
    "pairwise_distance_matrix",
    "double_center",
    "u_center",
    "distance_covariance",
    "distance_correlation",
]


def pairwise_distance_matrix(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute the full pairwise distance matrix of one representation.

    Parameters
    ----------
    X : numpy.ndarray
        Representation matrix of shape ``(n_samples, n_features)``.
    metric : str, default 'euclidean'
        Any metric accepted by :func:`scipy.spatial.distance.pdist`.
        ``'cosine'`` is a reasonable alternative for L2-normalised embeddings.

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix of shape ``(n, n)`` with a zero diagonal.
    """
    _warn_quadratic_memory(X.shape[0], "pairwise_distance_matrix")
    return squareform(pdist(X, metric=metric))


def double_center(D: np.ndarray) -> np.ndarray:
    """Apply the classical (biased) double centering to a distance matrix.

    Subtracts row means and column means and adds back the grand mean, so that
    all rows and columns sum to zero.

    Parameters
    ----------
    D : numpy.ndarray
        Distance matrix of shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Double-centered matrix of shape ``(n, n)``.
    """
    row = D.mean(axis=0, keepdims=True)
    col = D.mean(axis=1, keepdims=True)
    return D - row - col + D.mean()


def u_center(D: np.ndarray) -> np.ndarray:
    """Apply U-centering, yielding the unbiased distance-covariance estimator.

    The biased estimator has an O(1/n) upward bias that becomes severe for
    high-dimensional data — precisely the regime of foundation-model
    embeddings, where the biased dCor between two *independent* random matrices
    can still look substantial. Prefer this whenever the absolute value
    matters, or when comparing across sample sizes.

    Parameters
    ----------
    D : numpy.ndarray
        Distance matrix of shape ``(n, n)``, n >= 4.

    Returns
    -------
    numpy.ndarray
        U-centered matrix of shape ``(n, n)`` with a zero diagonal.

    References
    ----------
    Szekely & Rizzo (2014), "Partial distance correlation with methods for
    dissimilarities", Annals of Statistics.
    """
    n = D.shape[0]
    if n < 4:
        raise ValueError(f"U-centering needs at least 4 samples, got {n}")

    row = D.sum(axis=1, keepdims=True) / (n - 2)
    col = D.sum(axis=0, keepdims=True) / (n - 2)
    grand = D.sum() / ((n - 1) * (n - 2))

    U = D - row - col + grand
    np.fill_diagonal(U, 0.0)
    return U


def distance_covariance(
    A: np.ndarray, B: np.ndarray, unbiased: bool = True
) -> float:
    """Squared distance covariance from two *already centered* matrices.

    Parameters
    ----------
    A, B : numpy.ndarray
        Centered distance matrices of shape ``(n, n)``, from
        :func:`u_center` (unbiased) or :func:`double_center` (biased).
    unbiased : bool, default True
        Must match the centering used to produce A and B — it selects the
        normalisation constant.

    Returns
    -------
    float
        Squared distance covariance. May be slightly negative under the
        unbiased estimator when the true value is zero.
    """
    n = A.shape[0]
    inner = float(np.dot(A.ravel(), B.ravel()))
    return inner / (n * (n - 3)) if unbiased else inner / (n * n)


def distance_correlation(
    X,
    Y,
    unbiased: bool = True,
    metric: str = "euclidean",
    center: bool = True,
) -> float:
    """Distance correlation between two paired representations.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Representation from the first model.
    Y : array-like of shape (n_samples, d2)
        Representation from the second model, rows paired with X.
    unbiased : bool, default True
        Use the U-centered (unbiased) estimator. Recommended, and unbiased is
        the default here — the biased estimator is badly inflated for
        high-dimensional embeddings. Note the unbiased version can return
        small negative values, which are then clipped to 0.
    metric : str, default 'euclidean'
        Distance used to build each representation's distance matrix.
    center : bool, default True
        Column-center each representation first. Euclidean distances are
        already translation invariant, so this only matters for other metrics.

    Returns
    -------
    float
        Distance correlation in ``[0, 1]``. 0 indicates statistical
        independence; 1 indicates the two representations are related by a
        similarity transform (orthogonal map plus scaling).

    Notes
    -----
    Memory is O(n^2) with four n x n float64 matrices live at peak; time is
    O(n^2 d). Subsample to a few thousand patches per comparison.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 8))
    >>> round(distance_correlation(X, X), 6)
    1.0
    """
    X, Y = prepare_pair(X, Y, center=center)

    Dx = pairwise_distance_matrix(X, metric=metric)
    Dy = pairwise_distance_matrix(Y, metric=metric)

    centerer = u_center if unbiased else double_center
    A = centerer(Dx)
    B = centerer(Dy)

    dcov_xy = distance_covariance(A, B, unbiased=unbiased)
    dvar_x = distance_covariance(A, A, unbiased=unbiased)
    dvar_y = distance_covariance(B, B, unbiased=unbiased)

    denom = np.sqrt(max(dvar_x, 0.0) * max(dvar_y, 0.0))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.sqrt(max(dcov_xy, 0.0) / denom), 0.0, 1.0))
