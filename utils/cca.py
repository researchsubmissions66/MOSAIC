"""Canonical-correlation-based representation similarity: SVCCA and PWCCA.

CCA finds pairs of directions, one in each representation, that are maximally
correlated across the paired samples. The resulting canonical correlations are
invariant to *any* invertible linear transform of either feature space — a
stronger invariance than CKA's. That strength is also the weakness: raw CCA is
dominated by low-variance, noise-like directions, so both refinements below
exist to suppress them.

* **SVCCA** (Raghu et al., NeurIPS 2018) first truncates each representation to
  the top principal subspace, then runs CCA on what survives.
* **PWCCA** (Morcos et al., NeurIPS 2018) runs CCA on the full space but
  weights each canonical correlation by how much of the original
  representation that canonical direction actually accounts for.

Both return a scalar in ``[0, 1]``.
"""

from __future__ import annotations

import warnings

import numpy as np

from .preprocessing import prepare_pair

__all__ = [
    "cca_decomposition",
    "cca_correlations",
    "mean_cca_correlation",
    "svcca",
    "pwcca",
]


def _orthonormal_basis(
    X: np.ndarray, rank_tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a numerically safe orthonormal basis for the column space of X.

    Implemented via a thin SVD with relative-tolerance rank truncation, which
    is more robust than QR for the near-rank-deficient embeddings produced by
    foundation models (dead units, duplicated features).

    Parameters
    ----------
    X : numpy.ndarray
        Column-centered matrix of shape ``(n, d)``.
    rank_tol : float, default 1e-10
        Singular values below ``rank_tol * s_max`` are treated as zero.

    Returns
    -------
    U : numpy.ndarray of shape (n, r)
        Orthonormal basis of the column space, r = numerical rank.
    s : numpy.ndarray of shape (r,)
        Retained singular values.
    Vt : numpy.ndarray of shape (r, d)
        Right singular vectors.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        raise ValueError("representation has zero variance after centering")
    keep = s > rank_tol * s[0]
    return U[:, keep], s[keep], Vt[keep]


def cca_decomposition(
    X,
    Y,
    rank_tol: float = 1e-10,
    center: bool = True,
) -> dict:
    """Run canonical correlation analysis on a paired representation pair.

    Uses the orthogonalisation route: both matrices are replaced by orthonormal
    bases of their column spaces, and the canonical correlations are the
    singular values of the cross-product of those bases. This avoids explicitly
    inverting the (typically ill-conditioned) covariance matrices.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Representation from the first model.
    Y : array-like of shape (n_samples, d2)
        Representation from the second model, rows paired with X.
    rank_tol : float, default 1e-10
        Relative singular-value cutoff used for rank truncation.
    center : bool, default True
        Column-center both representations first.

    Returns
    -------
    dict
        With keys:

        ``correlations`` : ndarray of shape (k,)
            Canonical correlations, sorted descending, clipped to ``[0, 1]``.
            ``k = min(rank(X), rank(Y))``.
        ``x_variates`` : ndarray of shape (n, k)
            Canonical variates of X (orthonormal columns).
        ``y_variates`` : ndarray of shape (n, k)
            Canonical variates of Y (orthonormal columns).
        ``x_centered``, ``y_centered`` : ndarray
            The preprocessed inputs, kept so callers (e.g. PWCCA) do not have
            to recompute them.
        ``ranks`` : tuple of int
            Numerical ranks of X and Y.
    """
    X, Y = prepare_pair(X, Y, center=center)

    Qx, _, _ = _orthonormal_basis(X, rank_tol)
    Qy, _, _ = _orthonormal_basis(Y, rank_tol)

    U, s, Vt = np.linalg.svd(Qx.T @ Qy, full_matrices=False)
    rho = np.clip(s, 0.0, 1.0)

    return {
        "correlations": rho,
        "x_variates": Qx @ U,
        "y_variates": Qy @ Vt.T,
        "x_centered": X,
        "y_centered": Y,
        "ranks": (Qx.shape[1], Qy.shape[1]),
    }


def cca_correlations(X, Y, rank_tol: float = 1e-10) -> np.ndarray:
    """Return just the canonical correlations between two representations.

    Parameters
    ----------
    X, Y : array-like
        Row-paired representation matrices.
    rank_tol : float, default 1e-10
        Relative singular-value cutoff for rank truncation.

    Returns
    -------
    numpy.ndarray
        Canonical correlations sorted descending, shape ``(k,)``.
    """
    return cca_decomposition(X, Y, rank_tol=rank_tol)["correlations"]


def mean_cca_correlation(X, Y, rank_tol: float = 1e-10) -> float:
    """Plain (unweighted, untruncated) mean canonical correlation.

    Included as the *baseline* the SVCCA and PWCCA refinements are measured
    against. It is known to saturate near 1.0 for high-dimensional
    representations, so do not read it as a similarity on its own.

    Parameters
    ----------
    X, Y : array-like
        Row-paired representation matrices.
    rank_tol : float, default 1e-10
        Relative singular-value cutoff for rank truncation.

    Returns
    -------
    float
        Mean canonical correlation in ``[0, 1]``.
    """
    return float(np.mean(cca_correlations(X, Y, rank_tol=rank_tol)))


def _truncate_to_variance(
    X: np.ndarray,
    var_threshold: float | None,
    max_components: int | None,
    rank_tol: float,
) -> np.ndarray:
    """Project a matrix onto its leading principal subspace.

    Parameters
    ----------
    X : numpy.ndarray
        Column-centered matrix of shape ``(n, d)``.
    var_threshold : float or None
        Fraction of total variance to retain, e.g. 0.99. ``None`` disables the
        variance criterion.
    max_components : int or None
        Hard cap on the number of retained components, applied after the
        variance criterion.
    rank_tol : float
        Relative singular-value cutoff.

    Returns
    -------
    numpy.ndarray
        Principal-component scores of shape ``(n, k)``.
    """
    U, s, _ = _orthonormal_basis(X, rank_tol)

    k = s.size
    if var_threshold is not None:
        if not 0 < var_threshold <= 1:
            raise ValueError(
                f"var_threshold must be in (0, 1], got {var_threshold}"
            )
        explained = np.cumsum(s**2) / np.sum(s**2)
        k = int(np.searchsorted(explained, var_threshold) + 1)
        k = min(k, s.size)
    if max_components is not None:
        k = min(k, max_components)
    k = max(k, 1)

    return U[:, :k] * s[:k]


def svcca(
    X,
    Y,
    var_threshold: float | None = 0.99,
    max_components: int | None = None,
    rank_tol: float = 1e-10,
    center: bool = True,
) -> float:
    """Singular Vector CCA similarity.

    Each representation is first truncated to the leading principal subspace
    that explains ``var_threshold`` of its variance, discarding the
    low-variance directions that would otherwise let CCA find spurious
    correlations in noise. CCA is then run on the survivors and the mean
    canonical correlation is returned.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Representation from the first model.
    Y : array-like of shape (n_samples, d2)
        Representation from the second model, rows paired with X.
    var_threshold : float or None, default 0.99
        Fraction of variance retained per representation before CCA. This is
        the single most consequential SVCCA hyperparameter — sweep it
        (0.9 / 0.95 / 0.99) as a Phase I ablation. ``None`` skips variance-based
        truncation.
    max_components : int or None, default None
        Optional hard cap on retained components per representation. Useful for
        putting models of different embedding width (e.g. CONCH's 512 vs
        Virchow's 2560) on an equal footing.
    rank_tol : float, default 1e-10
        Relative singular-value cutoff for rank truncation.
    center : bool, default True
        Column-center both representations first.

    Returns
    -------
    float
        Mean canonical correlation over the truncated subspaces, in ``[0, 1]``.

    Notes
    -----
    SVCCA is symmetric: ``svcca(X, Y) == svcca(Y, X)``.
    """
    X, Y = prepare_pair(X, Y, center=center)
    Xr = _truncate_to_variance(X, var_threshold, max_components, rank_tol)
    Yr = _truncate_to_variance(Y, var_threshold, max_components, rank_tol)
    rho = cca_decomposition(Xr, Yr, rank_tol=rank_tol, center=False)["correlations"]
    return float(np.mean(rho))


def pwcca(
    X,
    Y,
    rank_tol: float = 1e-10,
    symmetric: bool = True,
    center: bool = True,
) -> float:
    """Projection-Weighted CCA similarity.

    Instead of truncating up front like SVCCA, PWCCA keeps every canonical
    correlation but weights it by how much of the *original* representation the
    corresponding canonical direction accounts for:

    .. math:: \\alpha_i = \\sum_j |\\langle h_i, x_j \\rangle|,
              \\qquad
              \\text{PWCCA} = \\sum_i \\tilde{\\alpha}_i \\rho_i

    where :math:`h_i` is the i-th canonical variate of X and :math:`x_j` are
    the original (centered) feature columns of X. Directions that barely
    participate in the representation get near-zero weight.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Representation from the first model. **This is the reference**: the
        weights are computed from X, which is what makes the raw measure
        asymmetric.
    Y : array-like of shape (n_samples, d2)
        Representation from the second model, rows paired with X.
    rank_tol : float, default 1e-10
        Relative singular-value cutoff for rank truncation.
    symmetric : bool, default True
        If True, return the mean of ``pwcca(X, Y)`` and ``pwcca(Y, X)``, giving
        a symmetric measure suitable for the pairwise similarity matrix and for
        hierarchical clustering. Set False to inspect the directional values —
        a large gap between the two directions is itself informative (it means
        one model's representation is closer to a subspace of the other's).
    center : bool, default True
        Column-center both representations first.

    Returns
    -------
    float
        Projection-weighted mean canonical correlation, in ``[0, 1]``.
    """
    X, Y = prepare_pair(X, Y, center=center)

    if symmetric:
        a = pwcca(X, Y, rank_tol=rank_tol, symmetric=False, center=False)
        b = pwcca(Y, X, rank_tol=rank_tol, symmetric=False, center=False)
        return float((a + b) / 2.0)

    fit = cca_decomposition(X, Y, rank_tol=rank_tol, center=False)
    rho = fit["correlations"]
    H = fit["x_variates"]  # (n, k), orthonormal columns

    # alpha_i = sum_j |<h_i, x_j>|  -> row sums of |H^T X|
    weights = np.abs(H.T @ X).sum(axis=1)
    total = weights.sum()
    if total <= 0:
        warnings.warn(
            "PWCCA weights are all zero; falling back to the unweighted mean "
            "canonical correlation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(np.mean(rho))
    weights = weights / total
    return float(np.dot(weights, rho))
