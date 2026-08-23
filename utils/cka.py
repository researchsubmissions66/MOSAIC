"""Centered Kernel Alignment (CKA) for comparing representation spaces.

CKA measures how similarly two representations arrange the *same* set of
samples, ignoring rotations, isotropic rescaling and (with centering) shifts of
the feature space. It is the workhorse metric of Phase I.

Reference
---------
Kornblith, Norouzi, Lee & Hinton (2019), "Similarity of Neural Network
Representations Revisited", ICML. The unbiased HSIC estimator follows
Song et al. (2012) as used by Nguyen, Raghu & Kornblith (2021).

Invariances
-----------
* orthogonal transforms of the feature space  -> invariant
* isotropic scaling                           -> invariant
* translation (given column centering)        -> invariant
* arbitrary invertible linear maps            -> NOT invariant (unlike CCA)

That last point is the reason CKA and SVCCA/PWCCA can disagree, and why the
plan runs both.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import _warn_quadratic_memory, prepare_pair

__all__ = [
    "gram_linear",
    "gram_rbf",
    "center_gram",
    "hsic",
    "cka_from_grams",
    "linear_cka",
    "kernel_cka",
]


def gram_linear(X: np.ndarray) -> np.ndarray:
    """Compute the linear Gram matrix ``X @ X.T``.

    Parameters
    ----------
    X : numpy.ndarray
        Representation matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    numpy.ndarray
        Gram matrix of shape ``(n_samples, n_samples)``.
    """
    return X @ X.T


def gram_rbf(X: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Compute an RBF (Gaussian) Gram matrix with a median-heuristic bandwidth.

    The kernel width is set to ``threshold`` times the median pairwise
    Euclidean distance, which makes the kernel scale-free and removes the need
    to tune sigma per model — important here, since foundation-model embeddings
    have very different norms.

    Parameters
    ----------
    X : numpy.ndarray
        Representation matrix of shape ``(n_samples, n_features)``.
    threshold : float, default 1.0
        Bandwidth as a fraction of the median pairwise distance. Kornblith
        et al. sweep 0.2-0.8; 1.0 is a mild, well-behaved default. Smaller
        values make the kernel more local (closer to a nearest-neighbour
        comparison), larger values push it towards the linear kernel.

    Returns
    -------
    numpy.ndarray
        Gram matrix of shape ``(n_samples, n_samples)``.

    Notes
    -----
    Memory is O(n^2); subsample before calling for n beyond ~20k.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    _warn_quadratic_memory(X.shape[0], "gram_rbf")

    dot = X @ X.T
    sq_norms = np.diag(dot)
    sq_dists = -2.0 * dot + sq_norms[:, None] + sq_norms[None, :]
    np.maximum(sq_dists, 0.0, out=sq_dists)  # kill negative round-off

    sq_median = float(np.median(sq_dists))
    if sq_median <= 0:
        raise ValueError(
            "median pairwise distance is zero — the representation is "
            "constant or contains duplicate rows only"
        )
    return np.exp(-sq_dists / (2.0 * threshold**2 * sq_median))


def center_gram(K: np.ndarray, unbiased: bool = False) -> np.ndarray:
    """Center a Gram matrix in feature space.

    Parameters
    ----------
    K : numpy.ndarray
        Symmetric Gram matrix of shape ``(n, n)``.
    unbiased : bool, default False
        If True, apply the centering used by the unbiased HSIC estimator of
        Song et al. (2012), which removes the O(1/n) bias that otherwise makes
        CKA depend on the sample size. Recommended when comparing values across
        analyses with different n (e.g. the organ-wise breakdown in Phase III,
        where cohorts have very different sizes).

    Returns
    -------
    numpy.ndarray
        Centered Gram matrix, shape ``(n, n)``.

    Raises
    ------
    ValueError
        If ``K`` is not square and symmetric.
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Gram matrix must be square, got shape {K.shape}")
    if not np.allclose(K, K.T, rtol=1e-6, atol=1e-6):
        raise ValueError("Gram matrix must be symmetric")

    K = K.copy()
    n = K.shape[0]

    if unbiased:
        if n < 4:
            raise ValueError(f"unbiased HSIC needs at least 4 samples, got {n}")
        np.fill_diagonal(K, 0.0)
        means = K.sum(axis=0, dtype=np.float64) / (n - 2)
        means -= means.sum() / (2 * (n - 1))
        K -= means[:, None]
        K -= means[None, :]
        np.fill_diagonal(K, 0.0)
    else:
        means = K.mean(axis=0, dtype=np.float64)
        means -= means.mean() / 2
        K -= means[:, None]
        K -= means[None, :]
    return K


def hsic(K: np.ndarray, L: np.ndarray, unbiased: bool = False) -> float:
    """Hilbert-Schmidt Independence Criterion between two Gram matrices.

    Parameters
    ----------
    K, L : numpy.ndarray
        Gram matrices of shape ``(n, n)``, **not** yet centered.
    unbiased : bool, default False
        Use the unbiased estimator (see :func:`center_gram`).

    Returns
    -------
    float
        HSIC value. Zero iff the two representations are independent under the
        chosen kernels; larger means more dependent.
    """
    if K.shape != L.shape:
        raise ValueError(f"Gram matrices must match: {K.shape} vs {L.shape}")
    n = K.shape[0]
    Kc = center_gram(K, unbiased=unbiased)
    Lc = center_gram(L, unbiased=unbiased)
    cross = float(np.dot(Kc.ravel(), Lc.ravel()))
    norm = n * (n - 3) if unbiased else (n - 1) ** 2
    return cross / norm


def cka_from_grams(K: np.ndarray, L: np.ndarray, unbiased: bool = False) -> float:
    """Compute CKA directly from two precomputed Gram matrices.

    Use this when you want a kernel that is not covered by
    :func:`linear_cka` / :func:`kernel_cka` (e.g. a cosine or a
    pathology-specific kernel), or when reusing Gram matrices across many
    comparisons.

    Parameters
    ----------
    K, L : numpy.ndarray
        Uncentered Gram matrices of shape ``(n, n)``.
    unbiased : bool, default False
        Use the unbiased HSIC estimator.

    Returns
    -------
    float
        CKA similarity in ``[0, 1]`` (the unbiased estimator can return small
        negative values when the true similarity is near zero).
    """
    num = hsic(K, L, unbiased=unbiased)
    den = np.sqrt(hsic(K, K, unbiased=unbiased) * hsic(L, L, unbiased=unbiased))
    if den <= 0:
        return 0.0
    return float(num / den)


def linear_cka(X, Y, unbiased: bool = False, center: bool = True) -> float:
    """Linear CKA between two paired representations.

    With the biased estimator this is computed in *feature* space as

    .. math:: \\text{CKA} = \\frac{\\|Y^\\top X\\|_F^2}
                                 {\\|X^\\top X\\|_F \\, \\|Y^\\top Y\\|_F}

    which costs O(n d^2) time and O(d^2) memory instead of the O(n^2) of the
    Gram-matrix formulation — the difference between tractable and not for
    100k+ patches. The two formulations are mathematically identical.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Representation from the first model.
    Y : array-like of shape (n_samples, d2)
        Representation from the second model, **rows paired with X**.
    unbiased : bool, default False
        Use the unbiased HSIC estimator. This forces the O(n^2) Gram-matrix
        path, so subsample first.
    center : bool, default True
        Column-center both representations before comparing. Leave on.

    Returns
    -------
    float
        CKA similarity, 1.0 for representations equal up to rotation and
        isotropic scaling.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 32))
    >>> Q, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    >>> round(linear_cka(X, 3.0 * X @ Q), 6)  # rotation + scaling invariant
    1.0
    """
    X, Y = prepare_pair(X, Y, center=center)

    if unbiased:
        return cka_from_grams(gram_linear(X), gram_linear(Y), unbiased=True)

    cross = X.T @ Y
    num = float(np.linalg.norm(cross, "fro") ** 2)
    den = float(
        np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    )
    if den <= 0:
        return 0.0
    return num / den


def kernel_cka(
    X,
    Y,
    threshold: float = 1.0,
    unbiased: bool = False,
    center: bool = True,
) -> float:
    """RBF-kernel CKA between two paired representations.

    The nonlinear counterpart of :func:`linear_cka`. Where linear CKA only sees
    second-order structure, kernel CKA is sensitive to nonlinear manifold
    structure — the relevant question for tissue morphology, where semantically
    similar patches may lie on a curved manifold rather than in a linear
    subspace.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Representation from the first model.
    Y : array-like of shape (n_samples, d2)
        Representation from the second model, rows paired with X.
    threshold : float, default 1.0
        RBF bandwidth as a fraction of the median pairwise distance. Sweep this
        (e.g. 0.2 / 0.4 / 0.8) as one of the Phase I ablations.
    unbiased : bool, default False
        Use the unbiased HSIC estimator.
    center : bool, default True
        Column-center both representations before building the kernels.

    Returns
    -------
    float
        Kernel CKA similarity in ``[0, 1]``.

    Notes
    -----
    Memory is O(n^2) — three n x n float64 matrices are live at peak. Subsample
    to ~5-10k patches per comparison.
    """
    X, Y = prepare_pair(X, Y, center=center)
    _warn_quadratic_memory(X.shape[0], "kernel_cka")
    K = gram_rbf(X, threshold=threshold)
    L = gram_rbf(Y, threshold=threshold)
    return cka_from_grams(K, L, unbiased=unbiased)
