"""Orthogonal Procrustes analysis for representation comparison and alignment.

Procrustes asks the most literal version of the central hypothesis: *if two
models really are the same space in different coordinates, there should be a
single rotation that maps one onto the other.* Unlike CKA and CCA, which score
similarity indirectly, Procrustes hands back the actual transform — so this
module does double duty as the Phase I metric and as the simplest baseline
aligner for Phases V and VI.

Reference
---------
Ding, Denain & Steinhardt (2021), "Grounding Representational Similarity with
Statistical Testing", NeurIPS — for the normalised Procrustes distance and its
stability properties relative to CKA/CCA.

Dimension mismatch
------------------
Foundation models have different embedding widths. Both matrices are
zero-padded to ``max(d1, d2)`` columns before alignment, which leaves Frobenius
norms and inner products untouched while permitting a square orthogonal
transform.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import as_matrix, prepare_pair

__all__ = [
    "pad_to_common_dim",
    "orthogonal_procrustes_similarity",
    "orthogonal_procrustes_distance",
    "fit_procrustes_transform",
    "apply_procrustes_transform",
]


def pad_to_common_dim(
    X: np.ndarray, Y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the narrower matrix so both have the same feature dimension.

    Padding with zeros is norm-preserving: ``||X_pad||_F == ||X||_F`` and
    ``X_pad.T @ Y_pad`` contains ``X.T @ Y`` as a block. This is what allows a
    square orthogonal transform to be used across models of different width.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d1)
    Y : numpy.ndarray of shape (n, d2)

    Returns
    -------
    tuple of numpy.ndarray
        Both matrices, each of shape ``(n, max(d1, d2))``.
    """
    d = max(X.shape[1], Y.shape[1])
    if X.shape[1] < d:
        X = np.pad(X, ((0, 0), (0, d - X.shape[1])))
    if Y.shape[1] < d:
        Y = np.pad(Y, ((0, 0), (0, d - Y.shape[1])))
    return X, Y


def orthogonal_procrustes_similarity(X, Y, center: bool = True) -> float:
    """Scale-invariant orthogonal Procrustes similarity.

    Defined as the nuclear norm of the cross-covariance, normalised by the
    Frobenius norms:

    .. math:: s(X, Y) = \\frac{\\|X^\\top Y\\|_*}{\\|X\\|_F \\, \\|Y\\|_F}

    This is exactly the quantity that the optimal rotation achieves, rescaled
    into ``[0, 1]``: 1.0 iff Y equals X up to rotation and isotropic scaling,
    0.0 iff the two column spaces are mutually orthogonal.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Representation from the first model.
    Y : array-like of shape (n_samples, d2)
        Representation from the second model, rows paired with X.
    center : bool, default True
        Column-center both representations first.

    Returns
    -------
    float
        Procrustes similarity in ``[0, 1]``. Symmetric in its arguments.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 16))
    >>> Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    >>> round(orthogonal_procrustes_similarity(X, X @ Q), 6)
    1.0
    """
    X, Y = prepare_pair(X, Y, center=center)
    nx = float(np.linalg.norm(X, "fro"))
    ny = float(np.linalg.norm(Y, "fro"))
    if nx <= 0 or ny <= 0:
        return 0.0
    nuclear = float(np.linalg.svd(X.T @ Y, compute_uv=False).sum())
    return float(np.clip(nuclear / (nx * ny), 0.0, 1.0))


def orthogonal_procrustes_distance(X, Y, center: bool = True) -> float:
    """Normalised orthogonal Procrustes distance.

    The residual of the best rotation after both representations are scaled to
    unit Frobenius norm:

    .. math:: d(X, Y) = \\sqrt{2 - 2 s(X, Y)}

    with :math:`s` as in :func:`orthogonal_procrustes_similarity`. Ranges over
    ``[0, sqrt(2)]``. This is a proper metric on representations modulo
    rotation and scale, which makes it the right input to hierarchical
    clustering and MDS when you want the geometry to be meaningful rather than
    just monotone.

    Parameters
    ----------
    X, Y : array-like
        Row-paired representation matrices.
    center : bool, default True
        Column-center both representations first.

    Returns
    -------
    float
        Procrustes distance in ``[0, sqrt(2)]``.
    """
    s = orthogonal_procrustes_similarity(X, Y, center=center)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * s)))


def fit_procrustes_transform(
    X,
    Y,
    scaling: bool = True,
    center: bool = True,
) -> dict:
    """Fit the orthogonal map that best takes Y onto X.

    Solves :math:`\\min_{Q, c} \\|X - c\\, Y Q\\|_F` subject to
    :math:`Q^\\top Q = I`, via the SVD of the cross-covariance.

    Parameters
    ----------
    X : array-like of shape (n_samples, d1)
        Target representation.
    Y : array-like of shape (n_samples, d2)
        Source representation to be rotated onto X, rows paired with X.
    scaling : bool, default True
        Also fit the optimal isotropic scale factor. Turn off if you want a
        strictly rigid map (e.g. when comparing L2-normalised embeddings).
    center : bool, default True
        Column-center both representations, and record the means so that
        :func:`apply_procrustes_transform` can reproduce the mapping on
        held-out data.

    Returns
    -------
    dict
        With keys:

        ``rotation`` : ndarray of shape (d, d)
            Orthogonal matrix Q, where ``d = max(d1, d2)``.
        ``scale`` : float
            Optimal scale factor c (1.0 if ``scaling=False``).
        ``x_mean``, ``y_mean`` : ndarray
            Column means removed from X and Y (zeros if ``center=False``).
        ``dim`` : int
            The common padded dimension d.
        ``x_dim``, ``y_dim`` : int
            Original feature dimensions, needed to unpad.
        ``disparity`` : float
            Normalised residual ``||X - c Y Q||_F^2 / ||X||_F^2`` — the
            fraction of X's variance the rotation fails to explain.

    Notes
    -----
    Fit on a training split and evaluate the disparity on a held-out split;
    a rotation fitted on n < d samples is trivially perfect and meaningless.
    """
    Xc, Yc = prepare_pair(X, Y, center=False)
    x_dim, y_dim = Xc.shape[1], Yc.shape[1]

    if center:
        x_mean = Xc.mean(axis=0)
        y_mean = Yc.mean(axis=0)
        Xc = Xc - x_mean
        Yc = Yc - y_mean
    else:
        x_mean = np.zeros(Xc.shape[1])
        y_mean = np.zeros(Yc.shape[1])

    Xp, Yp = pad_to_common_dim(Xc, Yc)

    U, s, Vt = np.linalg.svd(Yp.T @ Xp, full_matrices=False)
    Q = U @ Vt  # (d, d), maps Y -> X

    if scaling:
        denom = float(np.linalg.norm(Yp, "fro") ** 2)
        scale = float(s.sum() / denom) if denom > 0 else 1.0
    else:
        scale = 1.0

    residual = float(np.linalg.norm(Xp - scale * (Yp @ Q), "fro") ** 2)
    x_norm_sq = float(np.linalg.norm(Xp, "fro") ** 2)
    disparity = residual / x_norm_sq if x_norm_sq > 0 else np.nan

    return {
        "rotation": Q,
        "scale": scale,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "dim": Xp.shape[1],
        "x_dim": x_dim,
        "y_dim": y_dim,
        "disparity": disparity,
    }


def apply_procrustes_transform(Y, transform: dict, unpad: bool = True) -> np.ndarray:
    """Map a source representation into the target space of a fitted transform.

    Parameters
    ----------
    Y : array-like of shape (n_samples, d2)
        Representation in the source model's space (the same model the
        transform was fitted from).
    transform : dict
        Output of :func:`fit_procrustes_transform`.
    unpad : bool, default True
        Trim the result back to the target model's original dimension. Set
        False to keep the padded common dimension.

    Returns
    -------
    numpy.ndarray
        Y mapped into X's coordinate system, shape ``(n_samples, d1)``
        (or ``(n_samples, d)`` if ``unpad=False``).

    Raises
    ------
    ValueError
        If Y's feature dimension does not match the fitted source dimension.
    """
    Yc = as_matrix(Y)
    if Yc.shape[1] != transform["y_dim"]:
        raise ValueError(
            f"expected {transform['y_dim']} source features, got {Yc.shape[1]}"
        )

    Yc = Yc - transform["y_mean"]
    pad = transform["dim"] - Yc.shape[1]
    if pad > 0:
        Yc = np.pad(Yc, ((0, 0), (0, pad)))

    out = transform["scale"] * (Yc @ transform["rotation"])

    x_mean = transform["x_mean"]
    if x_mean.size < out.shape[1]:
        x_mean = np.pad(x_mean, (0, out.shape[1] - x_mean.size))
    out = out + x_mean

    if unpad:
        out = out[:, : transform["x_dim"]]
    return out
