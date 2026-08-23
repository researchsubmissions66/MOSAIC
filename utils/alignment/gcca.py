"""Generalized Canonical Correlation Analysis (MAX-VAR formulation).

GCCA is the most direct formalisation of the central hypothesis. It posits a
single latent matrix :math:`G` — the shared morphological representation — and
asks each model to be a linear view of it:

.. math:: \\min_{G, U_1 \\dots U_M} \\sum_m \\|G - X_m U_m\\|_F^2
          \\quad \\text{s.t.} \\quad G^\\top G = I

The constraint is what makes the problem well-posed (otherwise :math:`G = 0`
wins) and has a useful side effect: the shared space comes out orthonormal, so
distances in it are directly comparable across models.

Reference
---------
Carroll (1968); Kettenring (1971), "Canonical analysis of several sets of
variables", Biometrika — the MAX-VAR variant.

Solution
--------
With :math:`Q_m` an orthonormal basis of view m's column space, the objective
is maximised by the top-k eigenvectors of :math:`\\sum_m Q_m Q_m^\\top`. This
implementation never forms that n x n matrix: it eigendecomposes the much
smaller Gram matrix of the stacked bases instead, block by block, so memory
stays O(n * max_rank) rather than O(n^2).
"""

from __future__ import annotations

import numpy as np

from .base import BaseAligner

__all__ = ["GCCAAligner"]


class GCCAAligner(BaseAligner):
    """Generalized CCA (MAX-VAR) shared latent space.

    Parameters
    ----------
    latent_dim : int, default 64
        Dimensionality of the shared space.
    reg : float, default 1e-4
        Regularisation on each view's whitening, relative to that view's
        largest squared singular value. GCCA whitens every view, which
        amplifies its low-variance directions; without regularisation a single
        noisy model can dominate the shared space. Raise this (1e-2, 1e-1) if
        the shared space looks like it is fitting noise.
    rank_tol : float, default 1e-8
        Relative singular-value cutoff for each view's numerical rank.
    pca_dim : int or None, default None
        Per-view PCA pre-reduction.
    scaling : {'rms', 'std', 'none'}, default 'rms'
        Per-view scaling.
    decoder_reg : float, default 1e-6
        Ridge penalty for the decoders.
    random_state : int, default 0
        Seed.

    Attributes
    ----------
    G_ : numpy.ndarray
        The shared latent representation of the *training* patches, shape
        ``(n_train, latent_dim)``, with orthonormal columns.
    projections_ : dict of str to numpy.ndarray
        Per-model encoders ``U_m``, shape ``(n_features_out, latent_dim)``.
    eigenvalues_ : numpy.ndarray
        Top-k eigenvalues of the summed projection operator. Each lies in
        ``[1, M]``; a value near M means all M models agree on that latent
        direction, near 1 means only one model expresses it. **This is the
        headline diagnostic of Phase V** — the eigenvalue spectrum is a direct
        readout of how many dimensions of morphology the models share.
    view_agreement_ : numpy.ndarray
        ``eigenvalues_ / n_views``, in ``[0, 1]``, for convenient reporting.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> latent = rng.normal(size=(500, 8))
    >>> views = {f"m{i}": latent @ rng.normal(size=(8, 32 + 8 * i)) for i in range(3)}
    >>> aligner = GCCAAligner(latent_dim=8).fit(views)
    >>> Z = aligner.transform(views)
    >>> Z["m0"].shape
    (500, 8)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        reg: float = 1e-4,
        rank_tol: float = 1e-8,
        pca_dim: int | None = None,
        scaling: str = "rms",
        decoder_reg: float = 1e-6,
        random_state: int = 0,
    ):
        super().__init__(
            latent_dim=latent_dim,
            pca_dim=pca_dim,
            scaling=scaling,
            decoder_reg=decoder_reg,
            random_state=random_state,
        )
        self.reg = float(reg)
        self.rank_tol = float(rank_tol)

    def _fit(self, prepared: dict[str, np.ndarray]) -> None:
        names = list(prepared)
        k = self.latent_dim

        # Per-view thin SVD: X_m = A_m diag(s_m) B_m^T, with A_m an orthonormal
        # basis of the column space.
        bases: list[np.ndarray] = []
        svds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name in names:
            Xp = prepared[name]
            A, s, Bt = np.linalg.svd(Xp, full_matrices=False)
            if s.size == 0 or s[0] <= 0:
                raise ValueError(f"view {name!r} has zero variance")
            keep = s > self.rank_tol * s[0]
            A, s, Bt = A[:, keep], s[keep], Bt[keep]
            svds[name] = (A, s, Bt)
            bases.append(A)

        ranks = [A.shape[1] for A in bases]
        total_rank = int(sum(ranks))
        if k > total_rank:
            raise ValueError(
                f"latent_dim ({k}) exceeds the total rank across views ({total_rank})"
            )

        # Gram matrix of the horizontally stacked bases, assembled block-wise so
        # the stacked (n x total_rank) matrix is never materialised.
        offsets = np.cumsum([0] + ranks)
        T = np.empty((total_rank, total_rank))
        for i in range(len(names)):
            for j in range(i, len(names)):
                block = bases[i].T @ bases[j]
                T[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = block
                if i != j:
                    T[offsets[j] : offsets[j + 1], offsets[i] : offsets[i + 1]] = block.T

        eigvals, eigvecs = np.linalg.eigh(T)
        order = np.argsort(eigvals)[::-1][:k]
        lam = np.maximum(eigvals[order], 1e-12)
        W = eigvecs[:, order]

        # G = (stacked bases) @ W / sqrt(lambda), computed block-wise.
        G = np.zeros((prepared[names[0]].shape[0], k))
        for i, name in enumerate(names):
            G += bases[i] @ W[offsets[i] : offsets[i + 1]]
        G /= np.sqrt(lam)

        # Per-view encoder: U_m = pinv(X_m) G, regularised.
        self.projections_ = {}
        for name in names:
            A, s, Bt = svds[name]
            damp = self.reg * (s[0] ** 2)
            filt = s / (s**2 + damp)
            self.projections_[name] = Bt.T @ (filt[:, None] * (A.T @ G))

        self.G_ = G
        self.eigenvalues_ = lam
        self.view_agreement_ = lam / len(names)
        self.n_views_ = len(names)

    def _encode(self, name: str, Xp: np.ndarray) -> np.ndarray:
        return Xp @ self.projections_[name]
