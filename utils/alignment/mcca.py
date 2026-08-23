"""Multi-view Canonical Correlation Analysis (SUM-COR formulation).

Where GCCA posits an explicit latent matrix and asks each view to approximate
it, MCCA works entirely in terms of *correlations between views*: find one
direction per view such that the resulting projections are maximally correlated
with each other, then repeat for successive orthogonal directions.

.. math:: \\max_{w_1 \\dots w_M} \\sum_{m \\neq l} w_m^\\top C_{ml} w_l
          \\quad \\text{s.t.} \\quad \\sum_m w_m^\\top C_{mm} w_m = 1

which relaxes to the generalised eigenvalue problem :math:`C w = \\lambda D w`,
with C the full block covariance of the concatenated views and D its
block-diagonal part.

Reference
---------
Kettenring (1971), Biometrika — SUMCOR among the five classical multiset
generalisations. For two views this reduces exactly to ordinary CCA, which the
test suite checks.

Practical note
--------------
The eigenproblem is over the *summed* feature dimension, so with eight models
at full width it is a ~10000 x 10000 dense problem. Use ``pca_dim`` (128-256)
to keep it tractable; that also regularises the solution, which matters
because MCCA whitens each view and is otherwise happy to lock onto noise
directions.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from .base import BaseAligner

__all__ = ["MCCAAligner"]


class MCCAAligner(BaseAligner):
    """Multi-view CCA (SUMCOR) shared latent space.

    Parameters
    ----------
    latent_dim : int, default 64
        Number of shared canonical directions.
    reg : float, default 1e-3
        Ridge regularisation added to each view's within-covariance block,
        relative to that block's mean eigenvalue. MCCA is considerably more
        sensitive to this than GCCA — with high-dimensional embeddings the
        within-covariance is near-singular and an unregularised solve returns
        canonical correlations of 1.0 that mean nothing.
    pca_dim : int or None, default 128
        Per-view PCA pre-reduction. Defaults to 128 here (unlike other
        aligners) because the dense generalised eigenproblem scales cubically
        in the summed dimension.
    scaling : {'rms', 'std', 'none'}, default 'rms'
        Per-view scaling.
    decoder_reg : float, default 1e-6
        Ridge penalty for the decoders.
    random_state : int, default 0
        Seed.

    Attributes
    ----------
    projections_ : dict of str to numpy.ndarray
        Per-model encoders, shape ``(n_features_out, latent_dim)``.
    eigenvalues_ : numpy.ndarray
        Generalised eigenvalues, one per latent dimension.
    correlations_ : numpy.ndarray
        Mean pairwise correlation between views along each latent direction,
        in ``[0, 1]``. The interpretable version of ``eigenvalues_``: how well
        the models actually agree on each shared direction.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        reg: float = 1e-3,
        pca_dim: int | None = 128,
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

    def _fit(self, prepared: dict[str, np.ndarray]) -> None:
        names = list(prepared)
        dims = [prepared[n].shape[1] for n in names]
        offsets = np.cumsum([0] + dims)
        n = prepared[names[0]].shape[0]
        k = self.latent_dim

        if k > min(dims):
            raise ValueError(
                f"latent_dim ({k}) cannot exceed the smallest view dimension "
                f"({min(dims)}); reduce latent_dim or raise pca_dim"
            )

        X = np.hstack([prepared[nm] for nm in names])
        C = (X.T @ X) / max(n - 1, 1)

        # D: block-diagonal within-view covariance, ridge-regularised.
        D = np.zeros_like(C)
        for i in range(len(names)):
            lo, hi = offsets[i], offsets[i + 1]
            block = C[lo:hi, lo:hi]
            lam = self.reg * (np.trace(block) / max(block.shape[0], 1))
            D[lo:hi, lo:hi] = block + lam * np.eye(block.shape[0])

        eigvals, eigvecs = scipy.linalg.eigh(C, D)
        order = np.argsort(eigvals)[::-1][:k]
        eigvals = eigvals[order]
        W = eigvecs[:, order]

        self.projections_ = {
            name: W[offsets[i] : offsets[i + 1]] for i, name in enumerate(names)
        }
        self.eigenvalues_ = eigvals
        self.n_views_ = M = len(names)

        # For SUMCOR the eigenvalue relates to the mean pairwise correlation as
        # lambda = 1 + (M - 1) * rho_bar, so invert that for reporting.
        self.correlations_ = np.clip((eigvals - 1.0) / max(M - 1, 1), 0.0, 1.0)

        # Normalise each direction so the shared space has unit-variance axes,
        # making latent distances comparable across dimensions.
        Zs = [prepared[nm] @ self.projections_[nm] for nm in names]
        stds = np.mean([Z.std(axis=0) for Z in Zs], axis=0)
        stds = np.maximum(stds, 1e-12)
        for name in names:
            self.projections_[name] = self.projections_[name] / stds

    def _encode(self, name: str, Xp: np.ndarray) -> np.ndarray:
        return Xp @ self.projections_[name]
