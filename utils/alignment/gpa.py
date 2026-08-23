"""Generalized Procrustes Analysis: rotate every model onto a common consensus.

The most conservative aligner in the suite, and for that reason the most
informative about the central hypothesis. It is restricted to *orthogonal*
maps: no stretching, no reweighting of directions, only rotation (and one
global scale per view). If a rigid rotation suffices to superimpose eight
independently trained models, they really are the same space in different
coordinates. If it does not, whatever CCA achieves is buying agreement by
distorting the geometry.

Algorithm
---------
Classical Gower/ten Berge iteration:

1. Reduce every view to ``latent_dim`` principal components (needed because the
   models have different widths and a rotation cannot change dimension).
2. Initialise the consensus as the first view.
3. Rotate each view onto the consensus by orthogonal Procrustes.
4. Recompute the consensus as the mean of the rotated views.
5. Repeat 3-4 until the consensus stops moving.

The decoder is exact rather than fitted: the inverse of an orthogonal map is
its transpose, so round-trip error comes only from the PCA truncation in step 1.

Reference
---------
Gower (1975), "Generalized Procrustes analysis", Psychometrika.
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import BaseAligner

__all__ = ["GeneralizedProcrustesAligner"]


class GeneralizedProcrustesAligner(BaseAligner):
    """Rigid (orthogonal) alignment of all models onto a common consensus.

    Parameters
    ----------
    latent_dim : int, default 64
        Shared space dimensionality. Every view is PCA-reduced to this many
        components before rotation.
    max_iter : int, default 300
        Maximum consensus iterations. Synthetic data converges in under ten;
        real foundation-model embeddings, which are far from being rotations
        of each other, need substantially more.
    tol : float, default 1e-7
        Convergence threshold on the relative change in consensus. The tail of
        the iteration converges slowly, so a tighter tolerance buys no
        meaningful accuracy and just trips the non-convergence warning.
    scale : bool, default True
        Fit one isotropic scale factor per view alongside its rotation. Leave
        on: models differ substantially in embedding norm, and a pure rotation
        would otherwise be forced to compromise.
    scaling : {'rms', 'std', 'none'}, default 'rms'
        Per-view preprocessing scale.
    decoder_reg : float, default 1e-6
        Unused for this aligner (the decoder is the exact transpose) but kept
        for interface compatibility.
    random_state : int, default 0
        Seed.

    Attributes
    ----------
    rotations_ : dict of str to numpy.ndarray
        Per-model orthogonal matrices, shape ``(latent_dim, latent_dim)``.
    scales_ : dict of str to float
        Per-model isotropic scale factors.
    consensus_ : numpy.ndarray
        The consensus configuration on the training patches, shape
        ``(n_train, latent_dim)``.
    n_iter_ : int
        Iterations actually run.
    residuals_ : list of float
        Consensus movement per iteration; a monotone decrease to ~0 confirms
        convergence.
    view_residuals_ : dict of str to float
        Final per-model relative residual ``||R_m(X_m) - consensus||^2 /
        ||consensus||^2``. Which models refuse to rotate into agreement is a
        direct answer to "which models are geometrically idiosyncratic".
    """

    def __init__(
        self,
        latent_dim: int = 64,
        max_iter: int = 300,
        tol: float = 1e-7,
        scale: bool = True,
        scaling: str = "rms",
        decoder_reg: float = 1e-6,
        random_state: int = 0,
    ):
        # PCA reduction to latent_dim is intrinsic to this method, so it is
        # wired into the preprocessing rather than left to the caller.
        super().__init__(
            latent_dim=latent_dim,
            pca_dim=latent_dim,
            scaling=scaling,
            decoder_reg=decoder_reg,
            random_state=random_state,
        )
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scale = bool(scale)

    def _fit(self, prepared: dict[str, np.ndarray]) -> None:
        names = list(prepared)
        k = self.latent_dim

        views = {}
        for name in names:
            Xp = prepared[name]
            if Xp.shape[1] != k:
                # PCA could not supply k components (view rank too low).
                Xp = np.pad(Xp, ((0, 0), (0, k - Xp.shape[1])))
            views[name] = Xp

        rotations = {name: np.eye(k) for name in names}
        scales = {name: 1.0 for name in names}
        consensus = views[names[0]].copy()

        self.residuals_ = []
        self.n_iter_ = 0
        for it in range(self.max_iter):
            for name in names:
                R, s = _procrustes_to(views[name], consensus, allow_scale=self.scale)
                rotations[name] = R
                scales[name] = s

            if self.scale:
                # Without a constraint the scales shrink geometrically (all
                # views scaled by c < 1 shrinks the consensus by c, which
                # shrinks the next scales again) and the iteration never
                # converges. Ten Berge's normalisation fixes the total scale.
                sq = sum(s**2 for s in scales.values())
                factor = np.sqrt(len(names) / max(sq, 1e-12))
                for name in names:
                    scales[name] *= factor

            new_consensus = np.mean(
                [scales[n] * (views[n] @ rotations[n]) for n in names], axis=0
            )

            denom = float(np.linalg.norm(consensus)) or 1.0
            shift = float(np.linalg.norm(new_consensus - consensus)) / denom
            consensus = new_consensus
            self.residuals_.append(shift)
            self.n_iter_ = it + 1
            if shift < self.tol:
                break
        else:
            warnings.warn(
                f"generalised Procrustes did not converge in {self.max_iter} "
                f"iterations (last shift {self.residuals_[-1]:.2e}).",
                RuntimeWarning,
                stacklevel=3,
            )

        self.rotations_ = rotations
        self.scales_ = scales
        self.consensus_ = consensus

        c_norm = float(np.linalg.norm(consensus) ** 2) or 1.0
        self.view_residuals_ = {
            name: float(
                np.linalg.norm(scales[name] * (views[name] @ rotations[name]) - consensus)
                ** 2
                / c_norm
            )
            for name in names
        }

    def _encode(self, name: str, Xp: np.ndarray) -> np.ndarray:
        k = self.latent_dim
        if Xp.shape[1] < k:
            Xp = np.pad(Xp, ((0, 0), (0, k - Xp.shape[1])))
        return self.scales_[name] * (Xp @ self.rotations_[name])

    def _decode(self, name: str, Z: np.ndarray) -> np.ndarray:
        """Exact inverse: transpose the rotation and undo the scale."""
        out = (Z / self.scales_[name]) @ self.rotations_[name].T
        n_out = self.preprocessors_[name].n_features_out_
        return out[:, :n_out]


def _procrustes_to(
    X: np.ndarray, target: np.ndarray, allow_scale: bool = True
) -> tuple[np.ndarray, float]:
    """Find the orthogonal map (and scale) taking X onto a target.

    Parameters
    ----------
    X : numpy.ndarray
        Source configuration of shape ``(n, k)``.
    target : numpy.ndarray
        Target configuration of shape ``(n, k)``.
    allow_scale : bool, default True
        Also fit an isotropic scale factor.

    Returns
    -------
    tuple
        ``(rotation, scale)`` minimising ``||scale * X @ rotation - target||_F``.
    """
    U, s, Vt = np.linalg.svd(X.T @ target, full_matrices=False)
    R = U @ Vt
    if not allow_scale:
        return R, 1.0
    denom = float(np.linalg.norm(X, "fro") ** 2)
    scale = float(s.sum() / denom) if denom > 0 else 1.0
    return R, scale
