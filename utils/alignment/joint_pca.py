"""Joint PCA: the simplest possible shared latent space.

Concatenate every model's embedding of a patch into one long vector and take
the leading principal components of the result. No correlation structure is
optimised, no iteration is run — which makes this the baseline the more
sophisticated aligners have to beat. If joint PCA matches GCCA on downstream
tasks, the extra machinery is not earning its place in the paper.

Per-model projections
---------------------
Plain joint PCA requires *all* views at once, which would make it useless for
the per-model projection functions Phase V calls for. So the latent space is
defined by the joint PCA, and each model then gets its own ridge-regression
encoder fitted to predict those shared coordinates from that model alone. Each
encoder's R^2 is retained as ``encoder_r2_``: it measures how much of the
shared space a single model can recover on its own, which is a genuinely
interesting quantity — a model with low R^2 contributes information the others
lack.
"""

from __future__ import annotations

import numpy as np

from .base import BaseAligner, _ridge_fit

__all__ = ["JointPCAAligner"]


class JointPCAAligner(BaseAligner):
    """Shared latent space from PCA of the concatenated views.

    Parameters
    ----------
    latent_dim : int, default 64
        Number of joint principal components.
    encoder_reg : float, default 1e-4
        Ridge penalty for the per-model encoders.
    whiten : bool, default False
        Scale each latent axis to unit variance. Off by default because it
        distorts local neighbourhoods — the evaluation module's
        ``neighborhood_preservation`` drops noticeably when it is on. Turn it
        on only if a downstream consumer needs isotropic latent axes.
    pca_dim : int or None, default None
        Optional per-view PCA pre-reduction applied *before* concatenation.
        Also equalises each model's contribution when widths differ a lot
        (e.g. CONCH's 512 against Virchow's 2560).
    scaling : {'rms', 'std', 'none'}, default 'rms'
        Per-view scaling. Do not set to ``'none'`` here — with raw scales the
        concatenation is dominated by whichever model has the largest norms,
        and the joint components degenerate into that model's own PCA.
    decoder_reg : float, default 1e-6
        Ridge penalty for the decoders.
    random_state : int, default 0
        Seed.

    Attributes
    ----------
    components_ : numpy.ndarray
        Joint PCA loadings over the concatenated space, shape
        ``(latent_dim, total_features)``.
    explained_variance_ratio_ : numpy.ndarray
        Fraction of total concatenated variance per component.
    encoders_ : dict of str to numpy.ndarray
        Per-model ridge encoders, shape ``(n_features_out + 1, latent_dim)``.
    encoder_r2_ : dict of str to float
        In-sample R^2 of each model's encoder against the joint coordinates.
    view_loadings_ : dict of str to float
        Fraction of each joint component's squared loading mass belonging to
        each model, averaged over components — how much each model drives the
        shared space.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        encoder_reg: float = 1e-4,
        whiten: bool = False,
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
        self.encoder_reg = float(encoder_reg)
        self.whiten = bool(whiten)

    def _fit(self, prepared: dict[str, np.ndarray]) -> None:
        names = list(prepared)
        dims = [prepared[n].shape[1] for n in names]
        offsets = np.cumsum([0] + dims)
        k = self.latent_dim

        X = np.hstack([prepared[n] for n in names])
        if k > min(X.shape):
            raise ValueError(
                f"latent_dim ({k}) exceeds the concatenated matrix rank "
                f"({min(X.shape)})"
            )

        # X is already centered view-by-view, so its column means are ~0.
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[:k]
        total = float((s**2).sum()) or 1.0
        self.explained_variance_ratio_ = (s[:k] ** 2) / total

        Z = U[:, :k] * s[:k]
        if self.whiten:
            # Unit variance per axis makes latent distances comparable across
            # dimensions, at the cost of amplifying low-variance directions and
            # distorting local neighbourhoods.
            z_std = np.maximum(Z.std(axis=0), 1e-12)
            Z = Z / z_std
        else:
            z_std = np.ones(k)
        self.z_std_ = z_std

        self.encoders_ = {
            name: _ridge_fit(prepared[name], Z, self.encoder_reg) for name in names
        }

        ss_tot = float((Z**2).sum()) or 1.0
        self.encoder_r2_ = {}
        for name in names:
            pred = self._encode(name, prepared[name])
            self.encoder_r2_[name] = float(1.0 - ((Z - pred) ** 2).sum() / ss_tot)

        # How much of each component's loading mass belongs to each model.
        mass = self.components_**2
        self.view_loadings_ = {
            name: float(mass[:, offsets[i] : offsets[i + 1]].sum(axis=1).mean())
            for i, name in enumerate(names)
        }
        self.joint_scores_ = Z

    def _encode(self, name: str, Xp: np.ndarray) -> np.ndarray:
        W = self.encoders_[name]
        return Xp @ W[:-1] + W[-1]
