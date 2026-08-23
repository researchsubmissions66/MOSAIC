"""Optimal-transport alignment, including the unsupervised regime.

Every other aligner here consumes the patch pairing: it knows that row i of
UNI's matrix and row i of CONCH's matrix are the same tissue. That makes the
alignment problem comparatively easy, and it means a successful alignment shows
only that a map *exists* given correspondences.

Optimal transport can in principle drop that assumption. Given two clouds of
embeddings with no correspondence at all, it simultaneously estimates a soft
matching between them and the rotation that best superimposes them.

Method
------
Wasserstein Procrustes (Grave, Joulin & Berthet, AISTATS 2019), alternating:

1. with the rotation R fixed, solve entropic OT between ``X R`` and ``Y`` to
   get a soft coupling P (Sinkhorn, log-domain);
2. with P fixed, update R by orthogonal Procrustes on the coupled
   cross-covariance ``X^T P Y``.

Both steps run on minibatches, with the cross-covariance accumulated as an
exponential moving average so the rotation moves smoothly, and the entropic
regularisation annealed from smooth to sharp. Sinkhorn is implemented here
directly rather than via POT, to avoid the dependency.

Status of the two modes
-----------------------
**Supervised (the default, and what you should use).** With the row pairing
supplied, this is exact orthogonal Procrustes on whitened clouds. It recovers
planted rotations perfectly in testing (matching accuracy 1.000).

**Unsupervised (`supervised=False`) — exploratory, do not rely on it.** On
synthetic data with a planted shared manifold, this implementation reached at
best ~15-25x chance matching accuracy and frequently landed at chance. Two
distinct obstacles are involved:

* *Identifiability.* A rotationally symmetric cloud (an isotropic Gaussian)
  carries no information that could pin down the rotation — no algorithm can
  succeed there, and the whitening this method requires deliberately removes
  the second-order structure, leaving only higher moments to identify R. Real
  patch embeddings cluster by tissue type, so the structure exists in
  principle, but how much is an empirical question.
* *Optimisation.* The objective is non-convex, and the transport cost used to
  pick among restarts barely discriminates between good and bad rotations.

Treat any unsupervised result as a lower bound that needs its own tuning, and
report it against the ``1 / batch_size`` chance level rather than in absolute
terms. The supervised/unsupervised gap is itself the interesting quantity.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .base import BaseAligner

__all__ = ["OptimalTransportAligner", "sinkhorn_log", "sinkhorn_coupling"]


def sinkhorn_log(
    C: np.ndarray,
    reg: float,
    n_iter: int = 200,
    tol: float = 1e-9,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Entropic optimal transport in the log domain.

    The log-domain formulation is numerically stable for the small ``reg``
    values needed to get a near-deterministic matching, where the naive
    matrix-scaling implementation underflows.

    Parameters
    ----------
    C : numpy.ndarray
        Cost matrix of shape ``(n, m)``.
    reg : float
        Entropic regularisation strength. Smaller gives a sharper (more
        permutation-like) coupling but converges more slowly.
    n_iter : int, default 200
        Maximum Sinkhorn iterations.
    tol : float, default 1e-9
        Convergence tolerance on the dual potentials.
    a, b : numpy.ndarray, optional
        Source and target marginals. Default to uniform.

    Returns
    -------
    tuple of numpy.ndarray
        Dual potentials ``(f, g)``, from which the coupling is
        ``exp((f_i + g_j - C_ij)/reg) * a_i * b_j``.
    """
    n, m = C.shape
    log_a = np.log(a) if a is not None else np.full(n, -np.log(n))
    log_b = np.log(b) if b is not None else np.full(m, -np.log(m))

    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(n_iter):
        f_prev = f
        f = -reg * logsumexp((g[None, :] - C) / reg + log_b[None, :], axis=1)
        g = -reg * logsumexp((f[:, None] - C) / reg + log_a[:, None], axis=0)
        if np.max(np.abs(f - f_prev)) < tol:
            break
    return f, g


def sinkhorn_coupling(
    C: np.ndarray, reg: float, n_iter: int = 200
) -> np.ndarray:
    """Compute the entropic OT coupling matrix.

    Parameters
    ----------
    C : numpy.ndarray
        Cost matrix of shape ``(n, m)``.
    reg : float
        Entropic regularisation strength.
    n_iter : int, default 200
        Maximum Sinkhorn iterations.

    Returns
    -------
    numpy.ndarray
        Coupling of shape ``(n, m)``, with rows summing to ``1/n`` and columns
        to ``1/m``.
    """
    n, m = C.shape
    f, g = sinkhorn_log(C, reg, n_iter=n_iter)
    log_a = np.full(n, -np.log(n))
    log_b = np.full(m, -np.log(m))
    return np.exp(
        (f[:, None] + g[None, :] - C) / reg + log_a[:, None] + log_b[None, :]
    )


def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared Euclidean cost matrix between two point clouds."""
    return np.maximum(
        (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2.0 * (X @ Y.T), 0.0
    )


class OptimalTransportAligner(BaseAligner):
    """Align models by Wasserstein Procrustes onto a reference model's space.

    Parameters
    ----------
    latent_dim : int, default 64
        Shared space dimensionality; every view is PCA-reduced to it.
    reference : str or None, default None
        Model whose (reduced) space becomes the shared space. ``None`` uses the
        first model supplied.
    supervised : bool, default True
        Use the known row pairing instead of estimating a coupling, which
        reduces the method to exact orthogonal Procrustes on the whitened
        clouds. This is the default because it is the mode that reliably
        works — see the warning about ``supervised=False`` in the class notes.
    reg : float, default 2.0
        Starting entropic regularisation, relative to the mean cost of each
        batch (so it is scale-free).
    reg_end : float, default 0.01
        Final entropic regularisation; the schedule decays geometrically from
        ``reg`` to ``reg_end`` over the iterations. Annealing matters: a large
        initial value gives a smooth, near-uniform coupling that lets the
        rotation move freely, and shrinking it sharpens the matching once the
        rotation is roughly right. A fixed small value gets stuck immediately.
    whiten : bool, default True
        Whiten each view's PCA components to unit variance. **Required for this
        method.** Two clouds whose principal-component spectra differ are not
        related by any rotation, so without whitening the objective has no good
        solution to find — in testing, unsupervised alignment sat at exactly
        chance until this was enabled.
    batch_size : int, default 500
        Minibatch size for the OT subproblems. Cost is O(batch_size^2) in
        memory and O(batch_size^2 * n_sinkhorn) in time.
    n_iter : int, default 300
        Alternating iterations.
    sinkhorn_iter : int, default 100
        Sinkhorn iterations per OT subproblem.
    momentum : float, default 0.9
        EMA coefficient for the accumulated cross-covariance. High values
        stabilise the rotation across noisy minibatches.
    n_restarts : int, default 3
        Random restarts; the run with the lowest final transport cost wins.
        The unsupervised objective is non-convex and genuinely does get stuck.
    scaling : {'rms', 'std', 'none'}, default 'rms'
        Per-view scaling.
    decoder_reg : float, default 1e-6
        Ridge penalty for the decoders.
    verbose : bool, default False
        Print per-restart progress.
    random_state : int, default 0
        Seed.

    Attributes
    ----------
    rotations_ : dict of str to numpy.ndarray
        Per-model orthogonal maps into the reference space.
    transport_costs_ : dict of str to float
        Final transport cost per model; lower means the two clouds superimpose
        better.
    matching_accuracy_ : dict of str to float
        Fraction of a held-out batch whose OT-matched partner under the fitted
        rotation is the true paired patch; chance is ``1 / batch_size``. Under
        ``supervised=True`` this is a sanity check on the fit (expect ~1.0
        when the models really are rotations of each other). Under
        ``supervised=False`` it is the experiment's actual result — but see
        the module docstring on why that mode is unreliable.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        reference: str | None = None,
        supervised: bool = True,
        reg: float = 2.0,
        reg_end: float = 0.01,
        whiten: bool = True,
        batch_size: int = 500,
        n_iter: int = 300,
        sinkhorn_iter: int = 100,
        momentum: float = 0.9,
        n_restarts: int = 3,
        scaling: str = "rms",
        decoder_reg: float = 1e-6,
        verbose: bool = False,
        random_state: int = 0,
    ):
        super().__init__(
            latent_dim=latent_dim,
            pca_dim=latent_dim,
            scaling=scaling,
            whiten=whiten,
            decoder_reg=decoder_reg,
            random_state=random_state,
        )
        self.reference = reference
        self.supervised = bool(supervised)
        self.reg = float(reg)
        self.reg_end = float(reg_end)
        self.batch_size = int(batch_size)
        self.n_iter = int(n_iter)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.momentum = float(momentum)
        self.n_restarts = int(n_restarts)
        self.verbose = bool(verbose)

    def _fit(self, prepared: dict[str, np.ndarray]) -> None:
        names = list(prepared)
        k = self.latent_dim

        ref = self.reference or names[0]
        if ref not in prepared:
            raise KeyError(f"reference {ref!r} is not among the views {names}")
        self.reference_ = ref

        views = {}
        for name in names:
            Xp = prepared[name]
            if Xp.shape[1] < k:
                Xp = np.pad(Xp, ((0, 0), (0, k - Xp.shape[1])))
            views[name] = Xp
        target = views[ref]

        self.rotations_ = {ref: np.eye(k)}
        self.transport_costs_ = {ref: 0.0}
        self.matching_accuracy_ = {}

        for name in names:
            if name == ref:
                continue
            R, cost = self._align_pair(views[name], target, name)
            self.rotations_[name] = R
            self.transport_costs_[name] = cost
            self.matching_accuracy_[name] = self._matching_accuracy(
                views[name] @ R, target
            )

    def _align_pair(
        self, X: np.ndarray, Y: np.ndarray, name: str
    ) -> tuple[np.ndarray, float]:
        """Estimate the orthogonal map taking X's cloud onto Y's cloud."""
        k = X.shape[1]

        if self.supervised:
            U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
            R = U @ Vt
            return R, float(np.mean(np.sum((X @ R - Y) ** 2, axis=1)))

        n = X.shape[0]
        batch = min(self.batch_size, n)

        # One fixed evaluation batch shared by every restart. Scoring each
        # restart on its own random batch would compare incomparable numbers —
        # the between-batch variance swamps the between-restart differences.
        eval_rng = np.random.default_rng(self.random_state + 7919)
        eval_i = eval_rng.choice(n, size=batch, replace=False)
        eval_j = eval_rng.choice(n, size=batch, replace=False)

        best_R, best_cost = None, np.inf
        for restart in range(self.n_restarts):
            rng = np.random.default_rng(self.random_state + restart)
            R = np.eye(k) if restart == 0 else _random_orthogonal(k, rng)
            M_ema = np.zeros((k, k))

            for it in range(self.n_iter):
                i = rng.choice(n, size=batch, replace=False)
                j = rng.choice(n, size=batch, replace=False)
                Xb, Yb = X[i] @ R, Y[j]

                C = _sq_dists(Xb, Yb)
                reg = self._reg_at(it) * max(float(C.mean()), 1e-12)
                P = sinkhorn_coupling(C, reg, n_iter=self.sinkhorn_iter)

                M = X[i].T @ (P @ Yb) * batch
                M_ema = self.momentum * M_ema + (1 - self.momentum) * M
                U, _, Vt = np.linalg.svd(M_ema, full_matrices=False)
                R = U @ Vt

            cost = self._transport_cost(X[eval_i] @ R, Y[eval_j])
            if cost < best_cost:
                best_cost, best_R = cost, R
            if self.verbose:
                print(f"  [{name}] restart {restart}: cost={cost:.4f}")

        if best_R is None:  # pragma: no cover - defensive
            raise RuntimeError(f"alignment failed for view {name!r}")
        return best_R, float(best_cost)

    def _reg_at(self, it: int) -> float:
        """Geometrically annealed entropic regularisation at iteration ``it``."""
        if self.n_iter <= 1:
            return self.reg_end
        frac = it / (self.n_iter - 1)
        return float(self.reg * (self.reg_end / self.reg) ** frac)

    def _transport_cost(self, Xb: np.ndarray, Yb: np.ndarray) -> float:
        """Entropic transport cost between two clouds, for restart selection.

        Parameters
        ----------
        Xb, Yb : numpy.ndarray
            Already-subsampled, already-rotated point clouds.

        Returns
        -------
        float
            Transport cost under the final (sharp) regularisation.
        """
        C = _sq_dists(Xb, Yb)
        reg = self.reg_end * max(float(C.mean()), 1e-12)
        P = sinkhorn_coupling(C, reg, n_iter=self.sinkhorn_iter)
        return float((P * C).sum())

    def _matching_accuracy(
        self, X_aligned: np.ndarray, Y: np.ndarray, batch: int | None = None
    ) -> float:
        """Fraction of a random batch whose OT match is the true paired patch.

        Uses the row pairing only for *evaluation*, never for fitting.
        """
        rng = np.random.default_rng(self.random_state + 1)
        n = X_aligned.shape[0]
        batch = min(batch or self.batch_size, n)
        idx = rng.choice(n, size=batch, replace=False)

        C = _sq_dists(X_aligned[idx], Y[idx])
        reg = self.reg_end * max(float(C.mean()), 1e-12)
        P = sinkhorn_coupling(C, reg, n_iter=self.sinkhorn_iter)
        return float(np.mean(np.argmax(P, axis=1) == np.arange(batch)))

    def _encode(self, name: str, Xp: np.ndarray) -> np.ndarray:
        k = self.latent_dim
        if Xp.shape[1] < k:
            Xp = np.pad(Xp, ((0, 0), (0, k - Xp.shape[1])))
        return Xp @ self.rotations_[name]

    def _decode(self, name: str, Z: np.ndarray) -> np.ndarray:
        """Exact inverse: the transpose of the orthogonal map."""
        out = Z @ self.rotations_[name].T
        return out[:, : self.preprocessors_[name].n_features_out_]


def _random_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random orthogonal matrix from the Haar measure."""
    Q, R = np.linalg.qr(rng.normal(size=(k, k)))
    return Q * np.sign(np.diag(R))
