"""Common machinery for shared-latent-space aligners.

Every Phase V method — GCCA, MCCA, generalised Procrustes, joint PCA, the
shared autoencoder, optimal transport — ultimately provides the same two
things per model:

* an **encoder** mapping that model's embedding into the shared latent space,
* a **decoder** mapping the shared latent space back into that model's space.

:class:`BaseAligner` owns everything those methods have in common: input
validation, per-view preprocessing (centering, scaling, optional PCA
pre-reduction), the fitted linear decoders, and serialisation. Subclasses
implement only :meth:`BaseAligner._fit` and :meth:`BaseAligner._encode`.

Preprocessing pipeline
----------------------
Each view is centered, divided by its per-sample RMS Frobenius norm, and
optionally PCA-reduced. The scaling matters more than it looks: pathology
foundation models produce embeddings whose norms differ by an order of
magnitude, and without it concatenation-based methods are dominated by
whichever model happens to have the largest activations. The scale is defined
per *sample* rather than per *dataset* so that it transfers unchanged to
held-out patches.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

from ..preprocessing import as_matrix

__all__ = ["BaseAligner", "ViewPreprocessor", "split_views"]


class ViewPreprocessor:
    """Per-view centering, scaling and optional PCA reduction.

    Parameters
    ----------
    scaling : {'rms', 'std', 'none'}, default 'rms'
        * ``'rms'`` — divide by the root-mean-square row norm, putting every
          view on a comparable overall scale while preserving its internal
          geometry.
        * ``'std'`` — divide each feature by its standard deviation
          (whitening-lite); changes the geometry, use only deliberately.
        * ``'none'`` — center only.
    pca_dim : int or None, default None
        Reduce each view to this many principal components before alignment.
        Strongly recommended for the concatenation-based methods when models
        have thousands of feature dimensions.
    whiten : bool, default False
        Scale PCA components to unit variance. Helps CCA-family conditioning,
        hurts methods that assume a preserved metric (Procrustes, OT).
    random_state : int, default 0
        Seed for the randomised SVD used when ``pca_dim`` is set.

    Attributes
    ----------
    mean_ : numpy.ndarray
        Per-feature mean removed at fit time, shape ``(d,)``.
    scale_ : float or numpy.ndarray
        Scaling applied after centering.
    components_ : numpy.ndarray or None
        PCA loadings of shape ``(pca_dim, d)``, or None if unused.
    explained_variance_ratio_ : numpy.ndarray or None
        Fraction of view variance retained per component.
    """

    def __init__(
        self,
        scaling: str = "rms",
        pca_dim: int | None = None,
        whiten: bool = False,
        random_state: int = 0,
    ):
        if scaling not in ("rms", "std", "none"):
            raise ValueError(
                f"unknown scaling {scaling!r}; expected 'rms', 'std' or 'none'"
            )
        self.scaling = scaling
        self.pca_dim = pca_dim
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "ViewPreprocessor":
        """Learn the preprocessing parameters from one view.

        Parameters
        ----------
        X : numpy.ndarray
            View matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        ViewPreprocessor
            Self, fitted.
        """
        X = as_matrix(X)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        if self.scaling == "rms":
            rms = float(np.sqrt((Xc**2).sum() / Xc.shape[0]))
            self.scale_ = max(rms, 1e-12)
        elif self.scaling == "std":
            self.scale_ = np.maximum(Xc.std(axis=0), 1e-12)
        else:
            self.scale_ = 1.0
        Xs = Xc / self.scale_

        if self.pca_dim is not None:
            k = min(self.pca_dim, min(Xs.shape))
            if k < self.pca_dim:
                warnings.warn(
                    f"pca_dim={self.pca_dim} exceeds the view rank; using {k}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
            self.components_ = Vt[:k]
            total = float((s**2).sum())
            self.explained_variance_ratio_ = (s[:k] ** 2) / total if total > 0 else None
            self.singular_values_ = s[:k]
            self.pca_scale_ = (
                np.maximum(s[:k] / np.sqrt(max(Xs.shape[0] - 1, 1)), 1e-12)
                if self.whiten
                else 1.0
            )
        else:
            self.components_ = None
            self.explained_variance_ratio_ = None
            self.singular_values_ = None
            self.pca_scale_ = 1.0

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = (
            self.components_.shape[0] if self.components_ is not None else X.shape[1]
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted preprocessing to a view.

        Parameters
        ----------
        X : numpy.ndarray
            View matrix of shape ``(n_samples, n_features_in)``.

        Returns
        -------
        numpy.ndarray
            Preprocessed matrix of shape ``(n_samples, n_features_out)``.
        """
        X = as_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        Xs = (X - self.mean_) / self.scale_
        if self.components_ is not None:
            Xs = (Xs @ self.components_.T) / self.pca_scale_
        return Xs

    def inverse_transform(self, Xp: np.ndarray) -> np.ndarray:
        """Undo the preprocessing, returning to the original feature space.

        Parameters
        ----------
        Xp : numpy.ndarray
            Preprocessed matrix of shape ``(n_samples, n_features_out)``.

        Returns
        -------
        numpy.ndarray
            Matrix of shape ``(n_samples, n_features_in)``. When ``pca_dim``
            was set this is a reconstruction, not an exact inverse — the
            discarded components are gone for good.
        """
        Xp = np.asarray(Xp, dtype=np.float64)
        if self.components_ is not None:
            Xp = (Xp * self.pca_scale_) @ self.components_
        return Xp * self.scale_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the preprocessing and apply it in one step."""
        return self.fit(X).transform(X)


def split_views(
    views: Mapping[str, np.ndarray],
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Split every view with one shared row partition.

    Alignment quality must always be read off held-out patches: a map fitted
    with more parameters than samples reconstructs its training set perfectly
    and tells you nothing. Because the views are row-paired, the *same* split
    has to be applied to all of them.

    Parameters
    ----------
    views : mapping of str to array-like
        ``{model_name: embedding_matrix}``, row-paired.
    test_size : float, default 0.2
        Fraction of patches held out.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    tuple
        ``(train_views, test_views, train_idx, test_idx)``.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    n_rows = {as_matrix(v, copy=False).shape[0] for v in views.values()}
    if len(n_rows) != 1:
        raise ValueError("all views must have the same number of rows")
    n = n_rows.pop()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])

    train = {k: as_matrix(v)[train_idx] for k, v in views.items()}
    test = {k: as_matrix(v)[test_idx] for k, v in views.items()}
    return train, test, train_idx, test_idx


class BaseAligner(ABC):
    """Abstract base class for shared-latent-space aligners.

    Subclasses implement :meth:`_fit` (learn the shared space from all
    preprocessed views) and :meth:`_encode` (map one preprocessed view into
    it). Everything else — preprocessing, decoders, reconstruction, I/O — is
    inherited.

    Parameters
    ----------
    latent_dim : int, default 64
        Dimensionality of the shared latent space.
    pca_dim : int or None, default None
        Per-view PCA pre-reduction before alignment. ``None`` uses full
        embeddings.
    scaling : {'rms', 'std', 'none'}, default 'rms'
        Per-view scaling, see :class:`ViewPreprocessor`.
    whiten : bool, default False
        Scale each view's PCA components to unit variance. Requires
        ``pca_dim``. Necessary for the optimal-transport aligner (two clouds
        with different variance spectra are not related by any rotation) and
        harmless-to-helpful for the CCA family, which whitens internally
        anyway.
    decoder_reg : float, default 1e-6
        Ridge penalty for the least-squares decoders, relative to the mean
        eigenvalue of the latent covariance.
    random_state : int, default 0
        Seed.

    Attributes
    ----------
    view_names_ : list of str
        Model names, in fit order.
    preprocessors_ : dict of str to ViewPreprocessor
        Fitted per-view preprocessing.
    decoders_ : dict of str to numpy.ndarray
        Fitted linear maps from the shared space back to each preprocessed
        view, shape ``(latent_dim + 1, n_features_out)`` (last row is the
        intercept).
    is_fitted_ : bool
        Whether :meth:`fit` has completed.
    """

    #: Set by subclasses that produce a single consensus embedding rather than
    #: one embedding per view (currently none; kept for clarity of intent).
    _consensus_only = False

    def __init__(
        self,
        latent_dim: int = 64,
        pca_dim: int | None = None,
        scaling: str = "rms",
        whiten: bool = False,
        decoder_reg: float = 1e-6,
        random_state: int = 0,
    ):
        self.latent_dim = int(latent_dim)
        self.pca_dim = pca_dim
        self.scaling = scaling
        self.whiten = bool(whiten)
        self.decoder_reg = float(decoder_reg)
        self.random_state = int(random_state)
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit(self, prepared: dict[str, np.ndarray]) -> None:
        """Learn the shared space from preprocessed views.

        Parameters
        ----------
        prepared : dict of str to numpy.ndarray
            Preprocessed, row-paired views.
        """

    @abstractmethod
    def _encode(self, name: str, Xp: np.ndarray) -> np.ndarray:
        """Map one preprocessed view into the shared latent space.

        Parameters
        ----------
        name : str
            Model name.
        Xp : numpy.ndarray
            Preprocessed view of shape ``(n_samples, n_features_out)``.

        Returns
        -------
        numpy.ndarray
            Shared-space coordinates of shape ``(n_samples, latent_dim)``.
        """

    def _decode(self, name: str, Z: np.ndarray) -> np.ndarray:
        """Map shared-space coordinates back to a preprocessed view.

        The default is the ridge-regression decoder fitted in :meth:`fit`.
        Subclasses with a native inverse (e.g. the autoencoder, or Procrustes'
        exact transpose) override this.

        Parameters
        ----------
        name : str
            Model name.
        Z : numpy.ndarray
            Shared-space coordinates of shape ``(n_samples, latent_dim)``.

        Returns
        -------
        numpy.ndarray
            Preprocessed-space reconstruction.
        """
        W = self.decoders_[name]
        return Z @ W[:-1] + W[-1]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, views: Mapping[str, np.ndarray]) -> "BaseAligner":
        """Fit the shared latent space across all views.

        Parameters
        ----------
        views : mapping of str to array-like
            ``{model_name: embedding_matrix}``. All matrices must have the
            same number of rows, in the same patch order; feature dimensions
            may differ.

        Returns
        -------
        BaseAligner
            Self, fitted.
        """
        prepared = self._prepare_fit(views)
        self._fit(prepared)
        self._fit_decoders(prepared)
        self.is_fitted_ = True
        return self

    def transform(self, views: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map several views into the shared latent space.

        Parameters
        ----------
        views : mapping of str to array-like
            ``{model_name: embedding_matrix}``. May be a subset of the models
            seen at fit time — that is the point of per-model projections.

        Returns
        -------
        dict of str to numpy.ndarray
            ``{model_name: shared_coordinates}``, each ``(n_samples, latent_dim)``.
        """
        self._check_fitted()
        return {name: self.transform_view(name, X) for name, X in views.items()}

    def transform_view(self, name: str, X: np.ndarray) -> np.ndarray:
        """Map a single model's embeddings into the shared latent space.

        Parameters
        ----------
        name : str
            Model name, as seen at fit time.
        X : array-like
            Embedding matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Shared-space coordinates of shape ``(n_samples, latent_dim)``.
        """
        self._check_fitted()
        self._check_view(name)
        return self._encode(name, self.preprocessors_[name].transform(X))

    def inverse_transform(self, Z: np.ndarray, name: str) -> np.ndarray:
        """Map shared-space coordinates back into one model's space.

        This is the "out of the shared space" projection, and the basis for
        cross-model transfer in Phase VI: encode with model A, decode as model
        B.

        Parameters
        ----------
        Z : array-like
            Shared-space coordinates of shape ``(n_samples, latent_dim)``.
        name : str
            Target model name.

        Returns
        -------
        numpy.ndarray
            Reconstruction in the target model's original feature space.
        """
        self._check_fitted()
        self._check_view(name)
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[1] != self.latent_dim:
            raise ValueError(
                f"expected shared coordinates with {self.latent_dim} dimensions, "
                f"got shape {Z.shape}"
            )
        return self.preprocessors_[name].inverse_transform(self._decode(name, Z))

    def fit_transform(self, views: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Fit the shared space and return the transformed views."""
        return self.fit(views).transform(views)

    def reconstruct(self, views: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Round-trip each view through the shared space.

        Parameters
        ----------
        views : mapping of str to array-like
            ``{model_name: embedding_matrix}``.

        Returns
        -------
        dict of str to numpy.ndarray
            Reconstructions in each model's original feature space.
        """
        return {
            name: self.inverse_transform(self.transform_view(name, X), name)
            for name, X in views.items()
        }

    def translate(self, X: np.ndarray, source: str, target: str) -> np.ndarray:
        """Convert one model's embeddings into another's via the shared space.

        The Phase VI operation: ``CONCH -> shared -> UNI``.

        Parameters
        ----------
        X : array-like
            Embeddings in the source model's space.
        source, target : str
            Model names.

        Returns
        -------
        numpy.ndarray
            Embeddings in the target model's feature space.
        """
        return self.inverse_transform(self.transform_view(source, X), target)

    def consensus(self, views: Mapping[str, np.ndarray]) -> np.ndarray:
        """Average the shared-space coordinates over all supplied views.

        The consensus embedding is the natural "one representation of this
        patch" to hand to a downstream MIL model in Phase VIII, and is more
        robust than any single view's projection.

        Parameters
        ----------
        views : mapping of str to array-like
            ``{model_name: embedding_matrix}``.

        Returns
        -------
        numpy.ndarray
            Mean shared-space coordinates, shape ``(n_samples, latent_dim)``.
        """
        Zs = self.transform(views)
        return np.mean(np.stack(list(Zs.values()), axis=0), axis=0)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _prepare_fit(self, views: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Validate views, fit preprocessors, and return preprocessed views."""
        if len(views) < 2:
            raise ValueError(
                f"a shared latent space needs at least 2 models, got {len(views)}"
            )

        mats = {name: as_matrix(X) for name, X in views.items()}
        n_rows = {X.shape[0] for X in mats.values()}
        if len(n_rows) != 1:
            detail = ", ".join(f"{k}={v.shape[0]}" for k, v in mats.items())
            raise ValueError(
                "all views must be row-paired (same patches, same order); "
                f"got sample counts: {detail}"
            )
        n = n_rows.pop()

        if n <= self.latent_dim:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be smaller than the number "
                f"of samples ({n})"
            )

        max_latent = min(
            (self.pca_dim or X.shape[1]) if X.shape[1] else 0 for X in mats.values()
        )
        max_latent = min(max_latent, min(X.shape[1] for X in mats.values()))
        if self.latent_dim > max_latent:
            warnings.warn(
                f"latent_dim ({self.latent_dim}) exceeds the smallest view's "
                f"usable dimension ({max_latent}); the extra dimensions cannot "
                "carry independent information.",
                RuntimeWarning,
                stacklevel=3,
            )

        self.view_names_ = list(mats)
        self.n_samples_fit_ = n
        self.preprocessors_ = {}
        prepared = {}
        for name, X in mats.items():
            pp = ViewPreprocessor(
                scaling=self.scaling,
                pca_dim=self.pca_dim,
                whiten=self.whiten,
                random_state=self.random_state,
            )
            prepared[name] = pp.fit_transform(X)
            self.preprocessors_[name] = pp
        return prepared

    def _fit_decoders(self, prepared: dict[str, np.ndarray]) -> None:
        """Fit ridge decoders from the shared space back to each view."""
        self.decoders_ = {}
        for name, Xp in prepared.items():
            Z = self._encode(name, Xp)
            self.decoders_[name] = _ridge_fit(Z, Xp, self.decoder_reg)

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call fit(views) first"
            )

    def _check_view(self, name: str) -> None:
        if name not in self.preprocessors_:
            raise KeyError(
                f"unknown model {name!r}; fitted models are {self.view_names_}"
            )

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Persist the fitted aligner to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file (joblib format).
        """
        from pathlib import Path

        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @staticmethod
    def load(path) -> "BaseAligner":
        """Load a fitted aligner from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            File written by :meth:`save`.

        Returns
        -------
        BaseAligner
            The restored aligner.
        """
        import joblib

        return joblib.load(path)

    def __repr__(self) -> str:
        state = "fitted" if getattr(self, "is_fitted_", False) else "unfitted"
        views = getattr(self, "view_names_", [])
        return (
            f"{type(self).__name__}(latent_dim={self.latent_dim}, "
            f"pca_dim={self.pca_dim}, {state}, views={list(views)})"
        )


def _ridge_fit(Z: np.ndarray, Y: np.ndarray, reg: float) -> np.ndarray:
    """Least-squares fit of ``Y ~ [Z, 1]`` with a relative ridge penalty.

    Parameters
    ----------
    Z : numpy.ndarray
        Predictors of shape ``(n, k)``.
    Y : numpy.ndarray
        Targets of shape ``(n, d)``.
    reg : float
        Ridge penalty, scaled by the mean diagonal of ``Z.T @ Z`` so it is
        invariant to the latent space's overall scale.

    Returns
    -------
    numpy.ndarray
        Coefficients of shape ``(k + 1, d)``; the last row is the intercept.
    """
    n, k = Z.shape
    Za = np.hstack([Z, np.ones((n, 1))])
    A = Za.T @ Za
    lam = reg * (np.trace(A[:k, :k]) / max(k, 1))
    A[:k, :k] += lam * np.eye(k)
    B = Za.T @ Y
    return np.linalg.solve(A, B)
