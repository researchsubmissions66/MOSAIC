"""Phase V: constructing a latent space shared across pathology foundation models.

Six approaches, one interface. Every aligner exposes:

* ``fit(views)`` — learn the shared space from row-paired embeddings,
* ``transform_view(name, X)`` — project one model into the shared space,
* ``inverse_transform(Z, name)`` — project back out into a model's space,
* ``translate(X, source, target)`` — convert between models via the shared space,
* ``consensus(views)`` — the mean shared embedding across models.

Choosing a method
-----------------
======================  ===========================================================
Aligner                 Use it to answer
======================  ===========================================================
``JointPCAAligner``     The baseline. Does anything more elaborate beat plain
                        concatenation + PCA?
``GCCAAligner``         Is there one latent matrix every model is a linear view
                        of? Its eigenvalue spectrum says how many dimensions of
                        morphology the models actually share.
``MCCAAligner``         Same question posed through pairwise correlations;
                        reduces exactly to CCA for two models.
``GeneralizedProcrustesAligner``
                        Is a *rigid rotation* enough? The strictest test, and
                        the cleanest evidence for "same space, different
                        coordinates".
``SharedAutoencoderAligner``
                        Is the shared structure nonlinear? If this substantially
                        beats GCCA, the linear methods were the limiting factor.
``OptimalTransportAligner``
                        Rigid alignment via optimal transport. Supervised mode
                        works well; the unsupervised mode (align *without*
                        patch correspondences) is exploratory — read its module
                        docstring before reporting anything from it.
======================  ===========================================================

>>> from utils.alignment import build_aligner, split_views
>>> train, test, _, _ = split_views(views, test_size=0.2)      # doctest: +SKIP
>>> aligner = build_aligner("gcca", latent_dim=64).fit(train)  # doctest: +SKIP
>>> Z = aligner.transform(test)                                # doctest: +SKIP
"""

from __future__ import annotations

from .autoencoder import SharedAutoencoderAligner
from .base import BaseAligner, ViewPreprocessor, split_views
from .gcca import GCCAAligner
from .gpa import GeneralizedProcrustesAligner
from .joint_pca import JointPCAAligner
from .mcca import MCCAAligner
from .optimal_transport import (
    OptimalTransportAligner,
    sinkhorn_coupling,
    sinkhorn_log,
)

__all__ = [
    "BaseAligner",
    "ViewPreprocessor",
    "split_views",
    "GCCAAligner",
    "MCCAAligner",
    "GeneralizedProcrustesAligner",
    "JointPCAAligner",
    "SharedAutoencoderAligner",
    "OptimalTransportAligner",
    "sinkhorn_log",
    "sinkhorn_coupling",
    "ALIGNER_REGISTRY",
    "available_aligners",
    "build_aligner",
]


#: Short name -> aligner class, for config-driven experiments.
ALIGNER_REGISTRY: dict[str, type[BaseAligner]] = {
    "joint_pca": JointPCAAligner,
    "gcca": GCCAAligner,
    "mcca": MCCAAligner,
    "procrustes": GeneralizedProcrustesAligner,
    "autoencoder": SharedAutoencoderAligner,
    "optimal_transport": OptimalTransportAligner,
}


def available_aligners() -> list[str]:
    """List the registered aligner names.

    Returns
    -------
    list of str
        Sorted names accepted by :func:`build_aligner`.
    """
    return sorted(ALIGNER_REGISTRY)


def build_aligner(name: str, **kwargs) -> BaseAligner:
    """Instantiate an aligner by name.

    Parameters
    ----------
    name : str
        One of :func:`available_aligners`.
    **kwargs
        Passed to the aligner's constructor (e.g. ``latent_dim=64``).

    Returns
    -------
    BaseAligner
        An unfitted aligner.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    try:
        cls = ALIGNER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown aligner {name!r}; available: {available_aligners()}"
        ) from None
    return cls(**kwargs)
