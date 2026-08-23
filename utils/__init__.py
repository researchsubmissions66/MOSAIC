"""Utilities for the shared-latent-space study of pathology foundation models.

Phase I (representational similarity) is implemented here:

``preprocessing``
    Coercion, pairing checks, centering, joint subsampling.
``cka``
    Linear and RBF-kernel Centered Kernel Alignment.
``cca``
    SVCCA and PWCCA (plus the raw CCA decomposition they build on).
``procrustes``
    Orthogonal Procrustes similarity/distance, and the fitted transform used
    as the baseline aligner in later phases.
``cosine``
    Cosine-RSM representational similarity analysis.
``distance_correlation``
    Szekely's distance correlation, biased and unbiased.
``pairwise``
    Metric registry and the N x N similarity-matrix driver.
``visualization``
    Heatmaps, dendrograms, MDS and UMAP of the model space.

All metrics take two **row-paired** matrices — the same patches encoded by two
different models — with possibly different feature dimensions, and return a
scalar similarity.

>>> from utils import compute_all_similarity_matrices, plot_model_space
>>> mats = compute_all_similarity_matrices(reps, max_samples=5000)  # doctest: +SKIP
>>> fig = plot_model_space(mats["linear_cka"], suptitle="Linear CKA")  # doctest: +SKIP
"""

from __future__ import annotations

from .cca import (
    cca_correlations,
    cca_decomposition,
    mean_cca_correlation,
    pwcca,
    svcca,
)
from .cka import (
    center_gram,
    cka_from_grams,
    gram_linear,
    gram_rbf,
    hsic,
    kernel_cka,
    linear_cka,
)
from .cosine import (
    cosine_rsa_similarity,
    cosine_rsm,
    l2_normalize,
    mean_cosine_similarity,
    rsa_similarity,
)
from .distance_correlation import (
    distance_correlation,
    distance_covariance,
    pairwise_distance_matrix,
)
from .features import (
    EncoderInfo,
    FeatureGroup,
    FeatureStore,
    load_encoder_config,
)
from .pairwise import (
    METRIC_REGISTRY,
    QUADRATIC_METRICS,
    available_metrics,
    compute_all_similarity_matrices,
    compute_similarity,
    compute_similarity_matrix,
    similarity_to_distance,
    stack_similarity_matrices,
)
from .preprocessing import (
    as_matrix,
    center_columns,
    normalize_frobenius,
    prepare_pair,
    subsample_indices,
    zscore_columns,
)
from .procrustes import (
    apply_procrustes_transform,
    fit_procrustes_transform,
    orthogonal_procrustes_distance,
    orthogonal_procrustes_similarity,
)
from .visualization import (
    cluster_assignments,
    hierarchical_linkage,
    mds_embedding,
    plot_clustered_heatmap,
    plot_dendrogram,
    plot_mds,
    plot_metric_panel,
    plot_model_space,
    plot_similarity_heatmap,
    plot_umap,
    save_figure,
    umap_embedding,
)

__version__ = "0.1.0"

__all__ = [
    # preprocessing
    "as_matrix",
    "center_columns",
    "normalize_frobenius",
    "prepare_pair",
    "subsample_indices",
    "zscore_columns",
    # cka
    "linear_cka",
    "kernel_cka",
    "cka_from_grams",
    "gram_linear",
    "gram_rbf",
    "center_gram",
    "hsic",
    # cca
    "svcca",
    "pwcca",
    "cca_correlations",
    "cca_decomposition",
    "mean_cca_correlation",
    # procrustes
    "orthogonal_procrustes_similarity",
    "orthogonal_procrustes_distance",
    "fit_procrustes_transform",
    "apply_procrustes_transform",
    # cosine
    "cosine_rsa_similarity",
    "cosine_rsm",
    "rsa_similarity",
    "mean_cosine_similarity",
    "l2_normalize",
    # distance correlation
    "distance_correlation",
    "distance_covariance",
    "pairwise_distance_matrix",
    # feature store
    "FeatureStore",
    "FeatureGroup",
    "EncoderInfo",
    "load_encoder_config",
    # pairwise driver
    "METRIC_REGISTRY",
    "QUADRATIC_METRICS",
    "available_metrics",
    "compute_similarity",
    "compute_similarity_matrix",
    "compute_all_similarity_matrices",
    "similarity_to_distance",
    "stack_similarity_matrices",
    # visualization
    "plot_similarity_heatmap",
    "plot_clustered_heatmap",
    "plot_dendrogram",
    "hierarchical_linkage",
    "cluster_assignments",
    "plot_mds",
    "mds_embedding",
    "plot_umap",
    "umap_embedding",
    "plot_model_space",
    "plot_metric_panel",
    "save_figure",
]
