"""Visualisation of the model-by-model similarity structure.

Turns the N x N matrices produced by :mod:`utils.pairwise` into the three
Phase I deliverables:

1. a similarity heatmap (raw, or reordered by hierarchical clustering),
2. a dendrogram over models,
3. an ordination of the model space via MDS and/or UMAP.

Every plotting function accepts an optional ``ax`` and returns the matplotlib
objects, so the panels can be composed into multi-part figures. Nothing is
shown or styled globally — the caller keeps control of the rcParams.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from .pairwise import similarity_to_distance

__all__ = [
    "plot_similarity_heatmap",
    "hierarchical_linkage",
    "plot_dendrogram",
    "plot_clustered_heatmap",
    "cluster_assignments",
    "mds_embedding",
    "plot_mds",
    "umap_embedding",
    "plot_umap",
    "plot_model_space",
    "plot_metric_panel",
    "plot_magnification_trends",
    "plot_magnification_panel",
    "save_figure",
]


def _as_labelled_matrix(
    S: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Split a similarity/distance matrix into a raw array plus model labels."""
    if isinstance(S, pd.DataFrame):
        arr = S.values.astype(np.float64)
        names = [str(x) for x in S.index]
    else:
        arr = np.asarray(S, dtype=np.float64)
        names = (
            [str(x) for x in labels]
            if labels is not None
            else [f"model_{i}" for i in range(arr.shape[0])]
        )
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {arr.shape}")
    if len(names) != arr.shape[0]:
        raise ValueError(
            f"got {len(names)} labels for a {arr.shape[0]}x{arr.shape[0]} matrix"
        )
    return arr, names


def plot_similarity_heatmap(
    S: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    fmt: str = "{:.2f}",
    annot_fontsize: float | None = None,
    cbar_label: str = "similarity",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7.0, 6.0),
) -> tuple[Figure, plt.Axes]:
    """Draw an N x N similarity matrix as an annotated heatmap.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity matrix. A DataFrame supplies its own model labels.
    labels : sequence of str, optional
        Model names, required only when ``S`` is a bare array.
    title : str, optional
        Axes title.
    cmap : str, default 'viridis'
        Matplotlib colormap.
    vmin, vmax : float, optional
        Colour limits. Leaving these ``None`` autoscales, which exaggerates
        small differences — set them explicitly (e.g. 0 and 1) when comparing
        heatmaps across metrics or organs.
    annotate : bool, default True
        Write the numeric value in each cell. Text colour flips automatically
        against the cell brightness.
    fmt : str, default '{:.2f}'
        Format string for the annotations.
    annot_fontsize : float, optional
        Annotation font size. Defaults to a size scaled to the axes width and
        model count, so cells stay legible whether the heatmap is standalone or
        squeezed into a multi-panel figure.
    cbar_label : str, default 'similarity'
        Label for the colour bar.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes instead of creating a figure.
    figsize : tuple of float, default (7.0, 6.0)
        Figure size, used only when ``ax`` is None.

    Returns
    -------
    tuple
        ``(figure, axes)``.
    """
    arr, names = _as_labelled_matrix(S, labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    if title:
        ax.set_title(title)

    if annotate:
        if annot_fontsize is None:
            # The heatmap is square (aspect='equal'), so a cell is at most
            # min(width, height)/n. Size the text to fit ~4 characters in it.
            bbox = ax.get_position()
            side_pt = (
                min(bbox.width * fig.get_figwidth(), bbox.height * fig.get_figheight())
                * 72.0
            )
            annot_fontsize = float(
                np.clip(side_pt / max(len(names), 1) * 0.30, 4.0, 9.0)
            )

        lo = np.nanmin(arr) if vmin is None else vmin
        hi = np.nanmax(arr) if vmax is None else vmax
        span = (hi - lo) or 1.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if not np.isfinite(val):
                    continue
                shade = (val - lo) / span
                ax.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    color="white" if shade < 0.55 else "black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig, ax


def hierarchical_linkage(
    S: pd.DataFrame | np.ndarray,
    method: str = "average",
    distance_mode: str = "one_minus",
    is_distance: bool = False,
) -> np.ndarray:
    """Compute the hierarchical-clustering linkage over models.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity matrix (or distance matrix, with ``is_distance``).
    method : str, default 'average'
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`.
        ``'average'`` (UPGMA) is the safe default for a similarity-derived
        distance; ``'ward'`` assumes Euclidean distances and is not valid here.
    distance_mode : str, default 'one_minus'
        How to convert similarity to distance — see
        :func:`utils.pairwise.similarity_to_distance`.
    is_distance : bool, default False
        Set True if ``S`` already holds distances.

    Returns
    -------
    numpy.ndarray
        Linkage matrix of shape ``(n_models - 1, 4)``.
    """
    if method == "ward":
        warnings.warn(
            "'ward' linkage assumes Euclidean distances; on a similarity-"
            "derived distance matrix its output is not interpretable. "
            "Prefer 'average' or 'complete'.",
            RuntimeWarning,
            stacklevel=2,
        )

    D = S if is_distance else similarity_to_distance(S, mode=distance_mode)
    arr, _ = _as_labelled_matrix(D)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    return linkage(squareform(arr, checks=False), method=method, optimal_ordering=True)


def plot_dendrogram(
    S: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
    method: str = "average",
    distance_mode: str = "one_minus",
    is_distance: bool = False,
    color_threshold: float | None = None,
    orientation: str = "right",
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> tuple[Figure, plt.Axes, dict]:
    """Draw a dendrogram over the foundation models.

    Answers the "which models cluster together" question directly: if models
    group by pretraining objective the branches should split vision-only from
    vision-language, whereas grouping by architecture or training corpus points
    somewhere else.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity (or distance) matrix.
    labels : sequence of str, optional
        Model names, required only when ``S`` is a bare array.
    method : str, default 'average'
        Linkage method.
    distance_mode : str, default 'one_minus'
        Similarity-to-distance conversion.
    is_distance : bool, default False
        Set True if ``S`` already holds distances.
    color_threshold : float, optional
        Distance at which to start colouring separate clusters.
    orientation : str, default 'right'
        Dendrogram orientation; ``'right'`` keeps long model names readable.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes.
    figsize : tuple of float, default (7.0, 5.0)
        Figure size, used only when ``ax`` is None.

    Returns
    -------
    tuple
        ``(figure, axes, dendrogram_dict)``. The third element is scipy's
        dendrogram dict, whose ``'ivl'`` key gives the leaf order.
    """
    _, names = _as_labelled_matrix(S, labels)
    Z = hierarchical_linkage(
        S, method=method, distance_mode=distance_mode, is_distance=is_distance
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    dd = dendrogram(
        Z,
        labels=names,
        orientation=orientation,
        color_threshold=color_threshold,
        ax=ax,
    )
    ax.set_title(title or f"Model clustering ({method} linkage)")
    axis_label = "distance"
    if orientation in ("right", "left"):
        ax.set_xlabel(axis_label)
    else:
        ax.set_ylabel(axis_label)
    fig.tight_layout()
    return fig, ax, dd


def cluster_assignments(
    S: pd.DataFrame | np.ndarray,
    n_clusters: int = 2,
    labels: Sequence[str] | None = None,
    method: str = "average",
    distance_mode: str = "one_minus",
    is_distance: bool = False,
) -> pd.Series:
    """Cut the dendrogram into a flat clustering of models.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity (or distance) matrix.
    n_clusters : int, default 2
        Number of clusters to cut into.
    labels : sequence of str, optional
        Model names, required only when ``S`` is a bare array.
    method : str, default 'average'
        Linkage method.
    distance_mode : str, default 'one_minus'
        Similarity-to-distance conversion.
    is_distance : bool, default False
        Set True if ``S`` already holds distances.

    Returns
    -------
    pandas.Series
        Cluster id per model, indexed by model name.
    """
    _, names = _as_labelled_matrix(S, labels)
    Z = hierarchical_linkage(
        S, method=method, distance_mode=distance_mode, is_distance=is_distance
    )
    assign = fcluster(Z, t=n_clusters, criterion="maxclust")
    return pd.Series(assign, index=names, name="cluster")


def plot_clustered_heatmap(
    S: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
    method: str = "average",
    distance_mode: str = "one_minus",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7.5, 7.0),
):
    """Draw a heatmap with rows and columns reordered by hierarchical clustering.

    Combines the first two deliverables into the single figure that usually
    ends up in the paper: the block structure of the reordered matrix and the
    dendrogram that produced it.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity matrix.
    labels : sequence of str, optional
        Model names, required only when ``S`` is a bare array.
    method : str, default 'average'
        Linkage method.
    distance_mode : str, default 'one_minus'
        Similarity-to-distance conversion used for the linkage. Note the cells
        still show *similarity*.
    cmap : str, default 'viridis'
        Matplotlib colormap.
    vmin, vmax : float, optional
        Colour limits.
    title : str, optional
        Figure title.
    figsize : tuple of float, default (7.5, 7.0)
        Figure size.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        The clustergrid; ``.fig`` is the matplotlib figure and
        ``.dendrogram_row.reordered_ind`` gives the leaf order.
    """
    import seaborn as sns

    arr, names = _as_labelled_matrix(S, labels)
    df = pd.DataFrame(arr, index=names, columns=names)
    Z = hierarchical_linkage(df, method=method, distance_mode=distance_mode)

    grid = sns.clustermap(
        df,
        row_linkage=Z,
        col_linkage=Z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        figsize=figsize,
        cbar_kws={"label": "similarity"},
    )
    if title:
        grid.figure.suptitle(title, y=1.02)
    return grid


def _mds_kwargs(n_components: int, metric: bool, random_state: int, n_init: int) -> dict:
    """Build MDS keyword arguments compatible with old and new scikit-learn.

    scikit-learn 1.9 renamed the MDS parameters: the boolean ``metric`` became
    ``metric_mds``, and ``dissimilarity='precomputed'`` became
    ``metric='precomputed'``. This resolves the right spelling at runtime so
    the code works on either side of that change.
    """
    import inspect

    from sklearn.manifold import MDS

    params = inspect.signature(MDS.__init__).parameters
    kwargs: dict = {
        "n_components": n_components,
        "random_state": random_state,
        "n_init": n_init,
    }

    if "metric_mds" in params:  # scikit-learn >= 1.9
        kwargs["metric_mds"] = metric
        kwargs["metric"] = "precomputed"
    else:  # scikit-learn < 1.9
        kwargs["metric"] = metric
        kwargs["dissimilarity"] = "precomputed"

    if "init" in params:
        # 'classical_mds' is deterministic (and the future default), so extra
        # restarts would be redundant.
        kwargs["init"] = "classical_mds"
        kwargs["n_init"] = 1
    return kwargs


def mds_embedding(
    S: pd.DataFrame | np.ndarray,
    n_components: int = 2,
    distance_mode: str = "one_minus",
    is_distance: bool = False,
    metric: bool = True,
    random_state: int = 0,
    n_init: int = 8,
) -> tuple[np.ndarray, float]:
    """Embed models into a low-dimensional space with MDS.

    MDS is the right ordination for N ~ 10 models: it works directly on the
    precomputed distance matrix and, unlike UMAP, has no neighbourhood
    parameters that are ill-defined at this sample size.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity (or distance) matrix.
    n_components : int, default 2
        Output dimensionality.
    distance_mode : str, default 'one_minus'
        Similarity-to-distance conversion.
    is_distance : bool, default False
        Set True if ``S`` already holds distances.
    metric : bool, default True
        Metric MDS (preserves distances) vs non-metric MDS (preserves only
        their ordering).
    random_state : int, default 0
        Seed for the SMACOF initialisation.
    n_init : int, default 8
        Number of restarts; MDS is non-convex, so more restarts give a more
        stable layout. Ignored on scikit-learn versions that support the
        deterministic ``'classical_mds'`` initialisation, which is used
        instead.

    Returns
    -------
    tuple
        ``(coordinates, stress)`` where coordinates has shape
        ``(n_models, n_components)`` and stress is Stress-1 (normalised, so
        comparable across metrics) when the installed scikit-learn supports
        it, otherwise raw stress. Values above ~0.2 mean the 2-D picture is
        not a faithful summary of the distance matrix — report it alongside
        the plot rather than reading structure into the layout.
    """
    from sklearn.manifold import MDS

    D = S if is_distance else similarity_to_distance(S, mode=distance_mode)
    arr, _ = _as_labelled_matrix(D)

    kwargs = _mds_kwargs(n_components, metric, random_state, n_init)
    try:
        mds = MDS(normalized_stress=True, **kwargs)
        coords = mds.fit_transform(arr)
    except (TypeError, ValueError):
        # Older scikit-learn rejects normalized_stress for metric MDS.
        mds = MDS(**kwargs)
        coords = mds.fit_transform(arr)

    return coords, float(mds.stress_)


def _scatter_models(
    coords: np.ndarray,
    names: Sequence[str],
    ax: plt.Axes,
    point_size: float = 90.0,
    text_offset: float = 0.015,
) -> None:
    """Scatter model points with non-overlapping text labels."""
    ax.scatter(coords[:, 0], coords[:, 1], s=point_size, zorder=3)
    span = np.ptp(coords, axis=0)
    dx = (span[0] or 1.0) * text_offset
    dy = (span[1] or 1.0) * text_offset
    for (x, y), name in zip(coords, names):
        ax.annotate(name, (x + dx, y + dy), fontsize=9, zorder=4)
    ax.axhline(0, lw=0.5, color="0.85", zorder=0)
    ax.axvline(0, lw=0.5, color="0.85", zorder=0)


def plot_mds(
    S: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
    distance_mode: str = "one_minus",
    is_distance: bool = False,
    metric: bool = True,
    random_state: int = 0,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (6.5, 5.5),
) -> tuple[Figure, plt.Axes, np.ndarray]:
    """Plot the model space in 2-D via MDS.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity (or distance) matrix.
    labels : sequence of str, optional
        Model names, required only when ``S`` is a bare array.
    distance_mode : str, default 'one_minus'
        Similarity-to-distance conversion.
    is_distance : bool, default False
        Set True if ``S`` already holds distances.
    metric : bool, default True
        Metric vs non-metric MDS.
    random_state : int, default 0
        Seed.
    title : str, optional
        Axes title. Defaults to one including the stress value.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes.
    figsize : tuple of float, default (6.5, 5.5)
        Figure size, used only when ``ax`` is None.

    Returns
    -------
    tuple
        ``(figure, axes, coordinates)``.
    """
    _, names = _as_labelled_matrix(S, labels)
    coords, stress = mds_embedding(
        S,
        distance_mode=distance_mode,
        is_distance=is_distance,
        metric=metric,
        random_state=random_state,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _scatter_models(coords, names, ax)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title or f"Model space (MDS, stress={stress:.3f})")
    fig.tight_layout()
    return fig, ax, coords


def umap_embedding(
    S: pd.DataFrame | np.ndarray,
    n_components: int = 2,
    n_neighbors: int | None = None,
    min_dist: float = 0.1,
    distance_mode: str = "one_minus",
    is_distance: bool = False,
    random_state: int = 0,
) -> np.ndarray:
    """Embed models into a low-dimensional space with UMAP.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity (or distance) matrix.
    n_components : int, default 2
        Output dimensionality.
    n_neighbors : int, optional
        Neighbourhood size. Defaults to ``n_models - 1`` (i.e. all other
        models), which is the only defensible choice at this sample size.
    min_dist : float, default 0.1
        UMAP's minimum-distance parameter.
    distance_mode : str, default 'one_minus'
        Similarity-to-distance conversion.
    is_distance : bool, default False
        Set True if ``S`` already holds distances.
    random_state : int, default 0
        Seed.

    Returns
    -------
    numpy.ndarray
        Coordinates of shape ``(n_models, n_components)``.

    Warns
    -----
    RuntimeWarning
        When fewer than 15 models are supplied. UMAP's manifold assumptions do
        not hold for a handful of points and the layout will be dominated by
        initialisation — with ~8 foundation models, **MDS is the metric to
        trust and UMAP is at best a cross-check**. This function exists for
        completeness and for the patch-level embeddings of later phases, where
        UMAP is genuinely the right tool.
    """
    import umap

    D = S if is_distance else similarity_to_distance(S, mode=distance_mode)
    arr, _ = _as_labelled_matrix(D)
    n = arr.shape[0]

    if n < 4:
        raise ValueError(f"UMAP needs at least 4 points, got {n}")
    if n < 15:
        warnings.warn(
            f"running UMAP on only {n} points; the layout is not "
            "statistically meaningful — prefer plot_mds for the model space.",
            RuntimeWarning,
            stacklevel=2,
        )

    k = n - 1 if n_neighbors is None else min(n_neighbors, n - 1)
    reducer = umap.UMAP(
        n_components=min(n_components, n - 2),
        n_neighbors=max(k, 2),
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
    )
    return reducer.fit_transform(arr)


def plot_umap(
    S: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
    n_neighbors: int | None = None,
    min_dist: float = 0.1,
    distance_mode: str = "one_minus",
    is_distance: bool = False,
    random_state: int = 0,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (6.5, 5.5),
) -> tuple[Figure, plt.Axes, np.ndarray]:
    """Plot the model space in 2-D via UMAP.

    See :func:`umap_embedding` for the caveat about small point counts.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity (or distance) matrix.
    labels : sequence of str, optional
        Model names, required only when ``S`` is a bare array.
    n_neighbors : int, optional
        Neighbourhood size; defaults to ``n_models - 1``.
    min_dist : float, default 0.1
        UMAP's minimum-distance parameter.
    distance_mode : str, default 'one_minus'
        Similarity-to-distance conversion.
    is_distance : bool, default False
        Set True if ``S`` already holds distances.
    random_state : int, default 0
        Seed.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes.
    figsize : tuple of float, default (6.5, 5.5)
        Figure size, used only when ``ax`` is None.

    Returns
    -------
    tuple
        ``(figure, axes, coordinates)``.
    """
    _, names = _as_labelled_matrix(S, labels)
    coords = umap_embedding(
        S,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        distance_mode=distance_mode,
        is_distance=is_distance,
        random_state=random_state,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _scatter_models(np.asarray(coords), names, ax)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or "Model space (UMAP)")
    fig.tight_layout()
    return fig, ax, np.asarray(coords)


def plot_model_space(
    S: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
    distance_mode: str = "one_minus",
    include_umap: bool = True,
    random_state: int = 0,
    suptitle: str | None = None,
    figsize: tuple[float, float] = (15.0, 4.6),
) -> Figure:
    """Draw the full Phase I summary figure for one metric.

    Lays out heatmap, dendrogram and MDS (optionally plus UMAP) as one row.

    Parameters
    ----------
    S : pandas.DataFrame or numpy.ndarray
        Square similarity matrix.
    labels : sequence of str, optional
        Model names, required only when ``S`` is a bare array.
    distance_mode : str, default 'one_minus'
        Similarity-to-distance conversion for clustering and ordination.
    include_umap : bool, default True
        Add a fourth UMAP panel. Set False for a cleaner three-panel figure
        (recommended when the model count is small).
    random_state : int, default 0
        Seed for MDS and UMAP.
    suptitle : str, optional
        Figure title, e.g. the metric name.
    figsize : tuple of float, default (15.0, 4.6)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure.
    """
    n_panels = 4 if include_umap else 3
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    plot_similarity_heatmap(S, labels, title="Similarity", ax=axes[0])
    plot_dendrogram(S, labels, distance_mode=distance_mode, title="Clustering", ax=axes[1])
    plot_mds(
        S, labels, distance_mode=distance_mode, random_state=random_state, ax=axes[2]
    )
    if include_umap:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            plot_umap(
                S,
                labels,
                distance_mode=distance_mode,
                random_state=random_state,
                ax=axes[3],
            )

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_metric_panel(
    matrices: Mapping[str, pd.DataFrame],
    n_cols: int = 4,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    suptitle: str | None = "Representational similarity across metrics",
    figsize_per_panel: tuple[float, float] = (4.2, 3.8),
) -> Figure:
    """Draw one heatmap per metric on a shared colour scale.

    The shared ``vmin``/``vmax`` is the point: it makes the *level* differences
    between metrics visible, not just the pattern within each.

    Parameters
    ----------
    matrices : mapping of str to pandas.DataFrame
        ``{metric_name: similarity_matrix}``, as returned by
        :func:`utils.pairwise.compute_all_similarity_matrices`.
    n_cols : int, default 4
        Panels per row.
    cmap : str, default 'viridis'
        Matplotlib colormap.
    vmin, vmax : float, optional
        Shared colour limits. Defaults to ``[0, 1]``; pass ``None`` to
        autoscale each panel independently.
    suptitle : str, optional
        Figure title.
    figsize_per_panel : tuple of float, default (4.2, 3.8)
        Size of each panel.

    Returns
    -------
    matplotlib.figure.Figure
        The composed figure.
    """
    if not matrices:
        raise ValueError("no matrices given")

    names = list(matrices)
    n = len(names)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    flat = axes.ravel()

    for ax, metric in zip(flat, names):
        plot_similarity_heatmap(
            matrices[metric],
            title=metric,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
    for ax in flat[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_magnification_trends(
    trends: pd.DataFrame,
    metric: str = "linear_cka",
    highlight: Sequence[str] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> tuple[Figure, plt.Axes]:
    """Plot each model pair's similarity against magnification.

    One line per model pair. Lines that stay flat mean the pair's relationship
    is magnification-independent; lines that cross mean the answer to "which
    models are most alike" depends on the magnification you looked at.

    Parameters
    ----------
    trends : pandas.DataFrame
        Long-format output of :func:`utils.ablation.similarity_trends`.
    metric : str, default 'linear_cka'
        Metric to plot.
    highlight : sequence of str, optional
        Pair labels (``'a~b'``) to draw in colour with everything else greyed
        out — useful for pointing at the control model's pairs.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes.
    figsize : tuple of float, default (7.0, 5.0)
        Figure size, used only when ``ax`` is None.

    Returns
    -------
    tuple
        ``(figure, axes)``.
    """
    sub = trends[trends["metric"] == metric]
    if sub.empty:
        raise ValueError(f"no rows for metric {metric!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    mags = sorted(sub["magnification"].unique())
    for pair, grp in sub.groupby("pair"):
        grp = grp.sort_values("magnification")
        muted = highlight is not None and pair not in highlight
        ax.plot(
            grp["magnification"],
            grp["similarity"],
            marker="o",
            markersize=4,
            linewidth=2.0 if not muted else 1.0,
            color="0.8" if muted else None,
            label=None if muted else pair,
            zorder=1 if muted else 3,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(mags)
    ax.set_xticklabels([f"{m:g}x" for m in mags])
    ax.minorticks_off()
    ax.set_xlabel("magnification")
    ax.set_ylabel(f"{metric} similarity")
    ax.set_title(title or f"{metric} across magnification")
    ax.grid(alpha=0.25, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles and len(handles) <= 16:
        ax.legend(fontsize=7, ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    return fig, ax


def plot_magnification_panel(
    results: Mapping[float, Mapping[str, pd.DataFrame]],
    metric: str = "linear_cka",
    labels: Sequence[str] | None = None,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    cmap: str = "viridis",
    suptitle: str | None = None,
    figsize_per_panel: tuple[float, float] = (4.2, 3.8),
) -> Figure:
    """Show one similarity heatmap per magnification on a shared colour scale.

    The shared scale is the point — it makes level shifts between
    magnifications visible rather than normalising them away.

    Parameters
    ----------
    results : mapping
        ``{magnification: {metric: similarity_matrix}}``.
    metric : str, default 'linear_cka'
        Metric to display.
    labels : sequence of str, optional
        Model display names, if the matrices are not already labelled.
    vmin, vmax : float, optional
        Shared colour limits. Defaults to ``[0, 1]``.
    cmap : str, default 'viridis'
        Matplotlib colormap.
    suptitle : str, optional
        Figure title.
    figsize_per_panel : tuple of float, default (4.2, 3.8)
        Size of each panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mags = sorted(results)
    missing = [m for m in mags if metric not in results[m]]
    if missing:
        raise KeyError(f"metric {metric!r} missing at magnifications {missing}")

    fig, axes = plt.subplots(
        1,
        len(mags),
        figsize=(figsize_per_panel[0] * len(mags), figsize_per_panel[1]),
        squeeze=False,
    )
    for ax, mag in zip(axes.ravel(), mags):
        plot_similarity_heatmap(
            results[mag][metric],
            labels=labels,
            title=f"{mag:g}x",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
    fig.suptitle(suptitle or f"{metric} across magnification")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path, dpi: int = 300, transparent: bool = False) -> None:
    """Save a figure, creating parent directories as needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or pathlib.Path
        Output path; the extension selects the format (``.pdf`` for the paper,
        ``.png`` for quick looks).
    dpi : int, default 300
        Resolution for raster formats.
    transparent : bool, default False
        Transparent background.
    """
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent)
