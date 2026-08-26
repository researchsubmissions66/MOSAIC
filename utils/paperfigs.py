"""Publication figures for MOSAIC, in the reference visual language.

Two archetypes, taken from the RecursiveMAS-style reference figures:

:func:`heatmap_row`
    A row of heatmap panels, each carrying its **own** colourbar, optionally
    masked to a staircase (lower triangle). The independent colourbars are the
    point: metrics sit at different levels, and a shared scale would flatten
    every panel but the widest-ranging one.

:func:`grouped_bars`
    Grouped bars per benchmark with the value printed **vertically inside**
    each bar, a light-to-dark fill ramp so the method being argued for is the
    darkest, and dashed separators between benchmark groups.

Both are pure matplotlib. That is deliberate: per-panel colourbars and rotated
in-bar labels are awkward under a grammar-of-graphics facet, which shares one
scale and one label geometry across panels.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

__all__ = [
    "PLOT_NAMES",
    "clean_label",
    "clean_labels",
    "PANEL_RAMPS",
    "BAR_RAMPS",
    "apply_style",
    "heatmap_row",
    "heatmap_panel",
    "grouped_bars",
    "trajectory_lines",
    "save_plot",
]

#: One sequential ramp per panel, cycled across a heatmap row so panels stay
#: visually distinguishable at a glance.
PANEL_RAMPS: dict[str, tuple[str, str]] = {
    "green": ("#ffffff", "#2f8b45"),
    "blue": ("#ffffff", "#1f6fb4"),
    "purple": ("#ffffff", "#5f3f9e"),
    "red": ("#ffffff", "#c0392b"),
    "teal": ("#ffffff", "#1f7a7a"),
    "orange": ("#ffffff", "#d2691e"),
    "magenta": ("#ffffff", "#a8327d"),
}
#: Must be at least as long as the widest panel row, or panels wrap onto a
#: colour already used and two different metrics read as the same series. Seven
#: entries covers the seven similarity metrics.
RAMP_CYCLE = ("green", "blue", "purple", "red", "teal", "orange", "magenta")

#: Light-to-dark categorical ramps for the bar panels. The last swatch is the
#: darkest, and the highlighted series is placed last so it lands there.
BAR_RAMPS: dict[str, list[str]] = {
    "blue": ["#d6e6f2", "#a9cce3", "#6aa5cd", "#2e6f9e"],
    "purple": ["#ded9ee", "#bcb3dc", "#9186c8", "#5b4a9f"],
    "red": ["#fbdcd2", "#f2a68c", "#e2725b", "#c0392b"],
    "green": ["#d8ecd4", "#a8d5a2", "#6cb26c", "#2f8b45"],
}


#: Registry key (and registry display name) -> the name as its authors write it.
#: Figures must not show internal keys like ``uni_v2`` or ``resnet50``; anything
#: not listed here falls back to underscores-to-hyphens in :func:`clean_label`.
PLOT_NAMES: dict[str, str] = {
    # encoders, keyed by both the registry key and its display_name so either
    # spelling resolves
    "uni_v2": "UNI2",
    "UNI2-h": "UNI2",
    "conch_v1": "CONCH",
    "CONCH": "CONCH",
    "conch_v15": "CONCH-v1.5",
    "CONCH v1.5": "CONCH-v1.5",
    "gigapath": "GigaPath",
    "Prov-GigaPath": "GigaPath",
    "virchow": "Virchow",
    "Virchow": "Virchow",
    "virchow2": "Virchow2",
    "Virchow2": "Virchow2",
    "hoptimus0": "H-optimus-0",
    "H-optimus-0": "H-optimus-0",
    "gpfm": "GPFM",
    # The display-name spellings matter as much as the keys: similarity CSVs
    # are written with display names (FeatureStore.display_names), so an
    # acronym missing here is title-cased into "Gpfm" / "Keep" / "Musk".
    "GPFM": "GPFM",
    "keep": "KEEP",
    "KEEP": "KEEP",
    "musk": "MUSK",
    "MUSK": "MUSK",
    "ctranspath": "CTransPath",
    "CTransPath": "CTransPath",
    "resnet50": "ResNet50",
    "ResNet50 (ImageNet)": "ResNet50",
    # conditions (identity entries so an already-clean label round-trips
    # unchanged rather than being title-cased into "Mosaic")
    "concat": "Concat",
    "shared": "MOSAIC",
    "Concat": "Concat",
    "MOSAIC": "MOSAIC",
    "single": "Single",
    # metric names used as axis labels
    "auc": "AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy": "Accuracy",
    "f1": "Macro F1",
    # metrics
    "linear_cka": "Linear CKA",
    "kernel_cka": "Kernel CKA",
    "svcca": "SVCCA",
    "pwcca": "PWCCA",
    "procrustes": "Procrustes",
    "cosine_rsa": "Cosine RSA",
    "distance_correlation": "Distance Correlation",
    "joint_pca": "Joint PCA",
    "gcca": "GCCA",
    "mcca": "MCCA",
    "autoencoder": "Autoencoder",
    "optimal_transport": "Optimal Transport",
    "unaligned_pca": "Unaligned (PCA)",
    "unaligned_truncate": "Unaligned (trunc.)",
    "abmil": "ABMIL",
    "transmil": "TransMIL",
    "mean": "Mean-pool",
    # retrieval metric columns
    "recall@1": "Recall@1",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "map": "mAP",
    "mrr": "MRR",
    "ndcg": "NDCG",
}

#: Gene symbols are always upper case, and appear inside task names.
_GENES = frozenset(
    {"TP53", "KRAS", "STK11", "PIK3CA", "GATA3", "MAP3K1", "KEAP1", "PIK3R1"}
)

#: Cohort tokens that keep their own casing inside task names.
_COHORTS = {
    "tcga": "TCGA",
    "cptac": "CPTAC",
    "brca": "BRCA",
    "luad": "LUAD",
    "lusc": "LUSC",
    "lscc": "LSCC",
    "coad": "COAD",
    "nsclc": "NSCLC",
    "idc": "IDC",
    "ilc": "ILC",
}


def clean_label(name: str) -> str:
    """Turn an internal identifier into the label an author would write.

    Underscores never survive into a figure. Known encoders, metrics and
    conditions map to their official spelling; anything else is split on
    underscores, with recognised cohort and gene tokens upper-cased and the
    rest title-cased.

    Parameters
    ----------
    name : str
        Registry key, display name, column name or task name.

    Returns
    -------
    str
        Display-ready label.

    Examples
    --------
    >>> clean_label("uni_v2")
    'UNI2'
    >>> clean_label("gigapath")
    'GigaPath'
    >>> clean_label("cptac_luad_tp53")
    'CPTAC LUAD TP53'
    >>> clean_label("some_new_model")
    'Some-New-Model'
    """
    if name in PLOT_NAMES:
        return PLOT_NAMES[name]

    tokens = str(name).split("_")
    if any(t.lower() in _COHORTS or t.upper() in _GENES for t in tokens):
        out = []
        for t in tokens:
            if t.upper() in _GENES:
                out.append(t.upper())
            elif t.lower() in _COHORTS:
                out.append(_COHORTS[t.lower()])
            else:
                out.append(t.capitalize())
        return " ".join(out)

    # Unknown identifier: never leave an underscore in a figure.
    return "-".join(t.capitalize() for t in tokens)


def clean_labels(names) -> list[str]:
    """Apply :func:`clean_label` across an iterable.

    Parameters
    ----------
    names : iterable of str

    Returns
    -------
    list of str
    """
    return [clean_label(n) for n in names]


def apply_style(base_size: float = 11.0) -> None:
    """Set the shared rcParams for every figure in this module.

    Parameters
    ----------
    base_size : float, default 11.0
        Base font size in points.
    """
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": base_size,
            "axes.titlesize": base_size + 1,
            "axes.labelsize": base_size,
            "axes.labelweight": "bold",
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _ramp(name: str):
    """Build a LinearSegmentedColormap from a named two-stop ramp."""
    low, high = PANEL_RAMPS.get(name, PANEL_RAMPS["green"])
    return mpl.colors.LinearSegmentedColormap.from_list(name, [low, high])


def heatmap_panel(
    ax,
    matrix: pd.DataFrame,
    title: str = "",
    ramp: str = "green",
    value_fmt: str = "{:.1f}",
    limits: tuple[float, float] | None = None,
    mask: str | None = None,
    xlab: str = "",
    ylab: str = "",
    label_size: float = 9.0,
    cbar_label: str = "",
    show_cbar: bool = True,
    n_cbar_ticks: int = 5,
):
    """Draw one heatmap panel with in-cell labels and its own colourbar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    matrix : pandas.DataFrame
        Matrix to display; row order is drawn bottom-to-top so the first row
        sits at the origin, matching the reference figures.
    title : str
        Panel title.
    ramp : str
        Key into :data:`PANEL_RAMPS`.
    value_fmt : str, default '{:.1f}'
        In-cell label format.
    limits : tuple of float, optional
        Colour limits; defaults to this panel's own min/max.
    mask : {'lower', 'upper'}, optional
        Blank one triangle to give the staircase look. ``'lower'`` keeps the
        lower-left triangle.
    xlab, ylab : str
        Axis labels.
    label_size : float, default 9.0
        In-cell label size.
    cbar_label : str
        Colourbar label.
    show_cbar : bool, default True
        Draw the colourbar.
    n_cbar_ticks : int, default 5
        Approximate number of colourbar ticks.

    Returns
    -------
    matplotlib.image.AxesImage
        The drawn image.
    """
    values = matrix.values.astype(float)
    n_rows, n_cols = values.shape

    shown = values.copy()
    if mask in ("lower", "upper"):
        idx = np.indices(values.shape)
        blank = idx[0] < idx[1] if mask == "lower" else idx[0] > idx[1]
        shown = np.where(blank, np.nan, shown)

    lo, hi = limits or (np.nanmin(shown), np.nanmax(shown))
    cmap = _ramp(ramp).copy()
    cmap.set_bad("white")

    im = ax.imshow(
        shown, cmap=cmap, vmin=lo, vmax=hi, origin="lower", aspect="equal"
    )

    # White cell separators, drawn as gridlines on the minor ticks.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    span = (hi - lo) or 1.0
    for i in range(n_rows):
        for j in range(n_cols):
            v = shown[i, j]
            if not np.isfinite(v):
                continue
            ax.text(
                j,
                i,
                value_fmt.format(v),
                ha="center",
                va="center",
                fontsize=label_size,
                fontweight="bold",
                color="white" if (v - lo) / span > 0.60 else "black",
            )

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(matrix.columns, rotation=0)
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title, fontweight="bold", pad=8)

    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.outline.set_linewidth(0.6)
        cbar.locator = mpl.ticker.MaxNLocator(nbins=n_cbar_ticks)
        cbar.update_ticks()
        if cbar_label:
            cbar.set_label(cbar_label, fontweight="bold")
    return im


def heatmap_row(
    matrices: Mapping[str, pd.DataFrame],
    value_fmt: str = "{:.1f}",
    mask: str | None = None,
    xlab: str = "",
    ylab: str = "",
    cbar_label: str = "",
    shared_limits: bool = False,
    panel_size: tuple[float, float] = (3.9, 4.1),
    label_size: float = 8.0,
    base_size: float = 11.0,
    suptitle: str | None = None,
    rotate_xticks: float = 0.0,
) -> Figure:
    """A row of heatmap panels, each with an independent colourbar.

    Parameters
    ----------
    matrices : mapping of str to pandas.DataFrame
        ``{panel_title: matrix}``, in display order.
    value_fmt : str, default '{:.1f}'
        In-cell label format.
    mask : {'lower', 'upper'}, optional
        Blank one triangle of every panel.
    xlab, ylab : str
        Axis labels; ``ylab`` is drawn on the leftmost panel only.
    cbar_label : str
        Colourbar label, drawn on the rightmost panel only.
    shared_limits : bool, default False
        Force one colour range across panels. Off by default — the reference
        gives each panel its own scale, which is what makes panels at different
        levels individually readable. Turn it on only when comparing levels
        *between* panels is the point.
    panel_size : tuple of float, default (3.9, 4.1)
        Size of each panel in inches. Taller than wide on purpose: the heatmap
        is drawn with equal aspect, so a wide-but-short panel collapses the
        plotting square and leaves the cells unreadably small.
    label_size : float, default 9.0
        In-cell label size.
    base_size : float, default 11.0
        Base font size.
    suptitle : str, optional
        Figure title.
    rotate_xticks : float, default 0.0
        Rotation for x tick labels; use 45 for long encoder names.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_style(base_size)
    names = list(matrices)
    if not names:
        raise ValueError("no matrices given")

    limits = None
    if shared_limits:
        limits = (
            min(float(np.nanmin(m.values)) for m in matrices.values()),
            max(float(np.nanmax(m.values)) for m in matrices.values()),
        )

    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(panel_size[0] * len(names), panel_size[1]),
        squeeze=False,
    )

    for i, (ax, name) in enumerate(zip(axes.ravel(), names)):
        heatmap_panel(
            ax,
            matrices[name],
            title=name,
            ramp=RAMP_CYCLE[i % len(RAMP_CYCLE)],
            value_fmt=value_fmt,
            limits=limits,
            mask=mask,
            xlab=xlab,
            ylab=ylab if i == 0 else "",
            label_size=label_size,
            cbar_label=cbar_label if i == len(names) - 1 else "",
        )
        if rotate_xticks:
            for lab in ax.get_xticklabels():
                lab.set_rotation(rotate_xticks)
                lab.set_ha("right")

    if suptitle:
        fig.suptitle(suptitle, fontsize=base_size + 3, fontweight="bold")
    fig.tight_layout()
    return fig


def grouped_bars(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    highlight: str | None = None,
    ramp: str = "blue",
    value_fmt: str = "{:.1f}",
    group_order: Sequence[str] | None = None,
    x_order: Sequence[str] | None = None,
    ylab: str = "Accuracy (%)",
    xlab: str = "",
    title: str = "",
    label_size: float = 8.0,
    base_size: float = 11.0,
    figsize: tuple[float, float] = (7.0, 4.6),
    inner_labels: bool = True,
    separators: bool = True,
    legend_ncol: int = 2,
    ax=None,
) -> tuple[Figure, plt.Axes]:
    """Grouped bars with values printed vertically inside each bar.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format results.
    x : str
        Column giving the benchmark on the x axis.
    y : str
        Column giving bar height.
    group : str
        Column giving the series (the fill).
    highlight : str, optional
        Series to render darkest and bold in the legend — the method being
        argued for. Moved to the end of the order so it takes the darkest
        swatch.
    ramp : str, default 'blue'
        Key into :data:`BAR_RAMPS`.
    value_fmt : str, default '{:.1f}'
        In-bar label format.
    group_order, x_order : sequence of str, optional
        Explicit ordering.
    ylab, xlab, title : str
        Labels.
    label_size : float, default 8.0
        In-bar label size.
    base_size : float, default 11.0
        Base font size.
    figsize : tuple of float, default (7.0, 4.6)
        Figure size.
    inner_labels : bool, default True
        Print values vertically inside the bars, as the reference does. Turn
        off for very short bars, where the label will not fit.
    separators : bool, default True
        Dashed vertical lines between benchmark groups.
    legend_ncol : int, default 2
        Legend columns.
    ax : matplotlib.axes.Axes, optional
        Draw into existing axes.

    Returns
    -------
    tuple
        ``(figure, axes)``.
    """
    apply_style(base_size)

    groups = list(group_order) if group_order else list(pd.unique(data[group]))
    if highlight and highlight in groups:
        groups = [g for g in groups if g != highlight] + [highlight]
    xs = list(x_order) if x_order else list(pd.unique(data[x]))

    colours = BAR_RAMPS.get(ramp, BAR_RAMPS["blue"])
    if len(groups) != len(colours):
        cmap = mpl.colors.LinearSegmentedColormap.from_list("r", colours)
        colours = [
            mpl.colors.to_hex(cmap(t)) for t in np.linspace(0.12, 1.0, len(groups))
        ]

    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))

    n_g = len(groups)
    width = 0.80 / n_g
    centres = np.arange(len(xs), dtype=float)

    lookup = data.set_index([x, group])[y].to_dict()
    top = 0.0

    for gi, g in enumerate(groups):
        offs = centres - 0.40 + width * (gi + 0.5)
        vals = [float(lookup.get((xv, g), np.nan)) for xv in xs]
        ax.bar(
            offs,
            vals,
            width=width * 0.94,
            color=colours[gi],
            edgecolor="none",
            label=g,
            zorder=3,
        )
        top = max(top, np.nanmax(vals) if np.isfinite(vals).any() else 0.0)

        if inner_labels:
            for xpos, v in zip(offs, vals):
                if not np.isfinite(v):
                    continue
                ax.text(
                    xpos,
                    v * 0.03 + 0.5,
                    value_fmt.format(v),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=label_size,
                    zorder=4,
                )

    if separators:
        for c in centres[:-1]:
            ax.axvline(c + 0.5, color="#333333", lw=0.9, ls="--", zorder=1)

    ax.set_xticks(centres)
    ax.set_xticklabels(xs)
    ax.set_xlim(-0.55, len(xs) - 0.45)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=base_size + 3, pad=10)

    handles = [Patch(facecolor=colours[i], label=g) for i, g in enumerate(groups)]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        ncol=legend_ncol,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.0,
        borderaxespad=0.3,
    )
    if highlight:
        for text in leg.get_texts():
            if text.get_text() == highlight:
                text.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def trajectory_lines(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    ylab: str = "",
    xlab: str = "",
    title: str = "",
    base_size: float = 11.0,
    figsize: tuple[float, float] = (7.0, 4.4),
    ax=None,
) -> tuple[Figure, plt.Axes]:
    """Line trajectories, for depth- or magnification-indexed curves.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format frame.
    x, y, group : str
        Columns for the axis, the value, and one line per level.
    ylab, xlab, title : str
        Labels.
    base_size : float, default 11.0
        Base font size.
    figsize : tuple of float, default (7.0, 4.4)
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    tuple
        ``(figure, axes)``.
    """
    apply_style(base_size)
    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))

    levels = list(pd.unique(data[group]))
    cmap = plt.get_cmap("Dark2")
    for i, lvl in enumerate(levels):
        sub = data[data[group] == lvl].sort_values(x)
        ax.plot(
            sub[x], sub[y], marker="o", markersize=4, lw=1.8,
            color=cmap(i % 8), label=str(lvl),
        )

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title, fontweight="bold", pad=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(levels) <= 14:
        ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    return fig, ax


def save_plot(fig: Figure, path, dpi: int = 300) -> None:
    """Save a figure, creating parent directories.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    path : str or pathlib.Path
        Output path; the extension selects the format (``.pdf`` for the paper).
    dpi : int, default 300
        Resolution for raster formats.
    """
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
