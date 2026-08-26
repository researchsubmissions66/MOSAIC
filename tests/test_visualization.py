"""Composition rules for the multi-panel model-space figure.

The claim this module earns is narrow but was worth a production failure:
:func:`plot_model_space` composes four panels, only one of which has a lower
bound on the model count, and it must not let that one panel fail the figure.
UMAP needs a neighbourhood graph and so needs four points; the heatmap,
dendrogram and MDS are all well defined down to two. A two-encoder group is a
real case here -- CONCH against CONCH v1.5 is a whole similarity run -- so the
small-n path is exercised, not hypothetical.

Run with::

    python -m pytest tests/test_visualization.py -q
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.visualization import (  # noqa: E402
    UMAP_MIN_POINTS,
    plot_model_space,
    umap_embedding,
)


def n_panels(fig) -> int:
    """Count real panels, ignoring the colourbar axes the heatmap attaches."""
    return sum(1 for ax in fig.axes if ax.get_label() != "<colorbar>")


def similarity(n: int, seed: int = 0) -> pd.DataFrame:
    """A valid similarity matrix over ``n`` models: symmetric, unit diagonal."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.3, 0.9, size=(n, n))
    S = (A + A.T) / 2.0
    np.fill_diagonal(S, 1.0)
    names = [f"model_{i}" for i in range(n)]
    return pd.DataFrame(S, index=names, columns=names)


@pytest.mark.parametrize("n", range(2, UMAP_MIN_POINTS))
def test_umap_panel_dropped_not_raised(n):
    """Too few models drops the panel and warns; it must not raise."""
    with pytest.warns(RuntimeWarning, match="skipping the UMAP panel"):
        fig = plot_model_space(similarity(n), include_umap=True)
    assert n_panels(fig) == 3


@pytest.mark.parametrize("n", range(2, UMAP_MIN_POINTS))
def test_umap_embedding_still_refuses(n):
    """The degradation belongs to the figure, not to the embedding itself."""
    with pytest.raises(ValueError, match=f"at least {UMAP_MIN_POINTS} points"):
        umap_embedding(similarity(n))


def test_explicit_opt_out_is_silent():
    """Declining the panel is not the same as having it taken away: no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_model_space(similarity(2), include_umap=False)
    assert n_panels(fig) == 3
    assert not [w for w in caught if "skipping the UMAP panel" in str(w.message)]


@pytest.mark.slow
def test_panel_count_unaffected_above_threshold():
    """At or above the threshold the request is honoured -- four panels."""
    pytest.importorskip("umap")
    fig = plot_model_space(similarity(UMAP_MIN_POINTS), include_umap=True)
    assert n_panels(fig) == 4
