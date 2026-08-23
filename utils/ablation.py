"""Repeating the Phase I and Phase V experiments across magnifications.

Magnification is the first ablation axis: the same models, the same slides, the
same metrics, run independently at 5x, 10x and 20x. It asks whether
representational agreement is a property of the models or of the scale they are
looked at — a 5x patch shows tissue architecture, a 20x patch shows nuclei, and
there is no reason to assume models agree equally about both.

Each magnification is a *separate replication*, not a paired comparison: the
coordinate grids differ, so there is no correspondence between a 5x patch and a
20x patch. What gets compared across magnifications is the **result** — the
N x N similarity matrix, or the alignment quality — never the embeddings.

Two questions this can answer, which need different summaries:

* *Does agreement get stronger or weaker with resolution?*
  :func:`magnification_summary` — the level of the similarity matrix.
* *Does the structure change, i.e. do different models become the close pair?*
  :func:`magnification_stability` — the ranking of model pairs, which can be
  stable even when the level shifts a lot, or shift even when the level does not.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .pairwise import compute_all_similarity_matrices

__all__ = [
    "run_similarity_ablation",
    "run_alignment_ablation",
    "similarity_trends",
    "magnification_summary",
    "magnification_stability",
    "rank_shift_report",
]


def run_similarity_ablation(
    series,
    metrics: Sequence[str] | None = None,
    n_patches: int = 20_000,
    max_slides: int | None = 200,
    max_samples: int | None = 5_000,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[dict[float, dict[str, pd.DataFrame]], dict[float, dict[str, np.ndarray]]]:
    """Run the full Phase I metric suite at every magnification in a series.

    Parameters
    ----------
    series : MagnificationSeries
        From :meth:`utils.features.FeatureStore.magnification_series`. Its
        encoder set is already intersected across magnifications, so the
        comparison isolates magnification.
    metrics : sequence of str, optional
        Metrics to run. Defaults to the seven Phase I metrics.
    n_patches : int, default 20000
        Patches sampled per magnification.
    max_slides : int or None, default 200
        Slides read per magnification (the same slides at each).
    max_samples : int or None, default 5000
        Subsample passed to the O(n^2) metrics.
    seed : int, default 0
        Shared seed. Using one seed across magnifications means the same slides
        are selected everywhere, so slide composition cannot explain a
        difference between magnifications.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    tuple
        ``(results, views)`` where ``results`` is
        ``{magnification: {metric: similarity_matrix}}`` and ``views`` is
        ``{magnification: {encoder: matrix}}`` — the latter returned so a
        caller can reuse the loaded features for Phase V without re-reading
        hundreds of gigabytes.
    """
    if verbose:
        print(f"Sampling {series.key} at {[f'{m:g}x' for m in series.magnifications]}")

    views = series.sample(
        n_patches=n_patches, max_slides=max_slides, seed=seed, verbose=verbose
    )

    results: dict[float, dict[str, pd.DataFrame]] = {}
    for mag in series.magnifications:
        if verbose:
            print(f"\n=== {mag:g}x ===")
            t0 = time.perf_counter()
        results[mag] = compute_all_similarity_matrices(
            views[mag], metrics=metrics, max_samples=max_samples, seed=seed
        )
        if verbose:
            print(f"  {time.perf_counter() - t0:.1f}s")
    return results, views


def run_alignment_ablation(
    views: Mapping[float, Mapping[str, np.ndarray]],
    methods: Sequence[str] = ("joint_pca", "gcca", "procrustes"),
    latent_dim: int = 64,
    test_size: float = 0.2,
    seed: int = 0,
    verbose: bool = True,
    **evaluate_kwargs,
) -> pd.DataFrame:
    """Fit and evaluate shared latent spaces at each magnification.

    Parameters
    ----------
    views : mapping of float to mapping
        ``{magnification: {encoder: matrix}}``, as returned by
        :func:`run_similarity_ablation`.
    methods : sequence of str, default ('joint_pca', 'gcca', 'procrustes')
        Aligners to run. The three linear ones are the default because they fit
        in seconds; add ``'autoencoder'`` for the nonlinear comparison.
    latent_dim : int, default 64
        Shared space dimensionality, held constant across magnifications.
    test_size : float, default 0.2
        Held-out fraction.
    seed : int, default 0
        Seed for the split and the aligners.
    verbose : bool, default True
        Print progress.
    **evaluate_kwargs
        Forwarded to :func:`utils.alignment_metrics.evaluate_aligner`.

    Returns
    -------
    pandas.DataFrame
        One row per ``(magnification, method)`` with the full evaluation
        summary, indexed by both.
    """
    from .alignment import build_aligner, split_views
    from .alignment_metrics import evaluate_aligner

    rows = []
    for mag in sorted(views):
        train, test, _, _ = split_views(views[mag], test_size=test_size, seed=seed)
        for method in methods:
            if verbose:
                print(f"  {mag:g}x {method} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            aligner = build_aligner(
                method, latent_dim=latent_dim, random_state=seed
            ).fit(train)
            summary = evaluate_aligner(aligner, test, **evaluate_kwargs)["summary"]
            if verbose:
                print(f"{time.perf_counter() - t0:.1f}s")
            rows.append(
                {"magnification": mag, "method": method, **summary.to_dict()}
            )
    return pd.DataFrame(rows).set_index(["magnification", "method"])


def similarity_trends(
    results: Mapping[float, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Flatten the ablation into one row per (model pair, magnification, metric).

    Parameters
    ----------
    results : mapping
        ``{magnification: {metric: similarity_matrix}}``.

    Returns
    -------
    pandas.DataFrame
        Columns ``magnification``, ``metric``, ``model_a``, ``model_b``,
        ``similarity``. Long format, ready for plotting or grouping.
    """
    rows = []
    for mag, mats in sorted(results.items()):
        for metric, S in mats.items():
            names = list(S.index)
            for a, b in combinations(names, 2):
                rows.append(
                    {
                        "magnification": mag,
                        "metric": metric,
                        "model_a": a,
                        "model_b": b,
                        "pair": f"{a}~{b}",
                        "similarity": float(S.loc[a, b]),
                    }
                )
    return pd.DataFrame(rows)


def magnification_summary(
    results: Mapping[float, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Mean off-diagonal similarity per magnification and metric.

    Answers the *level* question: do models agree more at low magnification
    (where they all see the same coarse architecture) or at high magnification
    (where fine nuclear detail might separate them)?

    Parameters
    ----------
    results : mapping
        ``{magnification: {metric: similarity_matrix}}``.

    Returns
    -------
    pandas.DataFrame
        Magnifications as rows, metrics as columns, mean off-diagonal
        similarity as values, plus a ``spread`` column giving the standard
        deviation across model pairs (a metric can keep its mean while the
        models fan out).
    """
    long = similarity_trends(results)
    table = long.pivot_table(
        index="magnification", columns="metric", values="similarity", aggfunc="mean"
    )
    spread = long.groupby("magnification")["similarity"].std()
    table["spread"] = spread
    return table


def magnification_stability(
    results: Mapping[float, Mapping[str, pd.DataFrame]],
    metric: str = "linear_cka",
    correlation: str = "spearman",
) -> pd.DataFrame:
    """Agreement between the similarity matrices at different magnifications.

    Answers the *structure* question: even if agreement is higher at 20x, is it
    the same models that are close to each other? A high value means
    magnification rescales everything uniformly; a low value means the
    conclusion about which models resemble which actually depends on the
    magnification you chose — which would be a finding in its own right, and a
    caveat on every single-magnification result in the literature.

    Parameters
    ----------
    results : mapping
        ``{magnification: {metric: similarity_matrix}}``.
    metric : str, default 'linear_cka'
        Which metric's matrices to compare.
    correlation : {'spearman', 'pearson'}, default 'spearman'
        Correlation over the model pairs. Spearman asks only whether the
        *ordering* of pairs is preserved.

    Returns
    -------
    pandas.DataFrame
        Symmetric magnification-by-magnification correlation matrix.
    """
    mags = sorted(results)
    missing = [m for m in mags if metric not in results[m]]
    if missing:
        raise KeyError(f"metric {metric!r} missing at magnifications {missing}")

    vecs = {}
    for m in mags:
        S = results[m][metric]
        iu = np.triu_indices(S.shape[0], k=1)
        vecs[m] = S.values[iu]

    fn = stats.spearmanr if correlation == "spearman" else stats.pearsonr
    out = np.eye(len(mags))
    for i, a in enumerate(mags):
        for j in range(i + 1, len(mags)):
            b = mags[j]
            out[i, j] = out[j, i] = float(fn(vecs[a], vecs[b])[0])

    labels = [f"{m:g}x" for m in mags]
    return pd.DataFrame(out, index=labels, columns=labels)


def rank_shift_report(
    results: Mapping[float, Mapping[str, pd.DataFrame]],
    metric: str = "linear_cka",
) -> pd.DataFrame:
    """Per-pair similarity across magnifications, with the change highlighted.

    The drill-down behind :func:`magnification_stability`: which specific model
    pairs are responsible for the structure changing.

    Parameters
    ----------
    results : mapping
        ``{magnification: {metric: similarity_matrix}}``.
    metric : str, default 'linear_cka'
        Metric to report.

    Returns
    -------
    pandas.DataFrame
        One row per model pair, one column per magnification, plus:

        ``delta``
            Highest-magnification value minus lowest.
        ``range``
            Max minus min across magnifications.
        ``rank_change``
            How far the pair moves in the ranking of pairs between the lowest
            and highest magnification. Non-zero values are what
            :func:`magnification_stability` is detecting.

        Sorted by ``range`` descending — the most magnification-sensitive
        pairs first.
    """
    long = similarity_trends(results)
    sub = long[long["metric"] == metric]
    table = sub.pivot_table(index="pair", columns="magnification", values="similarity")

    mags = sorted(table.columns)
    lo, hi = mags[0], mags[-1]
    table.columns = [f"{m:g}x" for m in mags]
    lo_c, hi_c = f"{lo:g}x", f"{hi:g}x"

    table["delta"] = table[hi_c] - table[lo_c]
    table["range"] = table[[f"{m:g}x" for m in mags]].max(axis=1) - table[
        [f"{m:g}x" for m in mags]
    ].min(axis=1)
    table["rank_change"] = table[lo_c].rank(ascending=False).astype(int) - table[
        hi_c
    ].rank(ascending=False).astype(int)

    return table.sort_values("range", ascending=False)
