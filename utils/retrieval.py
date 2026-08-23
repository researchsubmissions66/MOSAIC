"""Phase VII: cross-model retrieval.

Index a database with model A, query it with model B, and measure whether the
right things come back. This is the most practically consequential consequence
of a shared latent space: it would let a hospital that indexed its archive with
one foundation model query it with another, without re-embedding terabytes of
slides.

Two relevance definitions, because they answer different questions and only one
of them makes all four requested metrics meaningful:

* **Identity retrieval** — the correct answer is *the same patch*, so there is
  exactly one relevant item. Recall@K, MRR and NDCG are all informative;
  mAP collapses to MRR by definition, and is reported anyway for completeness.
  This is the strict test of whether alignment preserves patch identity.
* **Label retrieval** — the correct answers are *all patches of the same class*,
  so relevance is many-to-many and mAP and NDCG become genuinely distinct
  measures of ranking quality. This is the test of whether alignment preserves
  semantics, which is what actually matters for archive search.

The unaligned baseline
----------------------
Models have different embedding widths, so "retrieval without alignment" is not
directly defined. :func:`naive_common_space` supplies the fair control: reduce
each model to a common dimension *independently*, with no cross-model
information used. Any gain the aligners show over that baseline is attributable
to alignment rather than to dimensionality reduction.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .cosine import l2_normalize
from .preprocessing import as_matrix

__all__ = [
    "retrieval_metrics",
    "identity_retrieval",
    "label_retrieval",
    "naive_common_space",
    "cross_model_retrieval_table",
    "compare_retrieval",
]


def _score_matrix(Q: np.ndarray, D: np.ndarray, metric: str) -> np.ndarray:
    """Similarity scores between every query and every database item."""
    if metric == "cosine":
        return l2_normalize(Q) @ l2_normalize(D).T
    if metric == "euclidean":
        return -(
            (Q**2).sum(1)[:, None] + (D**2).sum(1)[None, :] - 2.0 * (Q @ D.T)
        )
    raise ValueError(f"unknown metric {metric!r}; expected 'cosine' or 'euclidean'")


def retrieval_metrics(
    scores: np.ndarray,
    relevant: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K, mAP, MRR and NDCG from a score matrix.

    Parameters
    ----------
    scores : numpy.ndarray
        Query-by-database similarity scores, shape ``(n_query, n_db)``. Higher
        is more similar.
    relevant : numpy.ndarray
        Boolean relevance mask of the same shape; ``relevant[i, j]`` is True if
        database item j is a correct answer for query i.
    ks : sequence of int, default (1, 5, 10)
        Cutoffs for Recall@K.

    Returns
    -------
    dict
        ``recall@k`` for each k, plus ``map``, ``mrr``, ``ndcg`` and
        ``median_rank`` (of the first relevant hit). Queries with no relevant
        item are skipped.

    Notes
    -----
    Recall@K here is the fraction of a query's relevant items found in the top
    K — for identity retrieval, where there is exactly one, that coincides with
    the usual "is it in the top K" reading.
    """
    if scores.shape != relevant.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {relevant.shape}")

    n_rel = relevant.sum(axis=1)
    keep = n_rel > 0
    if not keep.any():
        raise ValueError("no query has a relevant database item")

    scores, relevant, n_rel = scores[keep], relevant[keep], n_rel[keep]
    n_q, n_db = scores.shape

    order = np.argsort(-scores, axis=1)
    hits = np.take_along_axis(relevant, order, axis=1)  # (n_q, n_db) ranked

    positions = np.arange(1, n_db + 1)
    out: dict[str, float] = {}

    for k in ks:
        kk = min(k, n_db)
        out[f"recall@{k}"] = float(
            np.mean(hits[:, :kk].sum(axis=1) / np.minimum(n_rel, kk))
        )

    # Average precision: mean of precision@rank over each relevant hit.
    cum_hits = np.cumsum(hits, axis=1)
    precision_at = cum_hits / positions[None, :]
    ap = (precision_at * hits).sum(axis=1) / n_rel
    out["map"] = float(np.mean(ap))

    # Reciprocal rank of the first relevant hit.
    first = np.argmax(hits, axis=1)
    has_hit = hits.any(axis=1)
    rr = np.where(has_hit, 1.0 / (first + 1), 0.0)
    out["mrr"] = float(np.mean(rr))
    out["median_rank"] = float(np.median(np.where(has_hit, first + 1, n_db)))

    # NDCG with binary gains over the full ranking.
    discount = 1.0 / np.log2(positions + 1)
    dcg = (hits * discount[None, :]).sum(axis=1)
    ideal = np.array(
        [discount[: int(r)].sum() for r in n_rel], dtype=np.float64
    )
    out["ndcg"] = float(np.mean(dcg / np.maximum(ideal, 1e-12)))

    out["n_query"] = float(n_q)
    out["n_database"] = float(n_db)
    return out


def identity_retrieval(
    query: np.ndarray,
    database: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    metric: str = "cosine",
    max_samples: int | None = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Retrieve each query patch's own counterpart from another model's index.

    Parameters
    ----------
    query, database : numpy.ndarray
        Row-paired embeddings from two models, shape ``(n, d)`` each — row i of
        both is the same patch, which is the ground truth.
    ks : sequence of int, default (1, 5, 10)
        Recall cutoffs.
    metric : {'cosine', 'euclidean'}, default 'cosine'
        Retrieval metric.
    max_samples : int or None, default 5000
        Subsample before the O(n^2) score computation. Note that recall depends
        on database size, so this must be held constant when comparing runs.
    seed : int, default 0
        Subsampling seed.

    Returns
    -------
    dict
        Metrics from :func:`retrieval_metrics`. Chance Recall@1 is ``1 / n``.
    """
    Q = np.asarray(query, dtype=np.float64)
    D = np.asarray(database, dtype=np.float64)
    if Q.shape[0] != D.shape[0]:
        raise ValueError(
            f"query and database must be row-paired, got {Q.shape[0]} and {D.shape[0]}"
        )

    n = Q.shape[0]
    if max_samples is not None and n > max_samples:
        idx = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)
        Q, D = Q[idx], D[idx]
        n = max_samples

    scores = _score_matrix(Q, D, metric)
    return retrieval_metrics(scores, np.eye(n, dtype=bool), ks=ks)


def label_retrieval(
    query: np.ndarray,
    database: np.ndarray,
    query_labels: Sequence,
    database_labels: Sequence | None = None,
    ks: Sequence[int] = (1, 5, 10),
    metric: str = "cosine",
    exclude_self: bool = True,
    max_samples: int | None = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Retrieve items of the same class from another model's index.

    Parameters
    ----------
    query, database : numpy.ndarray
        Embeddings, shape ``(n_q, d)`` and ``(n_db, d)``.
    query_labels : sequence
        Class of each query item.
    database_labels : sequence, optional
        Class of each database item. Defaults to ``query_labels``, i.e. the
        same items indexed under a different model.
    ks : sequence of int, default (1, 5, 10)
        Recall cutoffs.
    metric : {'cosine', 'euclidean'}, default 'cosine'
        Retrieval metric.
    exclude_self : bool, default True
        Mask the diagonal when query and database are the same items. Without
        this a model trivially retrieves each item's own counterpart at rank 1
        and the semantic question goes unmeasured.
    max_samples : int or None, default 5000
        Subsample size.
    seed : int, default 0
        Subsampling seed.

    Returns
    -------
    dict
        Metrics from :func:`retrieval_metrics`. Chance mAP is roughly the class
        prevalence, so compare against that, not against zero.
    """
    Q = np.asarray(query, dtype=np.float64)
    D = np.asarray(database, dtype=np.float64)
    ql = np.asarray(query_labels)
    dl = np.asarray(database_labels) if database_labels is not None else ql

    if Q.shape[0] != ql.shape[0] or D.shape[0] != dl.shape[0]:
        raise ValueError("labels must match their embedding row counts")

    paired = Q.shape[0] == D.shape[0]
    if max_samples is not None and Q.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Q.shape[0], size=max_samples, replace=False)
        Q, ql = Q[idx], ql[idx]
        if paired:
            D, dl = D[idx], dl[idx]
        elif D.shape[0] > max_samples:
            jdx = rng.choice(D.shape[0], size=max_samples, replace=False)
            D, dl = D[jdx], dl[jdx]

    scores = _score_matrix(Q, D, metric)
    relevant = ql[:, None] == dl[None, :]

    if exclude_self and scores.shape[0] == scores.shape[1]:
        n = scores.shape[0]
        eye = np.eye(n, dtype=bool)
        scores = np.where(eye, -np.inf, scores)
        relevant = relevant & ~eye

    return retrieval_metrics(scores, relevant, ks=ks)


def naive_common_space(
    views: Mapping[str, np.ndarray],
    dim: int = 64,
    method: str = "pca",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Reduce every model to a common dimension without using cross-model info.

    The honest "no alignment" control. Each model is reduced on its own, so the
    resulting spaces share a dimension but nothing else — any cross-model
    retrieval above chance here would be coincidence.

    Parameters
    ----------
    views : mapping of str to numpy.ndarray
        ``{model: (n, d)}``, row-paired.
    dim : int, default 64
        Common dimensionality. Match the aligners' ``latent_dim`` so the
        comparison is like-for-like.
    method : {'pca', 'truncate'}, default 'pca'
        ``'pca'`` keeps each model's leading principal components;
        ``'truncate'`` keeps its first ``dim`` raw features. PCA is the
        stronger and fairer baseline.
    seed : int, default 0
        Unused for the deterministic methods; accepted for interface symmetry.

    Returns
    -------
    dict of str to numpy.ndarray
        ``{model: (n, dim)}``.
    """
    out = {}
    for name, X in views.items():
        Xc = as_matrix(X)
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        if method == "pca":
            k = min(dim, min(Xc.shape))
            U, s, _ = np.linalg.svd(Xc, full_matrices=False)
            Z = U[:, :k] * s[:k]
            if k < dim:
                Z = np.pad(Z, ((0, 0), (0, dim - k)))
        elif method == "truncate":
            Z = Xc[:, :dim]
            if Z.shape[1] < dim:
                Z = np.pad(Z, ((0, 0), (0, dim - Z.shape[1])))
        else:
            raise ValueError(f"unknown method {method!r}")
        out[name] = Z
    return out


def cross_model_retrieval_table(
    latents: Mapping[str, np.ndarray],
    ks: Sequence[int] = (1, 5, 10),
    labels: Sequence | None = None,
    mode: str = "identity",
    metric: str = "cosine",
    max_samples: int | None = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Score retrieval for every ordered pair of models.

    Parameters
    ----------
    latents : mapping of str to numpy.ndarray
        ``{model: (n, k)}`` in a common space, row-paired.
    ks : sequence of int, default (1, 5, 10)
        Recall cutoffs.
    labels : sequence, optional
        Item classes. Required when ``mode='label'``.
    mode : {'identity', 'label'}, default 'identity'
        Relevance definition.
    metric : {'cosine', 'euclidean'}, default 'cosine'
        Retrieval metric.
    max_samples : int or None, default 2000
        Subsample size, held constant across pairs.
    seed : int, default 0
        Subsampling seed.

    Returns
    -------
    pandas.DataFrame
        One row per ordered ``(query, database)`` pair, including the
        same-model diagonal as an upper bound.
    """
    names = list(latents)
    rows = []
    for q in names:
        for d in names:
            if mode == "identity":
                scores = identity_retrieval(
                    latents[q],
                    latents[d],
                    ks=ks,
                    metric=metric,
                    max_samples=max_samples,
                    seed=seed,
                )
            elif mode == "label":
                if labels is None:
                    raise ValueError("mode='label' requires labels")
                scores = label_retrieval(
                    latents[q],
                    latents[d],
                    labels,
                    ks=ks,
                    metric=metric,
                    max_samples=max_samples,
                    seed=seed,
                )
            else:
                raise ValueError(f"unknown mode {mode!r}")
            rows.append(
                {"query": q, "database": d, "same_model": q == d, **scores}
            )
    return pd.DataFrame(rows)


def compare_retrieval(
    views: Mapping[str, np.ndarray],
    aligners: Mapping[str, object],
    ks: Sequence[int] = (1, 5, 10),
    labels: Sequence | None = None,
    mode: str = "identity",
    baseline_dim: int = 64,
    max_samples: int | None = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare unaligned baselines against aligned cross-model retrieval.

    Parameters
    ----------
    views : mapping of str to numpy.ndarray
        Held-out ``{model: embeddings}`` in original spaces, row-paired.
    aligners : mapping of str to BaseAligner
        ``{name: fitted_aligner}``.
    ks : sequence of int, default (1, 5, 10)
        Recall cutoffs.
    labels : sequence, optional
        Item classes, required for ``mode='label'``.
    mode : {'identity', 'label'}, default 'identity'
        Relevance definition.
    baseline_dim : int, default 64
        Dimension for the unaligned baselines. Set this to the aligners'
        ``latent_dim``.
    max_samples : int or None, default 2000
        Subsample size, constant across every condition.
    seed : int, default 0
        Seed.

    Returns
    -------
    pandas.DataFrame
        One row per ``(condition, query, database)``. Cross-model rows are the
        ones of interest; ``same_model`` rows give the ceiling.
    """
    conditions: dict[str, Mapping[str, np.ndarray]] = {
        "unaligned_pca": naive_common_space(views, dim=baseline_dim, method="pca"),
        "unaligned_truncate": naive_common_space(
            views, dim=baseline_dim, method="truncate"
        ),
    }
    for name, aligner in aligners.items():
        conditions[name] = aligner.transform(views)

    frames = []
    for condition, latents in conditions.items():
        table = cross_model_retrieval_table(
            latents,
            ks=ks,
            labels=labels,
            mode=mode,
            max_samples=max_samples,
            seed=seed,
        )
        table.insert(0, "condition", condition)
        frames.append(table)

    return pd.concat(frames, ignore_index=True)


def retrieval_summary(table: pd.DataFrame, metric_col: str = "recall@1") -> pd.DataFrame:
    """Summarise a retrieval table by condition, cross-model pairs only.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`compare_retrieval`.
    metric_col : str, default 'recall@1'
        Column to summarise.

    Returns
    -------
    pandas.DataFrame
        Mean of every metric per condition over cross-model pairs, sorted by
        ``metric_col`` descending.
    """
    cross = table[~table["same_model"]]
    cols = [
        c
        for c in cross.columns
        if c not in ("condition", "query", "database", "same_model")
    ]
    out = cross.groupby("condition")[cols].mean()
    return out.sort_values(metric_col, ascending=False)


__all__.append("retrieval_summary")
