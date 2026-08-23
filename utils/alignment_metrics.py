"""Evaluation of shared latent spaces.

A shared space can fail in two opposite ways, and any honest evaluation has to
measure both:

* **Under-alignment** — the models' projections never come into agreement, so
  there is no shared space to speak of. Caught by ``alignment_error``,
  ``paired_cosine`` and the retrieval metrics.
* **Collapse** — the projections agree perfectly because the aligner threw away
  everything that distinguished the patches. Caught by ``reconstruction_r2``,
  ``effective_rank`` and ``neighborhood_preservation``.

Optimising either family alone produces a degenerate result, so
:func:`evaluate_aligner` always reports them side by side.

Everything here should be run on **held-out patches** (see
:func:`utils.alignment.split_views`): a map with more parameters than training
samples reconstructs its training set perfectly and generalises not at all.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .cka import linear_cka
from .cosine import l2_normalize
from .preprocessing import as_matrix

__all__ = [
    "reconstruction_error",
    "alignment_error",
    "paired_cosine",
    "cross_model_retrieval",
    "shared_space_similarity",
    "neighborhood_preservation",
    "effective_rank",
    "evaluate_aligner",
    "compare_aligners",
    "cross_model_transfer_scores",
]


# ---------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------


def reconstruction_error(
    aligner,
    views: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Round-trip each view through the shared space and score the result.

    Parameters
    ----------
    aligner : BaseAligner
        A fitted aligner.
    views : mapping of str to array-like
        Held-out ``{model_name: embedding_matrix}``.

    Returns
    -------
    pandas.DataFrame
        One row per model with columns:

        ``nmse``
            ``||X - X_hat||^2 / ||X - mean(X)||^2``. Below 1 means the shared
            space beats predicting the mean; above 1 means it is worse than
            useless.
        ``r2``
            ``1 - nmse``, the fraction of the model's variance the shared space
            retains.
        ``cosine``
            Mean cosine between original and reconstructed embeddings —
            direction fidelity, which is what cosine-based retrieval depends on.
    """
    rows = []
    for name, X in views.items():
        X = as_matrix(X)
        X_hat = aligner.inverse_transform(aligner.transform_view(name, X), name)

        resid = float(((X - X_hat) ** 2).sum())
        total = float(((X - X.mean(axis=0)) ** 2).sum()) or 1.0
        nmse = resid / total
        cos = float(np.mean(np.sum(l2_normalize(X) * l2_normalize(X_hat), axis=1)))
        rows.append(
            {"model": name, "nmse": nmse, "r2": 1.0 - nmse, "cosine": cos}
        )
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------
# alignment quality
# ---------------------------------------------------------------------


def alignment_error(
    latents: Mapping[str, np.ndarray],
    normalize: bool = True,
) -> float:
    """Mean squared spread of the models' projections of the same patch.

    Parameters
    ----------
    latents : mapping of str to numpy.ndarray
        ``{model_name: shared_coordinates}``, row-paired.
    normalize : bool, default True
        Divide by the total variance of the consensus embedding, giving a
        scale-free number: 0 is perfect agreement, 1 means the disagreement
        between models is as large as the spread between patches — i.e. the
        shared space carries no usable patch identity.

    Returns
    -------
    float
        Normalised alignment error.
    """
    Z = np.stack([np.asarray(v, dtype=np.float64) for v in latents.values()], axis=0)
    consensus = Z.mean(axis=0)
    spread = float(((Z - consensus) ** 2).sum(axis=-1).mean())
    if not normalize:
        return spread
    total = float(((consensus - consensus.mean(axis=0)) ** 2).sum(axis=-1).mean())
    return spread / max(total, 1e-12)


def paired_cosine(
    latents: Mapping[str, np.ndarray],
    center: bool = True,
) -> pd.DataFrame:
    """Mean cosine between two models' projections of the same patch.

    Parameters
    ----------
    latents : mapping of str to numpy.ndarray
        ``{model_name: shared_coordinates}``, row-paired.
    center : bool, default True
        Subtract the per-dimension mean before computing cosines. This matters
        more than it sounds: when every patch shares a large common component —
        routine in shared latent spaces — the uncentered cosine saturates at
        1.000 even for a space whose cross-model Recall@1 is only 0.5. The
        centered version measures agreement about what makes *patches differ*,
        which is what a shared space is for.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix of mean paired cosines, indexed by model.
    """
    names = list(latents)
    arrays = {n: np.asarray(latents[n], dtype=np.float64) for n in names}
    if center:
        arrays = {n: X - X.mean(axis=0, keepdims=True) for n, X in arrays.items()}
    norm = {n: l2_normalize(X) for n, X in arrays.items()}

    S = np.eye(len(names))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            val = float(np.mean(np.sum(norm[a] * norm[b], axis=1)))
            S[i, j] = S[j, i] = val
    return pd.DataFrame(S, index=names, columns=names)


def cross_model_retrieval(
    query: np.ndarray,
    database: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    metric: str = "cosine",
    max_samples: int | None = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Retrieve each query patch from a database built with a *different* model.

    The Phase VI evaluation, reported here because it is also the sharpest
    single test of a shared space: index the database with model A's
    projections, query with model B's, and check whether patch i retrieves
    patch i. Chance-level Recall@1 is ``1 / n``.

    Parameters
    ----------
    query : numpy.ndarray
        Query-side shared coordinates, shape ``(n, k)``.
    database : numpy.ndarray
        Database-side shared coordinates, shape ``(n, k)``, row-paired with
        ``query`` (row i of both is the same patch — that is the ground truth).
    ks : sequence of int, default (1, 5, 10)
        Cutoffs for Recall@K.
    metric : {'cosine', 'euclidean'}, default 'cosine'
        Retrieval metric.
    max_samples : int or None, default 5000
        Subsample for the O(n^2) similarity computation.
    seed : int, default 0
        Subsampling seed.

    Returns
    -------
    dict
        ``recall@k`` for each k, plus ``mrr`` (mean reciprocal rank),
        ``median_rank``, and ``n`` (the database size the numbers refer to —
        recall is not comparable across different n).
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

    if metric == "cosine":
        scores = l2_normalize(Q) @ l2_normalize(D).T
    elif metric == "euclidean":
        scores = -(
            (Q**2).sum(1)[:, None] + (D**2).sum(1)[None, :] - 2.0 * (Q @ D.T)
        )
    else:
        raise ValueError(f"unknown metric {metric!r}")

    truth = scores[np.arange(n), np.arange(n)]
    # Rank of the true match = how many entries score strictly higher, +1.
    ranks = (scores > truth[:, None]).sum(axis=1) + 1

    out: dict[str, float] = {f"recall@{k}": float(np.mean(ranks <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks))
    out["median_rank"] = float(np.median(ranks))
    out["n"] = float(n)
    return out


def shared_space_similarity(
    latents: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Linear CKA between the models' projections inside the shared space.

    The Phase I metric re-run after alignment. Comparing this matrix with the
    pre-alignment one quantifies exactly what the alignment bought.

    Parameters
    ----------
    latents : mapping of str to numpy.ndarray
        ``{model_name: shared_coordinates}``, row-paired.

    Returns
    -------
    pandas.DataFrame
        Symmetric CKA matrix indexed by model.
    """
    names = list(latents)
    S = np.eye(len(names))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            S[i, j] = S[j, i] = linear_cka(latents[a], latents[b])
    return pd.DataFrame(S, index=names, columns=names)


def neighborhood_preservation(
    original: np.ndarray,
    embedded: np.ndarray,
    k: int = 10,
    max_samples: int | None = 3000,
    seed: int = 0,
) -> float:
    """Fraction of each patch's k nearest neighbours that survive the projection.

    The collapse detector. A shared space can score perfectly on alignment
    while having destroyed the local morphological structure that makes the
    embeddings useful downstream; this catches that.

    Parameters
    ----------
    original : numpy.ndarray
        Embeddings in the original model space, shape ``(n, d)``.
    embedded : numpy.ndarray
        Shared-space coordinates, shape ``(n, k_latent)``, row-paired.
    k : int, default 10
        Neighbourhood size.
    max_samples : int or None, default 3000
        Subsample for the O(n^2) neighbour search.
    seed : int, default 0
        Subsampling seed.

    Returns
    -------
    float
        Mean overlap in ``[0, 1]``. Chance level is roughly ``k / n``.
    """
    A = as_matrix(original)
    B = np.asarray(embedded, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        raise ValueError("original and embedded must be row-paired")

    n = A.shape[0]
    if max_samples is not None and n > max_samples:
        idx = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)
        A, B = A[idx], B[idx]
        n = max_samples

    k = min(k, n - 1)
    nn_a = _knn_indices(A, k)
    nn_b = _knn_indices(B, k)

    overlap = [len(set(nn_a[i]) & set(nn_b[i])) / k for i in range(n)]
    return float(np.mean(overlap))


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k nearest neighbours, excluding itself."""
    Xn = l2_normalize(X - X.mean(axis=0))
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]


def effective_rank(Z: np.ndarray) -> float:
    """Entropy-based effective rank of a shared representation.

    The other half of the collapse detector: an aligner can satisfy the
    alignment objective by mapping everything onto a low-dimensional blob. This
    reports how many latent dimensions are *actually* being used, as the
    exponential of the entropy of the normalised singular-value spectrum.

    Parameters
    ----------
    Z : numpy.ndarray
        Shared-space coordinates of shape ``(n, k)``.

    Returns
    -------
    float
        Effective rank in ``[1, k]``. Well below ``k`` means the shared space
        has partially collapsed.

    References
    ----------
    Roy & Vetterli (2007), "The effective rank: a measure of effective
    dimensionality", EUSIPCO.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0)
    s = np.linalg.svd(Zc, compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------------
# aggregate reports
# ---------------------------------------------------------------------


def evaluate_aligner(
    aligner,
    views: Mapping[str, np.ndarray],
    ks: Sequence[int] = (1, 5, 10),
    retrieval_samples: int | None = 2000,
    neighborhood_k: int = 10,
) -> dict:
    """Full evaluation of one fitted aligner on held-out views.

    Parameters
    ----------
    aligner : BaseAligner
        Fitted aligner.
    views : mapping of str to array-like
        Held-out ``{model_name: embedding_matrix}``.
    ks : sequence of int, default (1, 5, 10)
        Recall cutoffs.
    retrieval_samples : int or None, default 2000
        Subsample size for retrieval and neighbourhood metrics.
    neighborhood_k : int, default 10
        Neighbourhood size for preservation.

    Returns
    -------
    dict
        With keys:

        ``summary`` : pandas.Series
            Headline scalars — mean reconstruction R^2, alignment error, mean
            paired cosine, mean Recall@1, mean neighbourhood preservation,
            effective rank.
        ``reconstruction`` : pandas.DataFrame
            Per-model reconstruction scores.
        ``paired_cosine`` : pandas.DataFrame
            Pairwise agreement in the shared space.
        ``shared_cka`` : pandas.DataFrame
            Pairwise CKA in the shared space.
        ``retrieval`` : pandas.DataFrame
            Per ordered model pair, the cross-model retrieval scores.
        ``neighborhood`` : pandas.Series
            Per-model neighbourhood preservation.
    """
    latents = aligner.transform(views)
    names = list(latents)

    recon = reconstruction_error(aligner, views)
    cos = paired_cosine(latents)
    cka = shared_space_similarity(latents)
    align_err = alignment_error(latents)

    retrieval_rows = []
    for a, b in combinations(names, 2):
        for q, d in ((a, b), (b, a)):
            scores = cross_model_retrieval(
                latents[q], latents[d], ks=ks, max_samples=retrieval_samples
            )
            retrieval_rows.append({"query": q, "database": d, **scores})
    retrieval = pd.DataFrame(retrieval_rows)

    neigh = pd.Series(
        {
            name: neighborhood_preservation(
                views[name],
                latents[name],
                k=neighborhood_k,
                max_samples=retrieval_samples,
            )
            for name in names
        },
        name="neighborhood_preservation",
    )

    eff_rank = float(np.mean([effective_rank(latents[n]) for n in names]))
    off_diag = ~np.eye(len(names), dtype=bool)

    summary = pd.Series(
        {
            "reconstruction_r2": float(recon["r2"].mean()),
            "reconstruction_cosine": float(recon["cosine"].mean()),
            "alignment_error": align_err,
            "paired_cosine": float(cos.values[off_diag].mean()),
            "shared_cka": float(cka.values[off_diag].mean()),
            "recall@1": float(retrieval["recall@1"].mean()),
            f"recall@{ks[-1]}": float(retrieval[f"recall@{ks[-1]}"].mean()),
            "mrr": float(retrieval["mrr"].mean()),
            "neighborhood_preservation": float(neigh.mean()),
            "effective_rank": eff_rank,
            "latent_dim": float(aligner.latent_dim),
        },
        name=type(aligner).__name__,
    )

    return {
        "summary": summary,
        "reconstruction": recon,
        "paired_cosine": cos,
        "shared_cka": cka,
        "retrieval": retrieval,
        "neighborhood": neigh,
    }


def compare_aligners(
    aligners: Mapping[str, object],
    views: Mapping[str, np.ndarray],
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Evaluate several fitted aligners on the same held-out views.

    Parameters
    ----------
    aligners : mapping of str to BaseAligner
        ``{method_name: fitted_aligner}``.
    views : mapping of str to array-like
        Held-out views.
    **kwargs
        Forwarded to :func:`evaluate_aligner`.

    Returns
    -------
    tuple
        ``(summary_table, full_reports)`` where the table has one row per
        method and the reports dict holds each method's complete output.
    """
    reports, rows = {}, {}
    for label, aligner in aligners.items():
        report = evaluate_aligner(aligner, views, **kwargs)
        reports[label] = report
        rows[label] = report["summary"]
    return pd.DataFrame(rows).T, reports


def cross_model_transfer_scores(
    aligner,
    views: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Score model-to-model translation through the shared space.

    For every ordered pair (A, B), encode with A, decode as B, and compare
    against B's real embeddings of the same patches. This is the Phase VI
    experiment (``CONCH -> UNI``) evaluated in B's own feature space.

    Parameters
    ----------
    aligner : BaseAligner
        Fitted aligner.
    views : mapping of str to array-like
        Held-out ``{model_name: embedding_matrix}``.

    Returns
    -------
    pandas.DataFrame
        One row per ordered pair with ``cosine`` (mean cosine to the true
        target embedding) and ``r2`` (variance of the target explained). An
        ``r2`` at or below 0 means the translation is no better than predicting
        the target model's mean embedding.
    """
    names = list(views)
    rows = []
    for source in names:
        for target in names:
            if source == target:
                continue
            pred = aligner.translate(views[source], source, target)
            true = as_matrix(views[target])

            resid = float(((true - pred) ** 2).sum())
            total = float(((true - true.mean(axis=0)) ** 2).sum()) or 1.0
            cos = float(
                np.mean(np.sum(l2_normalize(true) * l2_normalize(pred), axis=1))
            )
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "cosine": cos,
                    "r2": 1.0 - resid / total,
                }
            )
    return pd.DataFrame(rows)
