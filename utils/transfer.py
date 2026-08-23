"""Phase VI: converting one model's representation into another's.

The operation is ``source -> shared space -> target``: encode a patch with model
A, decode it as if model B had produced it. If the central hypothesis holds,
the result should be usable *anywhere B's real embeddings are usable* — which
is a much stronger claim than the embeddings merely being correlated.

Four evaluations, in increasing order of how much they demand:

``cosine`` / ``r2``
    Is the translated vector close to B's real embedding of the same patch?
``retrieval``
    Does a translated query retrieve the right patch from a database of B's
    *real* embeddings, in B's own native space?
``linear probe``
    Can a classifier **trained on B's real embeddings** classify translated
    ones without ever having seen them? This is the practical test: it says a
    downstream model deployed on B works on data encoded with A.

Empirically these come apart, and **not in the direction you would expect**.
On the six-encoder CPTAC 10x/256px set with GCCA, cross-model transfer scores
R^2 0.59 and recall@1 0.55 — mediocre — while the linear probe retains 96% of
its accuracy (0.820 -> 0.787 balanced accuracy). So the translation does *not*
reproduce the target's embedding vector faithfully, yet it reliably lands on
the right side of the target's decision boundary.

The discriminative content transfers even though the geometry does not. That
means reconstruction error understates transferability, and any Phase VI claim
should be made on the probe rather than on R^2.

Transferability constraint
--------------------------
Transfer is only defined between encoders sharing a coordinate grid, since the
aligner has to be fitted on row-paired patches. Encoders in different
``(magnification, patch_size)`` groups — KEEP at 256px and MUSK at 384px, say —
cannot be transferred between without re-extracting one of them.
:func:`transferable_pairs` reports what is actually possible.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .cosine import l2_normalize
from .preprocessing import as_matrix
from .retrieval import retrieval_metrics

__all__ = [
    "transferable_pairs",
    "transfer_fidelity",
    "transfer_retrieval",
    "linear_probe_transfer",
    "evaluate_transfer",
]


def transferable_pairs(store, group_key: str | None = None) -> pd.DataFrame:
    """List the encoder pairs that can actually be transferred between.

    Parameters
    ----------
    store : FeatureStore
        The feature store.
    group_key : str, optional
        Restrict to one group. Defaults to every group with >= 2 encoders.

    Returns
    -------
    pandas.DataFrame
        One row per ordered ``(source, target)`` pair with its group. Pairs
        absent from this table need one of the two encoders re-extracted onto
        the other's grid before Phase VI can touch them.
    """
    groups = (
        [store.group(group_key)]
        if group_key
        else [g for g in store.groups.values() if g.n_encoders >= 2]
    )
    rows = []
    for g in groups:
        encoders = sorted(g.encoders)
        for src in encoders:
            for tgt in encoders:
                if src != tgt:
                    rows.append(
                        {
                            "group": g.key,
                            "source": src,
                            "target": tgt,
                            "magnification": g.magnification,
                            "patch_size": g.patch_size,
                        }
                    )
    return pd.DataFrame(rows)


def transfer_fidelity(
    predicted: np.ndarray, true: np.ndarray
) -> dict[str, float]:
    """Score a translation against the target model's real embeddings.

    Parameters
    ----------
    predicted : numpy.ndarray
        Translated embeddings, shape ``(n, d_target)``.
    true : numpy.ndarray
        The target model's real embeddings of the same patches.

    Returns
    -------
    dict
        ``cosine`` (mean row-wise cosine), ``r2`` (fraction of the target's
        variance explained) and ``nmse``. An ``r2`` at or below 0 means the
        translation is no better than predicting the target's mean embedding.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    truth = as_matrix(true)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {truth.shape}")

    resid = float(((truth - pred) ** 2).sum())
    total = float(((truth - truth.mean(axis=0)) ** 2).sum()) or 1.0
    cosine = float(np.mean(np.sum(l2_normalize(truth) * l2_normalize(pred), axis=1)))
    return {"cosine": cosine, "r2": 1.0 - resid / total, "nmse": resid / total}


def transfer_retrieval(
    predicted: np.ndarray,
    true: np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    max_samples: int | None = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Retrieve from the target model's real index using translated queries.

    Unlike the Phase VII retrieval, which runs inside the shared space, this
    runs in the **target model's native space** — the database is untouched real
    embeddings, exactly as a deployed system would hold them.

    Parameters
    ----------
    predicted : numpy.ndarray
        Translated query embeddings, shape ``(n, d_target)``.
    true : numpy.ndarray
        The target's real embeddings, row-paired with ``predicted``.
    ks : sequence of int, default (1, 5, 10)
        Recall cutoffs.
    max_samples : int or None, default 2000
        Database size. Recall depends on it, so hold it fixed across pairs.
    seed : int, default 0
        Subsampling seed.

    Returns
    -------
    dict
        Metrics from :func:`utils.retrieval.retrieval_metrics`.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    truth = as_matrix(true)

    n = pred.shape[0]
    if max_samples is not None and n > max_samples:
        idx = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)
        pred, truth = pred[idx], truth[idx]
        n = max_samples

    scores = l2_normalize(pred) @ l2_normalize(truth).T
    return retrieval_metrics(scores, np.eye(n, dtype=bool), ks=ks)


def linear_probe_transfer(
    predicted: np.ndarray,
    true: np.ndarray,
    labels: Sequence,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    seed: int = 0,
    max_iter: int = 1000,
) -> dict[str, float]:
    """Test whether a probe trained on real target features accepts translated ones.

    Trains logistic regression on the **target model's real** embeddings and
    evaluates it twice on held-out patches: once on the target's real
    embeddings (the ceiling) and once on embeddings translated from the source
    (the transfer). The gap between them is what Phase VI is really measuring —
    a small gap means a downstream model deployed on the target works unchanged
    on data encoded with the source.

    Parameters
    ----------
    predicted : numpy.ndarray
        Translated embeddings, shape ``(n, d_target)``.
    true : numpy.ndarray
        The target's real embeddings, row-paired.
    labels : sequence
        Class label per row.
    train_idx, test_idx : numpy.ndarray, optional
        Explicit split. If omitted, a random 70/30 split is drawn. Pass an
        explicit patient-grouped split when the labels come from slides.
    seed : int, default 0
        Seed for the split and the solver.
    max_iter : int, default 1000
        Solver iterations.

    Returns
    -------
    dict
        ``probe_true`` (ceiling accuracy), ``probe_translated`` (transfer
        accuracy), ``probe_gap`` (ceiling minus transfer), ``probe_ratio``
        (transfer / ceiling) and ``probe_chance`` (majority-class rate).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    pred = np.asarray(predicted, dtype=np.float64)
    truth = as_matrix(true)
    y = np.asarray(labels)

    if pred.shape[0] != truth.shape[0] or truth.shape[0] != y.shape[0]:
        raise ValueError("predicted, true and labels must have equal row counts")
    if len(set(y.tolist())) < 2:
        raise ValueError("need at least 2 classes for a linear probe")

    if train_idx is None or test_idx is None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        train_idx, test_idx = perm[:cut], perm[cut:]

    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(truth[train_idx], y[train_idx])

    acc_true = float(balanced_accuracy_score(y[test_idx], clf.predict(truth[test_idx])))
    acc_trans = float(balanced_accuracy_score(y[test_idx], clf.predict(pred[test_idx])))

    _, counts = np.unique(y[test_idx], return_counts=True)
    chance = float(counts.max() / counts.sum())

    return {
        "probe_true": acc_true,
        "probe_translated": acc_trans,
        "probe_gap": acc_true - acc_trans,
        "probe_ratio": acc_trans / acc_true if acc_true > 0 else float("nan"),
        "probe_chance": chance,
    }


def evaluate_transfer(
    aligner,
    views: Mapping[str, np.ndarray],
    labels: Sequence | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    ks: Sequence[int] = (1, 5, 10),
    max_samples: int | None = 2000,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Evaluate cross-model transfer for every ordered encoder pair.

    Parameters
    ----------
    aligner : BaseAligner
        Fitted aligner supplying ``translate``.
    views : mapping of str to numpy.ndarray
        Held-out ``{encoder: embeddings}``, row-paired.
    labels : sequence, optional
        Per-row class labels. Without them the linear-probe columns are
        omitted and only fidelity and retrieval are reported.
    pairs : sequence of tuple, optional
        Ordered ``(source, target)`` pairs. Defaults to all ordered pairs.
    ks : sequence of int, default (1, 5, 10)
        Recall cutoffs.
    max_samples : int or None, default 2000
        Retrieval database size.
    train_idx, test_idx : numpy.ndarray, optional
        Split for the linear probe.
    seed : int, default 0
        Seed.
    verbose : bool, default False
        Print each pair as it is scored.

    Returns
    -------
    pandas.DataFrame
        One row per ordered pair, with fidelity, retrieval and (if labels were
        given) linear-probe columns. A ``self`` row per encoder — translating
        it to itself — is included as the round-trip ceiling: it isolates how
        much is lost by the shared space alone, before any cross-model step.
    """
    names = list(views)
    if pairs is None:
        pairs = [(s, t) for s in names for t in names if s != t]
        pairs += [(n, n) for n in names]  # round-trip ceilings

    rows = []
    for source, target in pairs:
        if verbose:
            print(f"  {source} -> {target}")
        predicted = aligner.translate(views[source], source, target)
        truth = views[target]

        row: dict = {
            "source": source,
            "target": target,
            "self": source == target,
            **transfer_fidelity(predicted, truth),
        }
        row.update(
            {
                f"retrieval_{k}": v
                for k, v in transfer_retrieval(
                    predicted, truth, ks=ks, max_samples=max_samples, seed=seed
                ).items()
            }
        )
        if labels is not None:
            row.update(
                linear_probe_transfer(
                    predicted,
                    truth,
                    labels,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    seed=seed,
                )
            )
        rows.append(row)

    return pd.DataFrame(rows)


def transfer_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Summarise a transfer table, separating cross-model from round-trip rows.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`evaluate_transfer`.

    Returns
    -------
    pandas.DataFrame
        Two rows — ``cross_model`` and ``self_roundtrip`` — averaging the
        numeric columns. The gap between them is the cost of crossing models,
        as opposed to the cost of the shared space itself.
    """
    numeric = table.select_dtypes("number").columns
    out = table.groupby(table["self"].map({False: "cross_model", True: "self_roundtrip"}))[
        list(numeric)
    ].mean()
    out.index.name = "kind"
    return out


__all__.append("transfer_summary")
