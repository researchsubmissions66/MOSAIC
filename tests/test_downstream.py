"""Tests for Phases VII and VIII: retrieval, labels, bags and MIL.

The retrieval metrics are checked against hand-computable cases — a perfect
ranking, a worst-case ranking, and a known partial ranking — because a subtly
wrong mAP or NDCG would silently misreport the headline Phase VII result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.bags import build_bags, encode_bag  # noqa: E402
from utils.labels import (  # noqa: E402
    TASK_REGISTRY,
    _stage_group,
    available_tasks,
    grouped_split,
)
from utils.retrieval import (  # noqa: E402
    cross_model_retrieval_table,
    identity_retrieval,
    label_retrieval,
    naive_common_space,
    retrieval_metrics,
    retrieval_summary,
)

h5py = pytest.importorskip("h5py")


# --- retrieval metrics ---------------------------------------------------


def test_perfect_ranking_scores_one():
    n = 20
    scores = np.eye(n) * 2.0
    relevant = np.eye(n, dtype=bool)
    out = retrieval_metrics(scores, relevant, ks=(1, 5))
    for key in ("recall@1", "recall@5", "map", "mrr", "ndcg"):
        assert out[key] == pytest.approx(1.0), key
    assert out["median_rank"] == 1.0


def test_worst_ranking_scores_minimum():
    """The single relevant item ranked last in every query."""
    n = 10
    scores = np.zeros((n, n))
    relevant = np.zeros((n, n), dtype=bool)
    for i in range(n):
        scores[i] = np.arange(n)[::-1]  # descending, so index 0 ranks first
        scores[i, n - 1] = -1.0  # force the relevant item to rank last
        relevant[i, n - 1] = True
    out = retrieval_metrics(scores, relevant, ks=(1,))
    assert out["recall@1"] == 0.0
    assert out["mrr"] == pytest.approx(1.0 / n)
    assert out["median_rank"] == float(n)


def test_map_matches_hand_computation():
    """Two relevant items at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2.

    Scores are already descending, so column order is rank order.
    """
    scores = np.array([[4.0, 3.0, 2.0, 1.0]])
    relevant = np.array([[True, False, True, False]])
    out = retrieval_metrics(scores, relevant, ks=(1,))
    assert out["map"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert out["mrr"] == pytest.approx(1.0)


def test_map_reflects_rank_order_not_column_order():
    """The same relevance permuted to ranks 1 and 2 must score higher."""
    scores = np.array([[5.0, 3.0, 4.0, 1.0]])  # rank order: cols 0, 2, 1, 3
    relevant = np.array([[True, False, True, False]])  # hits at ranks 1 and 2
    out = retrieval_metrics(scores, relevant, ks=(1,))
    assert out["map"] == pytest.approx(1.0)


def test_ndcg_matches_hand_computation():
    """One relevant item at rank 2 -> NDCG = (1/log2(3)) / (1/log2(2))."""
    scores = np.array([[5.0, 4.0, 3.0]])
    relevant = np.array([[False, True, False]])
    out = retrieval_metrics(scores, relevant, ks=(1,))
    assert out["ndcg"] == pytest.approx(1.0 / np.log2(3))


def test_map_and_mrr_agree_for_single_relevant_item():
    """With exactly one relevant item, mAP is MRR by definition."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(30, 30))
    relevant = np.eye(30, dtype=bool)
    out = retrieval_metrics(scores, relevant, ks=(1,))
    assert out["map"] == pytest.approx(out["mrr"])


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        retrieval_metrics(np.zeros((3, 4)), np.zeros((3, 5), dtype=bool))


def test_metrics_reject_no_relevant_items():
    with pytest.raises(ValueError, match="no query has a relevant"):
        retrieval_metrics(np.zeros((3, 3)), np.zeros((3, 3), dtype=bool))


# --- identity / label retrieval -----------------------------------------


def test_identity_retrieval_perfect_for_same_space():
    X = np.random.default_rng(0).normal(size=(200, 12))
    out = identity_retrieval(X, X, ks=(1,))
    assert out["recall@1"] == 1.0


def test_identity_retrieval_chance_for_independent_spaces():
    rng = np.random.default_rng(0)
    out = identity_retrieval(
        rng.normal(size=(300, 12)), rng.normal(size=(300, 12)), ks=(1,)
    )
    assert out["recall@1"] < 0.05


def test_identity_retrieval_requires_pairing():
    X = np.random.default_rng(0).normal(size=(50, 8))
    with pytest.raises(ValueError, match="row-paired"):
        identity_retrieval(X, X[:-1])


def test_label_retrieval_excludes_self_by_default():
    """Without self-exclusion an item trivially retrieves itself at rank 1."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    lab = rng.integers(0, 3, size=60)
    with_self = label_retrieval(X, X, lab, ks=(1,), exclude_self=False)
    without = label_retrieval(X, X, lab, ks=(1,), exclude_self=True)
    assert with_self["mrr"] > without["mrr"]


def test_label_retrieval_beats_chance_for_clustered_data():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 8)) * 5
    lab = rng.integers(0, 4, size=200)
    X = centers[lab] + rng.normal(size=(200, 8)) * 0.2
    out = label_retrieval(X, X, lab, ks=(1,))
    assert out["recall@1"] > 0.9
    assert out["map"] > 0.8


def test_naive_common_space_equalises_dimension():
    rng = np.random.default_rng(0)
    views = {"a": rng.normal(size=(80, 20)), "b": rng.normal(size=(80, 45))}
    out = naive_common_space(views, dim=16)
    assert all(v.shape == (80, 16) for v in out.values())


def test_naive_common_space_does_not_align():
    """The unaligned control must not achieve cross-model retrieval."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(300, 10))
    views = {
        "a": latent @ rng.normal(size=(10, 32)),
        "b": latent @ rng.normal(size=(10, 48)),
    }
    naive = naive_common_space(views, dim=10)
    out = identity_retrieval(naive["a"], naive["b"], ks=(1,))
    assert out["recall@1"] < 0.5, "unaligned baseline should not solve the task"


def test_cross_model_retrieval_table_covers_all_pairs():
    rng = np.random.default_rng(0)
    latents = {n: rng.normal(size=(100, 8)) for n in ("a", "b", "c")}
    table = cross_model_retrieval_table(latents, ks=(1,), max_samples=100)
    assert len(table) == 9
    assert table[table["same_model"]]["recall@1"].min() == 1.0


def test_retrieval_summary_excludes_same_model():
    rng = np.random.default_rng(0)
    latents = {n: rng.normal(size=(60, 8)) for n in ("a", "b")}
    table = cross_model_retrieval_table(latents, ks=(1,), max_samples=60)
    table.insert(0, "condition", "x")
    summary = retrieval_summary(table)
    assert list(summary.index) == ["x"]
    assert summary.loc["x", "recall@1"] < 1.0


# --- labels --------------------------------------------------------------


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("Stage I", "early"),
        ("Stage IA", "early"),
        ("Stage IIB", "early"),
        ("Stage III", "late"),
        ("Stage IIIA", "late"),
        ("Stage IV", "late"),
        ("Stage IVA", "late"),
        ("Stage X", None),
        ("Stage 0", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_stage_grouping(stage, expected):
    """Substages carry trailing letters, which a word-boundary match would drop."""
    assert _stage_group(stage) == expected


def test_task_registry_is_well_formed():
    assert len(available_tasks()) == 14
    for name in available_tasks():
        t = TASK_REGISTRY[name]
        assert t.name == name
        assert len(t.classes) >= 2
        assert t.store_cohort in ("master_benchmark", "cptac_benchmark")
        assert t.source in ("tcga", "cptac", "tcga_clinical", "cptac_mutation")


def test_registry_composition_is_locked():
    """Guard the exact task set, so tasks cannot drift in or out unnoticed."""
    assert set(available_tasks()) == {
        "tcga_nsclc",
        "tcga_brca_subtype",
        "tcga_brca_stage",
        "tcga_nsclc_stage",
        "cptac_nsclc",
    } | {
        f"cptac_{c}_{g}"
        for c, genes in {
            "brca": ("pik3ca", "map3k1", "gata3"),
            "coad": ("kras", "pik3ca", "tp53"),
            "luad": ("tp53", "stk11", "kras"),
        }.items()
        for g in genes
    }


def test_no_tissue_of_origin_tasks():
    """Tissue-of-origin labels saturate and cannot rank encoders, so none exist.

    A task whose classes name different organs would be one; the mutation and
    staging tasks avoid it via their cohort filter.
    """
    organ_sets = [
        {"BRCA", "LUAD", "LUSC"},
        {"CPTAC-BRCA", "CPTAC-COAD", "CPTAC-LSCC", "CPTAC-LUAD"},
    ]
    for name in available_tasks():
        classes = set(TASK_REGISTRY[name].classes)
        for organs in organ_sets:
            assert len(classes & organs) < 3, f"{name} looks like tissue-of-origin"


def test_staging_tasks_are_cohort_confined():
    """Staging is registered per cohort, not pooled across organs."""
    staging = [t for t in TASK_REGISTRY.values() if t.column == "stage_group"]
    assert len(staging) == 2
    cohorts = {t.cohort_filter for t in staging}
    assert cohorts == {("cohort", "TCGA-BRCA"), ("cohort", "TCGA-LUNG")}


def test_cptac_mutation_tasks_are_cohort_confined():
    for name in available_tasks():
        t = TASK_REGISTRY[name]
        if t.source == "cptac_mutation":
            assert t.cohort_filter is not None, f"{name} is not cohort-confined"


def test_mutation_tasks_are_registered_per_cohort_and_gene():
    """12 mutation tasks, each confined to its own cohort.

    TP53 appears in three cohorts, so without the cohort filter those tasks
    would silently pool disjoint diseases into one label set.
    """
    from utils.labels import CPTAC_MUTATION_PANEL

    expected = sum(len(g) for g in CPTAC_MUTATION_PANEL.values())
    mutation_tasks = [
        t for t in TASK_REGISTRY.values() if t.source == "cptac_mutation"
    ]
    assert len(mutation_tasks) == expected == 9
    assert not any("LSCC" in t.cohort_filter[1] for t in mutation_tasks), (
        "CPTAC-LSCC mutation tasks were dropped: only 28 patients have features"
    )

    for t in mutation_tasks:
        assert t.cohort_filter is not None
        assert t.cohort_filter[0] == "collection"
        assert t.classes == ("WT", "MUT")

    tp53 = [t for t in mutation_tasks if t.column == "mut_TP53"]
    assert len({t.cohort_filter[1] for t in tp53}) == len(tp53) == 2


def test_grouped_split_never_splits_a_patient():
    rng = np.random.default_rng(0)
    n = 400
    patients = [f"p{i // 3}" for i in range(n)]  # 3 slides per patient
    labels = pd.DataFrame(
        {"slide_id": [f"s{i}" for i in range(n)],
         "patient_id": patients,
         "label": rng.choice(["A", "B"], size=n)}
    )
    tr, te = grouped_split(labels, test_size=0.25, seed=0)
    assert len(tr) + len(te) == n
    overlap = set(labels.iloc[tr]["patient_id"]) & set(labels.iloc[te]["patient_id"])
    assert overlap == set()


def test_grouped_split_is_stratified():
    rng = np.random.default_rng(1)
    n = 600
    labels = pd.DataFrame(
        {"patient_id": [f"p{i}" for i in range(n)],
         "label": rng.choice(["A", "B"], size=n, p=[0.8, 0.2])}
    )
    tr, te = grouped_split(labels, test_size=0.25, seed=0, stratify=True)
    p_tr = (labels.iloc[tr]["label"] == "B").mean()
    p_te = (labels.iloc[te]["label"] == "B").mean()
    assert abs(p_tr - p_te) < 0.06


def test_grouped_split_rejects_missing_group_column():
    labels = pd.DataFrame({"label": ["A", "B"]})
    with pytest.raises(KeyError, match="patient_id"):
        grouped_split(labels)


# --- bags ----------------------------------------------------------------


@pytest.fixture
def per_encoder():
    rng = np.random.default_rng(0)
    return {
        "a": rng.normal(size=(25, 16)),
        "b": rng.normal(size=(25, 8)),
    }


def test_encode_bag_single(per_encoder):
    out = encode_bag(per_encoder, "single", ["a", "b"])
    assert out.shape == (25, 16)


def test_encode_bag_concat(per_encoder):
    out = encode_bag(per_encoder, "concat", ["a", "b"])
    assert out.shape == (25, 24)


def test_encode_bag_shared_requires_aligner(per_encoder):
    with pytest.raises(ValueError, match="requires a fitted aligner"):
        encode_bag(per_encoder, "shared", ["a", "b"])


def test_encode_bag_rejects_unknown_condition(per_encoder):
    with pytest.raises(ValueError, match="unknown condition"):
        encode_bag(per_encoder, "nope", ["a"])


def test_encode_bag_shared_consensus_and_concat():
    from utils.alignment import build_aligner

    rng = np.random.default_rng(0)
    latent = rng.normal(size=(200, 6))
    views = {
        "a": latent @ rng.normal(size=(6, 16)),
        "b": latent @ rng.normal(size=(6, 8)),
    }
    aligner = build_aligner("joint_pca", latent_dim=6).fit(views)

    one = {k: v[:25] for k, v in views.items()}
    consensus = encode_bag(one, "shared", ["a", "b"], aligner, consensus=True)
    stacked = encode_bag(one, "shared", ["a", "b"], aligner, consensus=False)
    assert consensus.shape == (25, 6)
    assert stacked.shape == (25, 12)


# --- MIL -----------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_bags():
    """Bags whose label is encoded in the mean of the first feature."""
    rng = np.random.default_rng(0)
    bags, y = [], []
    for i in range(80):
        label = i % 2
        n = rng.integers(30, 80)
        x = rng.normal(size=(n, 12))
        x[:, 0] += 3.0 * label
        bags.append(x.astype(np.float32))
        y.append(label)
    return bags, np.array(y)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["abmil", "mean", "transmil"])
def test_mil_learns_a_separable_task(model, toy_bags):
    from utils.mil import MILConfig, evaluate_mil, train_mil

    bags, y = toy_bags
    cfg = MILConfig(model=model, epochs=25, device="cpu", seed=0, patience=8)
    fitted, history = train_mil(bags[:60], y[:60], config=cfg)
    scores = evaluate_mil(fitted, bags[60:], y[60:], device="cpu")
    assert scores["auc"] > 0.8, f"{model} auc={scores['auc']}"
    assert len(history["train_loss"]) > 0


@pytest.mark.slow
def test_mil_evaluate_reports_all_metrics(toy_bags):
    from utils.mil import MILConfig, evaluate_mil, train_mil

    bags, y = toy_bags
    cfg = MILConfig(model="abmil", epochs=5, device="cpu", seed=0)
    fitted, _ = train_mil(bags[:40], y[:40], config=cfg)
    scores = evaluate_mil(fitted, bags[40:], y[40:], device="cpu")
    for key in ("auc", "accuracy", "f1", "balanced_accuracy"):
        assert key in scores and np.isfinite(scores[key])


@pytest.mark.slow
def test_abmil_attention_sums_to_one(toy_bags):
    import torch

    from utils.mil import build_mil_model

    bags, _ = toy_bags
    model = build_mil_model("abmil", in_dim=12, n_classes=2)
    logits, attn = model(torch.as_tensor(bags[0]), return_attention=True)
    assert logits.shape == (1, 2)
    assert attn.shape == (bags[0].shape[0],)
    assert float(attn.sum()) == pytest.approx(1.0, abs=1e-5)


def test_build_mil_model_rejects_unknown():
    from utils.mil import build_mil_model

    with pytest.raises(KeyError, match="unknown MIL model"):
        build_mil_model("nope", 8, 2)


# --- bag loading against a synthetic store -------------------------------


def test_build_bags_from_store(tmp_path):
    """End-to-end: a synthetic feature group plus a label table yields bags."""
    from utils.features import FeatureStore

    rng = np.random.default_rng(0)
    root = tmp_path / "feats"
    grid = root / "cohortA" / "10x_256px_0px_overlap"
    for enc, dim in (("uni_v2", 16), ("conch_v1", 8)):
        d = grid / f"features_{enc}"
        d.mkdir(parents=True)
        for sl in ["s1", "s2", "s3", "s4"]:
            n = 30
            with h5py.File(d / f"{sl}.h5", "w") as h:
                h.create_dataset("features", data=rng.normal(size=(n, dim)))
                h.create_dataset("coords", data=np.zeros((n, 2), dtype=int))

    cfg = root / "e.yaml"
    cfg.write_text(
        f"feature_root: {root}\nencoders:\n"
        "  uni_v2:\n    display_name: UNI\n    dim: 16\n"
        "  conch_v1:\n    display_name: CONCH\n    dim: 8\n"
    )
    group = FeatureStore(config=cfg).group("cohortA/10x_256px")

    labels = pd.DataFrame(
        {
            "slide_id": ["s1", "s2", "s3", "s4", "missing"],
            "label": ["A", "A", "B", "B", "A"],
            "patient_id": ["p1", "p1", "p2", "p3", "p9"],
        }
    )
    bags, y, groups, class_names = build_bags(
        group, labels, condition="concat", max_patches=20
    )
    assert len(bags) == 4  # 'missing' is excluded
    assert bags[0].shape == (20, 24)
    assert class_names == ["A", "B"]
    assert list(groups) == ["p1", "p1", "p2", "p3"]
    assert y.tolist() == [0, 0, 1, 1]


# --- Phase VI: cross-model transfer --------------------------------------


@pytest.fixture(scope="module")
def transfer_setup():
    """Two encoders that are linear views of one latent, plus an aligner."""
    from utils.alignment import build_aligner

    rng = np.random.default_rng(0)
    n, k = 400, 8
    centers = rng.normal(size=(3, k)) * 4
    y = rng.integers(0, 3, size=n)
    latent = centers[y] + rng.normal(size=(n, k)) * 0.4
    views = {
        "a": latent @ rng.normal(size=(k, 24)) + 0.05 * rng.normal(size=(n, 24)),
        "b": latent @ rng.normal(size=(k, 16)) + 0.05 * rng.normal(size=(n, 16)),
    }
    aligner = build_aligner("gcca", latent_dim=k).fit(views)
    return aligner, views, y


def test_transfer_fidelity_perfect_for_identical():
    from utils.transfer import transfer_fidelity

    X = np.random.default_rng(0).normal(size=(100, 12))
    out = transfer_fidelity(X, X)
    assert out["cosine"] == pytest.approx(1.0)
    assert out["r2"] == pytest.approx(1.0)


def test_transfer_fidelity_rejects_shape_mismatch():
    from utils.transfer import transfer_fidelity

    with pytest.raises(ValueError, match="shape mismatch"):
        transfer_fidelity(np.zeros((5, 3)), np.zeros((5, 4)))


def test_translation_recovers_target_features(transfer_setup):
    """A -> B must land near B's real embeddings when they share a latent."""
    from utils.transfer import transfer_fidelity

    aligner, views, _ = transfer_setup
    pred = aligner.translate(views["a"], "a", "b")
    out = transfer_fidelity(pred, views["b"])
    assert out["r2"] > 0.7, out


def test_transfer_retrieval_beats_chance(transfer_setup):
    from utils.transfer import transfer_retrieval

    aligner, views, _ = transfer_setup
    pred = aligner.translate(views["a"], "a", "b")
    out = transfer_retrieval(pred, views["b"], ks=(1,), max_samples=200)
    assert out["recall@1"] > 0.5, out


def test_linear_probe_transfer_reports_ceiling_and_gap(transfer_setup):
    """A probe trained on real B features should still work on translated ones."""
    from utils.transfer import linear_probe_transfer

    aligner, views, y = transfer_setup
    pred = aligner.translate(views["a"], "a", "b")
    out = linear_probe_transfer(pred, views["b"], y, seed=0)

    assert out["probe_true"] > 0.9, "ceiling probe should solve a separable task"
    assert out["probe_translated"] > 0.6, out
    assert out["probe_gap"] == pytest.approx(
        out["probe_true"] - out["probe_translated"]
    )
    assert 0.0 <= out["probe_chance"] <= 1.0


def test_linear_probe_transfer_needs_two_classes(transfer_setup):
    from utils.transfer import linear_probe_transfer

    aligner, views, y = transfer_setup
    pred = aligner.translate(views["a"], "a", "b")
    with pytest.raises(ValueError, match="at least 2 classes"):
        linear_probe_transfer(pred, views["b"], np.zeros(len(y)))


def test_evaluate_transfer_covers_pairs_and_self(transfer_setup):
    from utils.transfer import evaluate_transfer, transfer_summary

    aligner, views, y = transfer_setup
    table = evaluate_transfer(
        aligner, views, labels=y, max_samples=200, seed=0
    )
    # 2 ordered cross pairs + 2 self round-trips
    assert len(table) == 4
    assert table["self"].sum() == 2

    summary = transfer_summary(table)
    assert set(summary.index) == {"cross_model", "self_roundtrip"}
    # Round-tripping a model to itself must not be worse than crossing models.
    assert summary.loc["self_roundtrip", "r2"] >= summary.loc["cross_model", "r2"]


def test_evaluate_transfer_without_labels_omits_probe(transfer_setup):
    from utils.transfer import evaluate_transfer

    aligner, views, _ = transfer_setup
    table = evaluate_transfer(aligner, views, labels=None, max_samples=200)
    assert "probe_true" not in table.columns
    assert "cosine" in table.columns


def test_transferable_pairs_are_within_group(tmp_path):
    """Encoders on different grids must not appear as a transferable pair."""
    from utils.features import FeatureStore
    from utils.transfer import transferable_pairs

    rng = np.random.default_rng(0)
    root = tmp_path / "f"
    layout = {"10x_256px_0px_overlap": ["uni_v2", "conch_v1"],
              "10x_384px_0px_overlap": ["musk"]}
    for grid, encs in layout.items():
        for enc in encs:
            d = root / "c" / grid / f"features_{enc}"
            d.mkdir(parents=True)
            with h5py.File(d / "s1.h5", "w") as h:
                h.create_dataset("features", data=rng.normal(size=(10, 8)))
                h.create_dataset("coords", data=np.zeros((10, 2), dtype=int))

    cfg = root / "e.yaml"
    cfg.write_text(
        f"feature_root: {root}\nencoders:\n"
        "  uni_v2:\n    display_name: UNI\n    dim: 8\n"
        "  conch_v1:\n    display_name: CONCH\n    dim: 8\n"
        "  musk:\n    display_name: MUSK\n    dim: 8\n"
    )
    pairs = transferable_pairs(FeatureStore(config=cfg))
    combos = set(zip(pairs["source"], pairs["target"]))
    assert combos == {("uni_v2", "conch_v1"), ("conch_v1", "uni_v2")}
    assert not any("musk" in c for c in combos), "musk is on a different grid"
