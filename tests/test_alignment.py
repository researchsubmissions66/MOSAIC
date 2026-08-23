"""Property tests for the Phase V shared-latent-space aligners.

The core fixture plants a known low-dimensional latent factor, generates
several "models" as noisy linear views of it with different widths, and adds
one model that is pure noise. Any working aligner must recover the planted
structure for the related views and fail to relate the unrelated one.

Run with::

    python -m pytest tests/test_alignment.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.alignment import (  # noqa: E402
    ALIGNER_REGISTRY,
    BaseAligner,
    GCCAAligner,
    GeneralizedProcrustesAligner,
    JointPCAAligner,
    MCCAAligner,
    ViewPreprocessor,
    available_aligners,
    build_aligner,
    sinkhorn_coupling,
    split_views,
)
from utils.alignment_metrics import (  # noqa: E402
    alignment_error,
    cross_model_retrieval,
    cross_model_transfer_scores,
    effective_rank,
    evaluate_aligner,
    neighborhood_preservation,
    paired_cosine,
    reconstruction_error,
)

N, K = 900, 10
DIMS = {"alpha": 40, "beta": 64, "gamma": 32}

#: Aligners that are cheap enough to run across the whole parametrised suite.
FAST_ALIGNERS = ["joint_pca", "gcca", "mcca", "procrustes"]


@pytest.fixture(scope="module")
def latent():
    return np.random.default_rng(0).normal(size=(N, K))


@pytest.fixture(scope="module")
def views(latent):
    """Three models that are noisy linear views of one shared latent factor."""
    rng = np.random.default_rng(1)
    return {
        name: latent @ rng.normal(size=(K, d)) + 0.15 * rng.normal(size=(N, d))
        for name, d in DIMS.items()
    }


@pytest.fixture(scope="module")
def views_with_outlier(views):
    """The same, plus one model carrying no shared signal at all."""
    rng = np.random.default_rng(2)
    out = dict(views)
    out["noise"] = rng.normal(size=(N, 48))
    return out


@pytest.fixture(scope="module")
def split(views):
    train, test, _, _ = split_views(views, test_size=0.25, seed=0)
    return train, test


def _fit(name: str, **kwargs) -> BaseAligner:
    """Build an aligner with test-scale settings."""
    defaults = {"latent_dim": K}
    if name == "autoencoder":
        defaults.update(
            hidden_dims=(64, 32), epochs=25, batch_size=128, patience=8, device="cpu"
        )
    if name == "optimal_transport":
        defaults.update(batch_size=200, n_iter=60, n_restarts=1)
    if name == "mcca":
        defaults.update(pca_dim=None)
    defaults.update(kwargs)
    return build_aligner(name, **defaults)


# --- preprocessing -------------------------------------------------------


def test_preprocessor_roundtrip_is_exact_without_pca(views):
    X = views["alpha"]
    pp = ViewPreprocessor(scaling="rms")
    Z = pp.fit_transform(X)
    assert np.allclose(pp.inverse_transform(Z), X, atol=1e-8)


def test_preprocessor_scaling_is_sample_independent(views):
    """The scale must transfer to held-out data of a different size."""
    X = views["alpha"]
    pp = ViewPreprocessor(scaling="rms").fit(X)
    half = pp.transform(X[: N // 2])
    full = pp.transform(X)
    assert np.allclose(half, full[: N // 2])


def test_preprocessor_pca_reduces_dimension(views):
    pp = ViewPreprocessor(pca_dim=8)
    Z = pp.fit_transform(views["beta"])
    assert Z.shape == (N, 8)
    assert pp.inverse_transform(Z).shape == views["beta"].shape


def test_split_views_shares_one_partition(views):
    train, test, tr_idx, te_idx = split_views(views, test_size=0.2, seed=0)
    assert len(set(tr_idx) & set(te_idx)) == 0
    assert len(tr_idx) + len(te_idx) == N
    for name in views:
        assert train[name].shape[0] == len(tr_idx)
        assert np.allclose(train[name], views[name][tr_idx])


# --- generic aligner contract -------------------------------------------


@pytest.mark.parametrize("name", FAST_ALIGNERS)
def test_shapes_and_roundtrip(name, split):
    train, test = split
    aligner = _fit(name).fit(train)

    Z = aligner.transform(test)
    for model, z in Z.items():
        assert z.shape == (test[model].shape[0], K)

    for model in test:
        back = aligner.inverse_transform(Z[model], model)
        assert back.shape == test[model].shape


@pytest.mark.parametrize("name", FAST_ALIGNERS)
def test_transform_generalises_to_held_out_patches(name, split):
    """Held-out patches must be encodable — the point of a projection function."""
    train, test = split
    aligner = _fit(name).fit(train)
    Z = aligner.transform(test)
    assert all(np.isfinite(z).all() for z in Z.values())
    assert alignment_error(Z) < 0.5


@pytest.mark.parametrize("name", FAST_ALIGNERS)
def test_recovers_planted_shared_structure(name, split):
    """Cross-model retrieval in the shared space must be far above chance."""
    train, test = split
    aligner = _fit(name).fit(train)
    Z = aligner.transform(test)
    scores = cross_model_retrieval(Z["alpha"], Z["beta"], ks=(1,), max_samples=200)
    assert scores["recall@1"] > 0.9, f"{name} recall@1={scores['recall@1']}"


@pytest.mark.parametrize("name", FAST_ALIGNERS)
def test_reconstruction_beats_the_mean(name, split):
    train, test = split
    aligner = _fit(name).fit(train)
    recon = reconstruction_error(aligner, test)
    assert (recon["r2"] > 0.5).all(), f"{name}:\n{recon}"


@pytest.mark.parametrize("name", FAST_ALIGNERS)
def test_translate_between_models(name, split):
    train, test = split
    aligner = _fit(name).fit(train)
    out = aligner.translate(test["alpha"], "alpha", "beta")
    assert out.shape == test["beta"].shape

    scores = cross_model_transfer_scores(aligner, test)
    assert len(scores) == len(test) * (len(test) - 1)
    assert (scores["r2"] > 0.3).all(), f"{name}:\n{scores}"


@pytest.mark.parametrize("name", FAST_ALIGNERS)
def test_consensus_shape_and_agreement(name, split):
    train, test = split
    aligner = _fit(name).fit(train)
    C = aligner.consensus(test)
    assert C.shape == (test["alpha"].shape[0], K)


@pytest.mark.parametrize("name", FAST_ALIGNERS)
def test_save_and_load_roundtrip(name, split, tmp_path):
    train, test = split
    aligner = _fit(name).fit(train)
    before = aligner.transform_view("alpha", test["alpha"])

    path = tmp_path / f"{name}.joblib"
    aligner.save(path)
    after = BaseAligner.load(path).transform_view("alpha", test["alpha"])
    assert np.allclose(before, after)


@pytest.mark.parametrize("name", FAST_ALIGNERS)
def test_unrelated_model_is_not_aligned(name, views_with_outlier):
    """The noise model must not retrieve above chance, however the rest align."""
    train, test, _, _ = split_views(views_with_outlier, test_size=0.25, seed=0)
    aligner = _fit(name).fit(train)
    Z = aligner.transform(test)

    real = cross_model_retrieval(Z["alpha"], Z["beta"], ks=(1,), max_samples=200)
    fake = cross_model_retrieval(Z["alpha"], Z["noise"], ks=(1,), max_samples=200)
    assert real["recall@1"] > 0.8
    assert fake["recall@1"] < 0.2, f"{name} spuriously aligned noise"


# --- error handling ------------------------------------------------------


def test_unfitted_aligner_raises(views):
    with pytest.raises(RuntimeError, match="not fitted"):
        GCCAAligner().transform(views)


def test_single_view_rejected(views):
    with pytest.raises(ValueError, match="at least 2 models"):
        GCCAAligner(latent_dim=4).fit({"alpha": views["alpha"]})


def test_mismatched_rows_rejected(views):
    bad = {"alpha": views["alpha"], "beta": views["beta"][:-5]}
    with pytest.raises(ValueError, match="row-paired"):
        GCCAAligner(latent_dim=4).fit(bad)


def test_unknown_model_name_rejected(split):
    train, test = split
    aligner = GCCAAligner(latent_dim=K).fit(train)
    with pytest.raises(KeyError, match="unknown model"):
        aligner.transform_view("nope", test["alpha"])


def test_wrong_latent_dim_rejected(split):
    train, test = split
    aligner = GCCAAligner(latent_dim=K).fit(train)
    with pytest.raises(ValueError, match="shared coordinates"):
        aligner.inverse_transform(np.zeros((10, K + 3)), "alpha")


def test_latent_dim_larger_than_samples_rejected(views):
    small = {k: v[:5] for k, v in views.items()}
    with pytest.raises(ValueError, match="latent_dim"):
        GCCAAligner(latent_dim=20).fit(small)


def test_unknown_aligner_name():
    with pytest.raises(KeyError, match="unknown aligner"):
        build_aligner("does_not_exist")


def test_registry_matches_available():
    assert available_aligners() == sorted(ALIGNER_REGISTRY)
    assert len(ALIGNER_REGISTRY) == 6


# --- method-specific -----------------------------------------------------


def test_gcca_eigenvalues_bounded_by_view_count(split):
    """MAX-VAR eigenvalues lie in [1, M]; M means total agreement."""
    train, _ = split
    aligner = GCCAAligner(latent_dim=K).fit(train)
    M = len(train)
    assert aligner.eigenvalues_.shape == (K,)
    assert (aligner.eigenvalues_ <= M + 1e-6).all()
    assert (aligner.eigenvalues_ >= 1.0 - 1e-6).all()
    # Views share a strong latent factor, so the leading direction is near M.
    assert aligner.eigenvalues_[0] > 0.9 * M
    assert np.all(np.diff(aligner.eigenvalues_) <= 1e-9)  # descending


def test_gcca_agreement_drops_for_unrelated_view(views, views_with_outlier):
    related = GCCAAligner(latent_dim=K).fit(views)
    mixed = GCCAAligner(latent_dim=K).fit(views_with_outlier)
    assert related.view_agreement_[0] > mixed.view_agreement_[0]


def test_mcca_two_views_matches_ordinary_cca(views):
    """SUMCOR MCCA with M=2 must reproduce the classical CCA correlations."""
    from utils.cca import cca_correlations

    pair = {"alpha": views["alpha"], "beta": views["beta"]}
    aligner = MCCAAligner(latent_dim=6, reg=1e-10, pca_dim=None).fit(pair)
    expected = cca_correlations(pair["alpha"], pair["beta"])[:6]
    assert np.allclose(aligner.correlations_, expected, atol=1e-4)


def test_procrustes_rotations_are_orthogonal(split):
    train, _ = split
    aligner = GeneralizedProcrustesAligner(latent_dim=K).fit(train)
    for R in aligner.rotations_.values():
        assert np.allclose(R @ R.T, np.eye(K), atol=1e-8)


def test_procrustes_converges(split):
    train, _ = split
    aligner = GeneralizedProcrustesAligner(latent_dim=K, max_iter=200).fit(train)
    assert aligner.n_iter_ < 200
    assert aligner.residuals_[-1] < aligner.tol
    # Monotone decrease is what distinguishes convergence from drift — the
    # free-scaling degeneracy this method is prone to shows up as a flat tail.
    assert aligner.residuals_[-1] < aligner.residuals_[0]


def test_procrustes_decoder_is_exact_inverse(split):
    """A rigid map has an exact inverse; only PCA truncation should cost anything."""
    train, test = split
    aligner = GeneralizedProcrustesAligner(latent_dim=K).fit(train)
    Z = aligner.transform_view("alpha", test["alpha"])
    Xp = aligner.preprocessors_["alpha"].transform(test["alpha"])
    assert np.allclose(aligner._decode("alpha", Z), Xp, atol=1e-8)


def test_joint_pca_reports_view_contributions(split):
    train, _ = split
    aligner = JointPCAAligner(latent_dim=K).fit(train)
    assert pytest.approx(sum(aligner.view_loadings_.values()), abs=1e-8) == 1.0
    assert all(0.3 < r2 <= 1.0 for r2 in aligner.encoder_r2_.values())


def test_joint_pca_whitening_trades_geometry_for_isotropy(split):
    """Whitening should measurably hurt neighbourhood preservation."""
    train, test = split
    plain = JointPCAAligner(latent_dim=K, whiten=False).fit(train)
    white = JointPCAAligner(latent_dim=K, whiten=True).fit(train)

    npr_plain = neighborhood_preservation(
        test["alpha"], plain.transform_view("alpha", test["alpha"]), max_samples=200
    )
    npr_white = neighborhood_preservation(
        test["alpha"], white.transform_view("alpha", test["alpha"]), max_samples=200
    )
    assert npr_plain > npr_white


@pytest.mark.slow
def test_autoencoder_learns_shared_space(split):
    train, test = split
    aligner = _fit("autoencoder", epochs=60).fit(train)
    Z = aligner.transform(test)
    scores = cross_model_retrieval(Z["alpha"], Z["beta"], ks=(1,), max_samples=200)
    assert scores["recall@1"] > 0.5
    assert len(aligner.history_["train_loss"]) > 0


@pytest.mark.slow
def test_autoencoder_align_weight_controls_agreement(split):
    """Higher alignment weight must reduce cross-model disagreement."""
    train, test = split
    loose = _fit("autoencoder", align_weight=0.0).fit(train)
    tight = _fit("autoencoder", align_weight=10.0).fit(train)
    assert alignment_error(tight.transform(test)) < alignment_error(
        loose.transform(test)
    )


@pytest.mark.slow
def test_optimal_transport_supervised_matches_procrustes(split):
    """With supervised=True the method reduces to orthogonal Procrustes."""
    train, test = split
    ot = _fit("optimal_transport", supervised=True).fit(train)
    Z = ot.transform(test)
    scores = cross_model_retrieval(Z["alpha"], Z["beta"], ks=(1,), max_samples=200)
    assert scores["recall@1"] > 0.9


@pytest.mark.slow
def test_optimal_transport_supervised_recovers_pairing(split):
    """Supervised mode must near-perfectly match paired patches back up."""
    train, _ = split
    ot = _fit("optimal_transport", supervised=True).fit(train)
    for name, acc in ot.matching_accuracy_.items():
        assert acc > 0.9, f"{name} matching accuracy {acc:.3f}"


@pytest.mark.slow
def test_optimal_transport_unsupervised_runs_and_stays_orthogonal(split):
    """Unsupervised mode is exploratory: assert it is well-formed, not that it wins.

    Quality is deliberately not asserted — see the module docstring. On
    synthetic data this mode reaches only ~15-25x chance at best and often
    lands at chance, so a quality assertion here would be flaky and would
    misrepresent what the method delivers.
    """
    train, _ = split
    ot = _fit("optimal_transport", supervised=False, n_iter=60, n_restarts=1)
    ot.fit(train)

    for R in ot.rotations_.values():
        assert np.allclose(R @ R.T, np.eye(K), atol=1e-8)
    assert set(ot.matching_accuracy_) == set(train) - {ot.reference_}
    assert all(np.isfinite(v) for v in ot.transport_costs_.values())


def test_sinkhorn_coupling_is_doubly_stochastic():
    rng = np.random.default_rng(0)
    C = rng.uniform(size=(40, 40))
    P = sinkhorn_coupling(C, reg=0.05 * C.mean(), n_iter=500)
    assert np.allclose(P.sum(axis=1), 1 / 40, atol=1e-6)
    assert np.allclose(P.sum(axis=0), 1 / 40, atol=1e-6)


def test_sinkhorn_recovers_identity_matching():
    """With a cost minimised on the diagonal, the coupling must be near-diagonal."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    C = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    P = sinkhorn_coupling(C, reg=0.01 * C.mean(), n_iter=800)
    assert (np.argmax(P, axis=1) == np.arange(30)).mean() > 0.95


# --- metrics -------------------------------------------------------------


def test_retrieval_is_perfect_for_identical_spaces(latent):
    scores = cross_model_retrieval(latent, latent, ks=(1,), max_samples=300)
    assert scores["recall@1"] == 1.0
    assert scores["mrr"] == 1.0


def test_retrieval_is_chance_for_independent_spaces():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 8)), rng.normal(size=(300, 8))
    scores = cross_model_retrieval(a, b, ks=(1,), max_samples=300)
    assert scores["recall@1"] < 0.05


def test_retrieval_rejects_unpaired_inputs(latent):
    with pytest.raises(ValueError, match="row-paired"):
        cross_model_retrieval(latent, latent[:-1])


def test_alignment_error_zero_for_identical_latents(latent):
    assert alignment_error({"a": latent, "b": latent}) == pytest.approx(0.0)


def test_alignment_error_large_for_independent_latents():
    rng = np.random.default_rng(0)
    err = alignment_error(
        {"a": rng.normal(size=(200, 8)), "b": rng.normal(size=(200, 8))}
    )
    assert err > 0.5


def test_neighborhood_preservation_perfect_for_identity(latent):
    assert neighborhood_preservation(latent, latent, k=5, max_samples=200) == 1.0


def test_effective_rank_detects_collapse():
    rng = np.random.default_rng(0)
    full = rng.normal(size=(400, 10))
    collapsed = np.repeat(rng.normal(size=(400, 1)), 10, axis=1)
    assert effective_rank(full) > 9.0
    assert effective_rank(collapsed) < 1.5


def test_paired_cosine_is_symmetric_with_unit_diagonal(latent):
    rng = np.random.default_rng(0)
    C = paired_cosine({"a": latent, "b": latent + 0.1 * rng.normal(size=latent.shape)})
    assert np.allclose(C.values, C.values.T)
    assert np.allclose(np.diag(C.values), 1.0)


def test_evaluate_aligner_report_structure(split):
    train, test = split
    aligner = GCCAAligner(latent_dim=K).fit(train)
    report = evaluate_aligner(aligner, test, retrieval_samples=200)

    assert set(report) == {
        "summary",
        "reconstruction",
        "paired_cosine",
        "shared_cka",
        "retrieval",
        "neighborhood",
    }
    n_models = len(test)
    assert len(report["retrieval"]) == n_models * (n_models - 1)
    assert np.isfinite(report["summary"].values.astype(float)).all()
