"""Property-based sanity checks for the Phase I similarity metrics.

These are not accuracy tests against a reference implementation; they check the
mathematical invariants each metric is supposed to satisfy. If one of these
fails, the corresponding column of the similarity matrix is not measuring what
the paper will claim it measures.

Run with::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (  # noqa: E402
    apply_procrustes_transform,
    cka_from_grams,
    compute_all_similarity_matrices,
    compute_similarity_matrix,
    cosine_rsa_similarity,
    distance_correlation,
    fit_procrustes_transform,
    gram_linear,
    hierarchical_linkage,
    kernel_cka,
    linear_cka,
    mds_embedding,
    mean_cosine_similarity,
    orthogonal_procrustes_distance,
    orthogonal_procrustes_similarity,
    pwcca,
    similarity_to_distance,
    stack_similarity_matrices,
    svcca,
)

N, D = 400, 24

#: Metrics expected to score 1.0 for a representation compared with itself.
IDENTITY_METRICS = [
    linear_cka,
    kernel_cka,
    svcca,
    pwcca,
    orthogonal_procrustes_similarity,
    cosine_rsa_similarity,
    distance_correlation,
]

#: Metrics invariant to orthogonal transforms and isotropic scaling.
INVARIANT_METRICS = IDENTITY_METRICS


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def X(rng):
    return rng.normal(size=(N, D))


@pytest.fixture(scope="module")
def rotation(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    return Q


@pytest.mark.parametrize("metric", IDENTITY_METRICS, ids=lambda f: f.__name__)
def test_self_similarity_is_one(metric, X):
    assert metric(X, X) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("metric", INVARIANT_METRICS, ids=lambda f: f.__name__)
def test_rotation_and_scale_invariance(metric, X, rotation):
    assert metric(X, 3.7 * X @ rotation) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("metric", INVARIANT_METRICS, ids=lambda f: f.__name__)
def test_translation_invariance(metric, X, rng):
    shift = rng.normal(size=(1, D)) * 10.0
    assert metric(X, X + shift) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("metric", INVARIANT_METRICS, ids=lambda f: f.__name__)
def test_symmetry(metric, X, rng):
    Y = rng.normal(size=(N, D + 7))
    assert metric(X, Y) == pytest.approx(metric(Y, X), abs=1e-8)


@pytest.mark.parametrize("metric", INVARIANT_METRICS, ids=lambda f: f.__name__)
def test_independent_representations_score_low(metric, X, rng):
    """Independent Gaussians should be far from 1, and never negative."""
    Y = rng.normal(size=(N, D))
    s = metric(X, Y)
    assert -0.05 <= s < 0.55, f"{metric.__name__} gave {s} for independent data"


@pytest.mark.parametrize("metric", INVARIANT_METRICS, ids=lambda f: f.__name__)
def test_shared_signal_beats_independent(metric, X, rng):
    """A representation sharing half its subspace must outscore an unrelated one."""
    half = D // 2
    shared = np.hstack([X[:, :half], rng.normal(size=(N, half))])
    unrelated = rng.normal(size=(N, D))
    assert metric(X, shared) > metric(X, unrelated)


def test_float32_input_is_accepted(X, rng):
    """Stored features are float32; converting them must not raise.

    Under NumPy 2, ``np.array(x, copy=False)`` raises whenever a dtype
    conversion forces a copy, which is exactly this case.
    """
    from utils import as_matrix

    X32 = X.astype(np.float32)
    assert as_matrix(X32, copy=False).dtype == np.float64
    assert linear_cka(X32, X32) == pytest.approx(1.0, abs=1e-6)


def test_handles_different_feature_dims(X, rng):
    Y = rng.normal(size=(N, 3 * D))
    for metric in INVARIANT_METRICS:
        s = metric(X, Y)
        assert np.isfinite(s), f"{metric.__name__} returned {s}"


# --- CKA specifics -------------------------------------------------------


def test_linear_cka_feature_and_gram_forms_agree(X, rng):
    Y = rng.normal(size=(N, D + 5))
    feature_form = linear_cka(X, Y)
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    gram_form = cka_from_grams(gram_linear(Xc), gram_linear(Yc))
    assert feature_form == pytest.approx(gram_form, abs=1e-9)


def test_unbiased_cka_less_inflated_on_independent_data(rng):
    """The unbiased estimator should sit closer to 0 for independent data."""
    A = rng.normal(size=(200, 64))
    B = rng.normal(size=(200, 64))
    assert linear_cka(A, B, unbiased=True) < linear_cka(A, B, unbiased=False)


def test_kernel_cka_detects_nonlinear_structure(rng):
    """A nonlinear reparameterisation linear CKA under-scores, kernel CKA sees."""
    t = rng.uniform(-np.pi, np.pi, size=(600, 1))
    X = np.hstack([t, rng.normal(scale=0.01, size=(600, 1))])
    Y = np.hstack([np.sin(t), np.cos(t)])
    assert kernel_cka(X, Y, threshold=0.4) > linear_cka(X, Y)


# --- CCA specifics -------------------------------------------------------


def test_svcca_truncation_reduces_noise_inflation(rng):
    """Adding pure-noise directions should hurt untruncated CCA more."""
    signal = rng.normal(size=(N, 6))
    X = np.hstack([signal, 0.001 * rng.normal(size=(N, 30))])
    Y = np.hstack([signal, 0.001 * rng.normal(size=(N, 30))])
    from utils import mean_cca_correlation

    assert svcca(X, Y, var_threshold=0.99) > mean_cca_correlation(X, Y)


def test_pwcca_asymmetric_mode_differs_by_direction(rng):
    """The weighting is taken from the first argument, so direction matters.

    The shared block dominates X's variance but is a rounding error in Y's, so
    the correlated canonical directions carry heavy weight from X's side and
    almost none from Y's.
    """
    shared = rng.normal(size=(N, 4))
    X = np.hstack([10.0 * shared, 0.1 * rng.normal(size=(N, 4))])
    Y = np.hstack([0.1 * shared, 10.0 * rng.normal(size=(N, 4))])

    fwd = pwcca(X, Y, symmetric=False)
    bwd = pwcca(Y, X, symmetric=False)
    assert fwd > bwd + 0.05
    assert pwcca(X, Y) == pytest.approx((fwd + bwd) / 2, abs=1e-8)


# --- Procrustes specifics ------------------------------------------------


def test_procrustes_distance_matches_similarity(X, rng):
    Y = rng.normal(size=(N, D))
    s = orthogonal_procrustes_similarity(X, Y)
    d = orthogonal_procrustes_distance(X, Y)
    assert d == pytest.approx(np.sqrt(2 - 2 * s), abs=1e-9)


def test_procrustes_transform_recovers_known_rotation(X, rotation):
    Y = X @ rotation.T  # so that Y @ rotation == X
    fit = fit_procrustes_transform(X, Y)
    assert fit["disparity"] == pytest.approx(0.0, abs=1e-12)
    recovered = apply_procrustes_transform(Y, fit)
    assert np.allclose(recovered, X, atol=1e-8)


def test_procrustes_transform_across_dims(X, rng):
    """Fitting from a wider source into a narrower target returns target width."""
    Y = rng.normal(size=(N, D + 11))
    fit = fit_procrustes_transform(X, Y)
    mapped = apply_procrustes_transform(Y, fit)
    assert mapped.shape == X.shape


# --- Distance correlation ------------------------------------------------


def test_distance_correlation_near_zero_for_independent(rng):
    A = rng.normal(size=(300, 5))
    B = rng.normal(size=(300, 5))
    assert distance_correlation(A, B, unbiased=True) < 0.15


def test_distance_correlation_detects_nonlinear_dependence(rng):
    t = rng.uniform(-1, 1, size=(400, 1))
    assert distance_correlation(t, t**2, unbiased=True) > 0.25


# --- Cosine / RSA --------------------------------------------------------


def test_mean_cosine_requires_matching_dims(X, rng):
    with pytest.raises(ValueError, match="shared feature space"):
        mean_cosine_similarity(X, rng.normal(size=(N, D + 1)))


def test_mean_cosine_of_identical_is_one(X):
    assert mean_cosine_similarity(X, X) == pytest.approx(1.0, abs=1e-10)


# --- Driver and outputs --------------------------------------------------


@pytest.fixture(scope="module")
def representations(rng):
    """Three models: two sharing a latent subspace, one independent."""
    latent = rng.normal(size=(N, 10))
    Wa, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    return {
        "model_a": latent @ rng.normal(size=(10, 32)),
        "model_b": (latent @ Wa) @ rng.normal(size=(10, 48)),
        "model_c": rng.normal(size=(N, 40)),
    }


def test_similarity_matrix_is_square_and_labelled(representations):
    S = compute_similarity_matrix(representations, "linear_cka")
    assert S.shape == (3, 3)
    assert list(S.index) == list(representations)
    assert np.allclose(np.diag(S.values), 1.0, atol=1e-6)
    assert np.allclose(S.values, S.values.T, atol=1e-10)


def test_similarity_matrix_recovers_planted_structure(representations):
    S = compute_similarity_matrix(representations, "linear_cka")
    assert S.loc["model_a", "model_b"] > S.loc["model_a", "model_c"]
    assert S.loc["model_a", "model_b"] > S.loc["model_b", "model_c"]


def test_all_metrics_run_and_agree_on_structure(representations):
    mats = compute_all_similarity_matrices(representations, max_samples=200)
    assert len(mats) == 7
    for name, S in mats.items():
        assert S.shape == (3, 3), name
        assert np.isfinite(S.values).all(), name
        assert S.loc["model_a", "model_b"] > S.loc["model_a", "model_c"], name


def test_mismatched_sample_counts_are_rejected(rng):
    bad = {"a": rng.normal(size=(50, 8)), "b": rng.normal(size=(60, 8))}
    with pytest.raises(ValueError, match="same patches"):
        compute_similarity_matrix(bad, "linear_cka")


def test_stack_similarity_matrices(representations):
    mats = compute_all_similarity_matrices(representations, max_samples=150)
    long = stack_similarity_matrices(mats)
    assert len(long) == 3  # 3 unordered pairs
    assert {"model_a", "model_b", "model_c"} >= set(long["model_a"])


@pytest.mark.parametrize("mode", ["one_minus", "angular", "sqrt_one_minus"])
def test_similarity_to_distance_properties(representations, mode):
    S = compute_similarity_matrix(representations, "linear_cka")
    Dm = similarity_to_distance(S, mode=mode)
    arr = Dm.values
    assert np.allclose(np.diag(arr), 0.0)
    assert np.allclose(arr, arr.T)
    assert (arr >= 0).all()


def test_linkage_and_mds_run(representations):
    S = compute_similarity_matrix(representations, "linear_cka")
    Z = hierarchical_linkage(S)
    assert Z.shape == (2, 4)
    coords, stress = mds_embedding(S)
    assert coords.shape == (3, 2)
    assert stress >= 0


def test_subsampling_is_shared_across_models(representations):
    """Independent subsampling would break pairing and destroy the signal."""
    S = compute_similarity_matrix(representations, "linear_cka", max_samples=100)
    assert S.loc["model_a", "model_b"] > 0.5
