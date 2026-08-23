"""Tests for the magnification ablation.

The load-bearing behaviour is that a magnification series holds the encoder set
and the slide set fixed, so a difference between magnifications can only be
caused by magnification. The synthetic store below deliberately gives one
magnification an extra encoder and one encoder a missing slide, so both
intersections have something real to do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ablation import (  # noqa: E402
    magnification_stability,
    magnification_summary,
    rank_shift_report,
    run_similarity_ablation,
    similarity_trends,
)
from utils.features import FeatureStore  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_magnification_panel,
    plot_magnification_trends,
)

h5py = pytest.importorskip("h5py")

MAGS = [5.0, 10.0, 20.0]
# uni_v2/conch_v1/resnet50 exist at every magnification; ctranspath only at 10x,
# mirroring CONCH being extracted at a single magnification in the real store.
ENCODERS = {
    "uni_v2": 64,
    "conch_v1": 32,
    "resnet50": 48,
}
SLIDES = ["s1", "s2", "s3"]


@pytest.fixture(scope="module")
def series_store(tmp_path_factory):
    """A store with a genuine 3-magnification series plus confounders."""
    root = tmp_path_factory.mktemp("magstore")
    rng = np.random.default_rng(0)

    for mag in MAGS:
        # Patch count scales with magnification, as it does on real slides.
        n = {5.0: 30, 10.0: 60, 20.0: 120}[mag]
        grid = root / "cohortA" / f"{mag:g}x_256px_0px_overlap"

        encoders = dict(ENCODERS)
        if mag == 10.0:
            encoders["ctranspath"] = 24  # present at one magnification only

        latent = {sl: rng.normal(size=(n, 8)) for sl in SLIDES}
        for enc, dim in encoders.items():
            d = grid / f"features_{enc}"
            d.mkdir(parents=True)
            W = rng.normal(size=(8, dim))
            for sl in SLIDES:
                if enc == "resnet50" and sl == "s3" and mag == 20.0:
                    continue  # a slide missing from one encoder
                feats = latent[sl] @ W + 0.1 * rng.normal(size=(n, dim))
                coords = np.stack(
                    [np.arange(n) * int(256 * 40 / mag), np.zeros(n, dtype=int)], axis=1
                )
                with h5py.File(d / f"{sl}.h5", "w") as h:
                    h.create_dataset("features", data=feats)
                    h.create_dataset("coords", data=coords)

    cfg = root / "encoders.yaml"
    lines = [f"feature_root: {root}", "encoders:"]
    for enc in list(ENCODERS) + ["ctranspath"]:
        lines += [f"  {enc}:", f"    display_name: {enc.upper()}", "    dim: 1"]
    cfg.write_text("\n".join(lines) + "\n")
    return FeatureStore(config=cfg)


@pytest.fixture(scope="module")
def series(series_store):
    return series_store.best_series()


def test_series_spans_all_magnifications(series):
    assert series.magnifications == MAGS
    assert series.patch_size == 256
    assert series.key == "cohortA/256px"


def test_series_intersects_encoders_across_magnifications(series):
    """ctranspath exists only at 10x, so it must be excluded from the series."""
    assert series.encoders == sorted(ENCODERS)
    assert "ctranspath" not in series.encoders


def test_series_shared_slides_exclude_partial_coverage(series):
    """resnet50 is missing s3 at 20x, so the series can only use s1 and s2."""
    assert series.shared_slides() == ["s1", "s2"]


def test_series_sample_shapes_and_pairing(series):
    views = series.sample(n_patches=40, seed=0)
    assert sorted(views) == MAGS
    for mag, per_enc in views.items():
        assert set(per_enc) == set(series.encoders)
        counts = {v.shape[0] for v in per_enc.values()}
        assert len(counts) == 1, f"{mag}x views must be row-paired"
        assert per_enc["uni_v2"].shape[1] == ENCODERS["uni_v2"]


def test_series_sample_uses_the_same_slides_everywhere(series):
    """Only the grid should differ between magnifications, not the tissue."""
    views = series.sample(n_patches=20, seed=1, max_slides=1)
    assert all(v["uni_v2"].shape[0] > 0 for v in views.values())


def test_magnification_series_discovery_ranking(series_store):
    found = series_store.magnification_series()
    assert len(found) == 1
    assert found[0].key == "cohortA/256px"


def test_min_encoders_filter(series_store):
    assert series_store.magnification_series(min_encoders=99) == []


# --- ablation outputs ----------------------------------------------------


@pytest.fixture(scope="module")
def results(series):
    res, _ = run_similarity_ablation(
        series,
        metrics=["linear_cka", "procrustes"],
        n_patches=40,
        max_slides=None,
        max_samples=40,
        verbose=False,
    )
    return res


def test_ablation_runs_at_every_magnification(results, series):
    assert sorted(results) == MAGS
    for mats in results.values():
        assert set(mats) == {"linear_cka", "procrustes"}
        for S in mats.values():
            assert S.shape == (len(series.encoders), len(series.encoders))
            assert np.allclose(np.diag(S.values), 1.0, atol=1e-6)


def test_similarity_trends_long_format(results, series):
    long = similarity_trends(results)
    n_pairs = len(series.encoders) * (len(series.encoders) - 1) // 2
    assert len(long) == len(MAGS) * 2 * n_pairs
    assert set(long.columns) == {
        "magnification",
        "metric",
        "model_a",
        "model_b",
        "pair",
        "similarity",
    }


def test_magnification_summary_shape(results):
    summary = magnification_summary(results)
    assert list(summary.index) == MAGS
    assert "linear_cka" in summary.columns
    assert "spread" in summary.columns
    assert summary["linear_cka"].between(-1, 1).all()


def test_magnification_stability_is_symmetric_with_unit_diagonal(results):
    stab = magnification_stability(results, metric="linear_cka")
    assert stab.shape == (3, 3)
    assert list(stab.index) == ["5x", "10x", "20x"]
    assert np.allclose(stab.values, stab.values.T)
    assert np.allclose(np.diag(stab.values), 1.0)


def test_magnification_stability_rejects_missing_metric(results):
    with pytest.raises(KeyError, match="missing at magnifications"):
        magnification_stability(results, metric="not_a_metric")


def test_rank_shift_report_columns(results, series):
    report = rank_shift_report(results, metric="linear_cka")
    n_pairs = len(series.encoders) * (len(series.encoders) - 1) // 2
    assert len(report) == n_pairs
    for col in ["5x", "10x", "20x", "delta", "range", "rank_change"]:
        assert col in report.columns
    # delta must equal the highest-magnification minus lowest-magnification value
    assert np.allclose(report["delta"], report["20x"] - report["5x"])
    # sorted by range, descending
    assert report["range"].is_monotonic_decreasing


def test_rank_shift_range_is_non_negative(results):
    report = rank_shift_report(results, metric="linear_cka")
    assert (report["range"] >= 0).all()


def test_plots_render(results):
    import matplotlib

    matplotlib.use("Agg")
    trends = similarity_trends(results)
    fig, ax = plot_magnification_trends(trends, metric="linear_cka")
    assert ax.get_xlabel() == "magnification"
    fig2 = plot_magnification_panel(results, metric="linear_cka")
    assert len(fig2.axes) >= len(MAGS)


def test_plot_rejects_unknown_metric(results):
    trends = similarity_trends(results)
    with pytest.raises(ValueError, match="no rows for metric"):
        plot_magnification_trends(trends, metric="nope")


def test_trends_recover_planted_ordering(results):
    """Sanity check: the ablation preserves that all views share one latent.

    Every encoder here is a linear view of the same latent factor, so pairwise
    similarity should be high at every magnification. A regression that broke
    pairing would show up as similarity collapsing towards zero.
    """
    long = similarity_trends(results)
    cka = long[long["metric"] == "linear_cka"]
    assert (cka.groupby("magnification")["similarity"].mean() > 0.5).all()


def test_alignment_ablation_runs(series):
    from utils.ablation import run_alignment_ablation

    views = series.sample(n_patches=120, seed=0)
    out = run_alignment_ablation(
        views,
        methods=["joint_pca"],
        latent_dim=6,
        seed=0,
        verbose=False,
        retrieval_samples=40,
    )
    assert isinstance(out, pd.DataFrame)
    assert list(out.index.names) == ["magnification", "method"]
    assert len(out) == len(MAGS)
    assert "recall@1" in out.columns
