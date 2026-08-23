"""Tests for the feature store: config parsing, discovery, and pairing.

Split in two. The first half builds a fake trident-shaped directory tree in a
tmpdir and exercises discovery/loading without touching the real store, so it
runs anywhere. The second half runs against the real feature store and skips
cleanly when it is absent (e.g. on a machine where the data is not mounted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.features import (  # noqa: E402
    DEFAULT_CONFIG,
    FeatureStore,
    load_encoder_config,
)

h5py = pytest.importorskip("h5py")

REAL_ROOT, _ = load_encoder_config(DEFAULT_CONFIG)
requires_real_store = pytest.mark.skipif(
    not REAL_ROOT.exists(), reason=f"feature store {REAL_ROOT} not available"
)


# --- synthetic store -----------------------------------------------------


@pytest.fixture(scope="module")
def fake_store(tmp_path_factory):
    """A miniature trident-shaped tree with two grids and a known outlier.

    ``20x_256px`` holds three encoders on three slides (fully paired).
    ``20x_224px`` holds two encoders on a *different* patch grid, and one of
    them is missing a slide — so shared-slide logic has something to exclude.
    """
    root = tmp_path_factory.mktemp("features")
    rng = np.random.default_rng(0)

    layout = {
        "20x_256px_0px_overlap": {
            "uni_v2": (1536, ["s1", "s2", "s3"]),
            "conch_v1": (512, ["s1", "s2", "s3"]),
            "resnet50": (1024, ["s1", "s2", "s3"]),
        },
        "20x_224px_0px_overlap": {
            "virchow": (2560, ["s1", "s2"]),
            "gpfm": (1024, ["s1"]),
        },
    }
    n_patches = {"s1": 40, "s2": 25, "s3": 30}

    for grid, encoders in layout.items():
        patch = int(grid.split("_")[1].replace("px", ""))
        for enc, (dim, slides) in encoders.items():
            d = root / "cohortA" / grid / f"features_{enc}"
            d.mkdir(parents=True)
            for sl in slides:
                n = n_patches[sl]
                # Coordinates depend only on the grid, exactly as trident does,
                # so encoders within a grid are paired and across grids are not.
                coords = np.stack(
                    [np.arange(n) * patch, np.zeros(n, dtype=int)], axis=1
                )
                with h5py.File(d / f"{sl}.h5", "w") as h:
                    h.create_dataset("features", data=rng.normal(size=(n, dim)))
                    h.create_dataset("coords", data=coords)

    config = root / "encoders.yaml"
    config.write_text(
        f"feature_root: {root}\n"
        "encoders:\n"
        "  uni_v2:\n    display_name: UNI2-h\n    dim: 1536\n    family: vision_ssl\n"
        "  conch_v1:\n    display_name: CONCH\n    dim: 512\n    family: vision_language\n"
        "  resnet50:\n    display_name: ResNet50\n    dim: 1024\n    family: supervised\n"
        "  virchow:\n    display_name: Virchow\n    dim: 2560\n    family: vision_ssl\n"
        "  gpfm:\n    display_name: GPFM\n    dim: 1024\n    family: vision_ssl\n"
    )
    return FeatureStore(config=config)


def test_discovers_both_grids(fake_store):
    assert set(fake_store.groups) == {"cohortA/20x_256px", "cohortA/20x_224px"}


def test_group_records_grid_metadata(fake_store):
    g = fake_store.group("cohortA/20x_256px")
    assert g.cohort == "cohortA"
    assert g.magnification == 20.0
    assert g.patch_size == 256
    assert g.n_encoders == 3


def test_best_group_maximises_encoder_count(fake_store):
    assert fake_store.best_group().key == "cohortA/20x_256px"


def test_shared_slides_exclude_partial_coverage(fake_store):
    """gpfm only has s1, so the 224px group shares exactly one slide."""
    assert fake_store.group("cohortA/20x_256px").slides() == ["s1", "s2", "s3"]
    assert fake_store.group("cohortA/20x_224px").slides() == ["s1"]
    # Restricting to the fully-covered encoder recovers its own slides.
    assert fake_store.group("cohortA/20x_224px").slides(["virchow"]) == ["s1", "s2"]


def test_load_slide_is_row_paired(fake_store):
    g = fake_store.group("cohortA/20x_256px")
    out = g.load_slide("s1")
    assert {k: v.shape for k, v in out.items()} == {
        "uni_v2": (40, 1536),
        "conch_v1": (40, 512),
        "resnet50": (40, 1024),
    }


def test_load_slide_with_coords(fake_store):
    out = fake_store.group("cohortA/20x_256px").load_slide("s1", with_coords=True)
    assert out["coords"].shape == (40, 2)


def test_verify_alignment_passes_within_a_grid(fake_store):
    assert fake_store.group("cohortA/20x_256px").verify_alignment("s1")


def test_sample_patches_shapes_and_pairing(fake_store):
    g = fake_store.group("cohortA/20x_256px")
    views = g.sample_patches(n_patches=50, seed=0)
    assert set(views) == {"uni_v2", "conch_v1", "resnet50"}
    counts = {v.shape[0] for v in views.values()}
    assert len(counts) == 1, "views must have equal row counts to be paired"
    assert views["uni_v2"].shape[1] == 1536


def test_sample_patches_spans_multiple_slides(fake_store):
    """Sampling per slide, not from a global pool, avoids big-slide dominance."""
    g = fake_store.group("cohortA/20x_256px")
    views = g.sample_patches(n_patches=60, seed=0)
    # 60 patches over 3 slides means every slide had to contribute.
    assert views["uni_v2"].shape[0] == 60


def test_sample_patches_is_reproducible(fake_store):
    g = fake_store.group("cohortA/20x_256px")
    a = g.sample_patches(n_patches=40, seed=3)
    b = g.sample_patches(n_patches=40, seed=3)
    assert np.array_equal(a["uni_v2"], b["uni_v2"])


def test_sample_patches_rows_correspond(fake_store):
    """Row i must be the same patch in every view.

    Checked by loading one slide directly and confirming the sampled rows are a
    consistent subset across encoders — if the sampler drew independent indices
    per encoder, the paired coordinates would not line up.
    """
    g = fake_store.group("cohortA/20x_256px")
    views = g.sample_patches(n_patches=40, slides=["s1"], seed=1)
    direct = g.load_slide("s1")

    # Locate each sampled row of uni_v2 in the full slide, then confirm the
    # same positions reproduce the other encoders' sampled rows.
    pos = [
        int(np.argmin(np.abs(direct["uni_v2"] - row).sum(axis=1)))
        for row in views["uni_v2"]
    ]
    assert np.allclose(direct["conch_v1"][pos], views["conch_v1"])
    assert np.allclose(direct["resnet50"][pos], views["resnet50"])


def test_unknown_encoder_rejected(fake_store):
    with pytest.raises(KeyError, match="not available"):
        fake_store.group("cohortA/20x_256px").load_slide("s1", encoders=["nope"])


def test_unknown_group_rejected(fake_store):
    with pytest.raises(KeyError, match="unknown group"):
        fake_store.group("cohortA/99x_1px")


def test_missing_root_raises(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"feature_root: {tmp_path / 'nope'}\nencoders: {{}}\n")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        FeatureStore(config=cfg)


def test_display_names_and_families(fake_store):
    g = fake_store.group("cohortA/20x_256px")
    encs = sorted(g.encoders)
    assert fake_store.display_names(encs) == ["CONCH", "ResNet50", "UNI2-h"]
    assert fake_store.families(encs)["resnet50"] == "supervised"


def test_inventory_table(fake_store):
    df = fake_store.inventory()
    assert set(df["group"]) == {"cohortA/20x_256px", "cohortA/20x_224px"}
    assert df.iloc[0]["n_encoders"] == 3  # sorted by encoder count descending


# --- real store ----------------------------------------------------------


def test_registry_config_parses():
    """The shipped config must load and declare plausible dimensions."""
    root, encoders = load_encoder_config(DEFAULT_CONFIG)
    assert len(encoders) >= 10
    assert encoders["uni_v2"].dim == 1536
    assert encoders["resnet50"].family == "supervised"
    assert all(e.dim > 0 for e in encoders.values())


@requires_real_store
def test_real_store_declared_dims_match_disk():
    """Config dimensions must agree with what is actually stored.

    A mismatch here means the registry is describing a different checkpoint
    than the one the features came from — silent and damaging if unnoticed.
    """
    store = FeatureStore()
    checked = 0
    for group in store.groups.values():
        for enc, path in group.encoders.items():
            f = next(path.glob("*.h5"), None)
            if f is None:
                continue
            with h5py.File(f, "r") as h:
                dim = h["features"].shape[1]
            assert dim == store.encoder_info[enc].dim, (
                f"{enc} in {group.key}: config says "
                f"{store.encoder_info[enc].dim}, disk says {dim}"
            )
            checked += 1
    assert checked > 0


@requires_real_store
def test_real_store_best_group_is_pairable():
    store = FeatureStore()
    g = store.best_group()
    assert g.n_encoders >= 2
    slides = g.slides()
    assert len(slides) > 0
    assert g.verify_alignment(slides[0])
