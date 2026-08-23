"""Layer-wise comparison of foundation models across depth (Phase IV).

Re-crops patches from downloaded slides at the coordinates trident recorded,
runs each encoder with forward hooks on every transformer block, and compares
every block of every model against every block of every other.

Unlike the other stages this needs slide images and model weights, not the
feature store: intermediate activations were never saved. Run
``scripts/download_slides.py`` first.

Examples
--------
    python scripts/layerwise_alignment.py --encoders uni_v2 conch_v1 --out results/layerwise
    python scripts/layerwise_alignment.py --encoders uni_v2 conch_v1 resnet50 \\
        --n-patches 1000 --pool mean --out results/layerwise_mean
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.features import FeatureStore  # noqa: E402
from utils.layers import (  # noqa: E402
    alignment_trajectory,
    divergence_point,
    extract_layer_activations,
    layer_depth_profile,
    layerwise_similarity,
    plot_alignment_trajectories,
    plot_layer_matrix,
)
from utils.visualization import save_figure  # noqa: E402

DEFAULT_SLIDES = Path("/path/to/Datasets/tcga_slides_layerwise")


def load_patches(
    slide_dir: Path,
    group,
    n_patches: int,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Crop patches from local slides at the feature store's own coordinates.

    Using the recorded coordinates keeps this analysis on exactly the tissue the
    rest of the study uses, and lets the extracted final layer be checked
    against the stored embeddings.

    Parameters
    ----------
    slide_dir : pathlib.Path
        Directory of ``.svs`` files.
    group : FeatureGroup
        Supplies the coordinate grid and patch size.
    n_patches : int
        Total patches to crop.
    seed : int, default 0
        Sampling seed.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    tuple
        ``(images, slide_ids)`` where images is ``(n, H, W, 3)`` uint8 RGB.
    """
    import h5py
    import openslide

    slides = sorted(slide_dir.glob("*.svs"))
    if not slides:
        raise SystemExit(
            f"no .svs files in {slide_dir}. Run: python scripts/download_slides.py"
        )

    rng = np.random.default_rng(seed)
    per_slide = max(1, n_patches // len(slides))
    images, used = [], []

    for path in slides:
        slide_id = path.stem
        coord_file = group.path / "patches" / f"{slide_id}_patches.h5"
        if not coord_file.exists():
            if verbose:
                print(f"  no coordinates for {slide_id[:40]}, skipping")
            continue

        with h5py.File(coord_file, "r") as h:
            coords = h["coords"][:]
            attrs = dict(h["coords"].attrs)

        patch_px = int(attrs.get("patch_size", group.patch_size))
        level0_px = int(attrs.get("patch_size_level0", patch_px))

        wsi = openslide.OpenSlide(str(path))
        take = min(per_slide, len(coords))
        sel = rng.choice(len(coords), size=take, replace=False)

        for i in sel:
            x, y = int(coords[i][0]), int(coords[i][1])
            region = wsi.read_region((x, y), 0, (level0_px, level0_px)).convert("RGB")
            if level0_px != patch_px:
                region = region.resize((patch_px, patch_px))
            images.append(np.asarray(region, dtype=np.uint8))
            used.append(slide_id)
        wsi.close()
        if verbose:
            print(f"  {slide_id[:40]}: {take} patches at {patch_px}px")

    if not images:
        raise SystemExit("no patches could be cropped")
    return np.stack(images), used


def preprocess(images: np.ndarray, size: int, mean, std):
    """Resize and normalise cropped patches for a model.

    Parameters
    ----------
    images : numpy.ndarray
        ``(n, H, W, 3)`` uint8 RGB.
    size : int
        Target side length.
    mean, std : sequence of float
        Per-channel normalisation.

    Returns
    -------
    torch.Tensor
        ``(n, 3, size, size)`` float tensor.
    """
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


#: Geo dependencies that trident's IO module imports for slide-contour work.
#: They are irrelevant to loading model weights, and are not installed here, so
#: they are stubbed rather than pulled in as heavyweight dependencies.
_UNUSED_TRIDENT_DEPS = ("geopandas", "shapely", "shapely.geometry", "fiona", "pyproj")


def _import_trident_encoder_factory(trident_root: Path):
    """Import trident's ``encoder_factory`` without executing its package init.

    ``trident/__init__.py`` and ``trident/IO.py`` pull in geopandas/shapely for
    contour handling, which has nothing to do with the model zoo. Loading
    ``load.py`` directly under stubbed geo modules keeps the model-loading code
    (and its per-model timm kwargs, which are fiddly and worth reusing) without
    requiring the geo stack.

    Parameters
    ----------
    trident_root : pathlib.Path
        Path to the trident checkout, i.e. the directory containing ``trident/``.

    Returns
    -------
    callable
        ``encoder_factory``.
    """
    import importlib.util
    import sys
    import types

    for dep in _UNUSED_TRIDENT_DEPS:
        if dep not in sys.modules:
            stub = types.ModuleType(dep)
            stub.__path__ = []  # mark as a package so submodules resolve
            for attr in ("gpd", "GeoDataFrame", "Polygon", "Point", "box"):
                setattr(stub, attr, object)
            sys.modules[dep] = stub

    pkg = trident_root / "trident"
    if not (pkg / "patch_encoder_models" / "load.py").exists():
        raise SystemExit(
            f"trident model zoo not found under {trident_root}. "
            "Pass --trident-root."
        )

    for mod, path in [
        ("trident", pkg),
        ("trident.patch_encoder_models", pkg / "patch_encoder_models"),
    ]:
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            stub.__path__ = [str(path)]
            sys.modules[mod] = stub

    spec = importlib.util.spec_from_file_location(
        "trident.patch_encoder_models.load", pkg / "patch_encoder_models" / "load.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.encoder_factory


def load_encoder(name: str, trident_root: Path):
    """Load one encoder through trident's model zoo.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``'uni_v2'``.
    trident_root : pathlib.Path
        Path to the trident checkout.

    Returns
    -------
    tuple
        ``(model, image_size, mean, std)``.
    """
    encoder_factory = _import_trident_encoder_factory(trident_root)
    enc = encoder_factory(name)
    model = getattr(enc, "model", enc)

    size = 224
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = getattr(enc, "eval_transforms", None)
    if transform is not None:
        for t in getattr(transform, "transforms", []):
            if hasattr(t, "mean") and hasattr(t, "std"):
                mean, std = tuple(t.mean), tuple(t.std)
            crop = getattr(t, "size", None)
            if isinstance(crop, int):
                size = crop
            elif isinstance(crop, (tuple, list)) and crop:
                size = int(crop[0])
    return model, size, mean, std


def main() -> None:
    """Run the layer-wise analysis."""
    parser = argparse.ArgumentParser(
        description="Phase IV: layer-wise alignment across depth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--encoders", nargs="+", required=True, help="encoders to compare"
    )
    parser.add_argument("--group", default="master_benchmark/10x_256px",
                        help="feature group supplying patch coordinates")
    parser.add_argument("--slide-dir", type=Path, default=DEFAULT_SLIDES,
                        help="directory of downloaded slides")
    parser.add_argument("--n-patches", type=int, default=512, help="patches to crop")
    parser.add_argument(
        "--pool", default="cls", choices=["cls", "mean", "cls_mean"],
        help="how to reduce each block's tokens to one vector per patch",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="inference batch")
    parser.add_argument("--max-samples", type=int, default=4000,
                        help="patch subsample for the CKA computation")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="best-match CKA below which blocks count as diverged")
    parser.add_argument("--trident-root", type=Path,
                        default=Path("/path/to/trident"),
                        help="trident checkout supplying the model zoo")
    parser.add_argument("--device", default=None, help="torch device")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--format", default="png", help="figure format")
    args = parser.parse_args()

    if len(args.encoders) < 2:
        parser.error("need at least 2 encoders to compare")

    store = FeatureStore()
    group = store.group(args.group)
    print(f"{group}\n")

    print(f"Cropping {args.n_patches} patches from {args.slide_dir} ...")
    images, _ = load_patches(args.slide_dir, group, args.n_patches, seed=args.seed)
    print(f"  {images.shape[0]} patches of {images.shape[1]}px\n")

    activations = {}
    for name in args.encoders:
        print(f"Extracting {name} ...")
        model, size, mean, std = load_encoder(name, args.trident_root)
        batch = preprocess(images, size, mean, std)
        acts = extract_layer_activations(
            model, batch, pool=args.pool, batch_size=args.batch_size,
            device=args.device, verbose=True,
        )
        activations[name] = acts
        print(f"  {len(acts)} blocks, final width {list(acts.values())[-1].shape[1]}\n")
        del model

    out = args.out
    (out / "matrices").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    matrices, trajectories = {}, {}
    for a, b in itertools.combinations(args.encoders, 2):
        print(f"Comparing {a} vs {b} ...")
        matrix = layerwise_similarity(
            activations[a], activations[b],
            max_samples=args.max_samples, seed=args.seed,
        )
        matrices[(a, b)] = matrix
        matrix.to_csv(out / "matrices" / f"{a}__{b}.csv")

        traj = alignment_trajectory(matrix)
        traj.to_csv(out / "matrices" / f"trajectory_{a}__{b}.csv", index=False)
        trajectories[f"{a} -> {b}"] = traj

        fig, _ = plot_layer_matrix(matrix, title=f"{a} vs {b} ({args.pool} pooling)")
        save_figure(fig, out / "figures" / f"layers_{a}__{b}.{args.format}")

        stats = divergence_point(matrix, args.threshold)
        print(f"  divergence depth {stats['divergence_depth']:.2f}, "
              f"early {stats['early_similarity']:.3f} vs late "
              f"{stats['late_similarity']:.3f}")

    fig, _ = plot_alignment_trajectories(
        trajectories, title=f"Alignment across depth ({args.pool} pooling)"
    )
    save_figure(fig, out / "figures" / f"trajectories.{args.format}")

    profile = layer_depth_profile(matrices, args.threshold)
    profile.to_csv(out / "divergence_profile.csv", index=False)

    # Per-model self-similarity across depth, as the within-model reference.
    depth_rows = []
    for name, acts in activations.items():
        first = list(acts)[0]
        for i, layer in enumerate(acts):
            from utils.cka import linear_cka

            depth_rows.append(
                {
                    "encoder": name,
                    "layer": layer,
                    "depth": i / max(len(acts) - 1, 1),
                    "cka_to_first_block": linear_cka(acts[first], acts[layer]),
                    "width": acts[layer].shape[1],
                }
            )
    pd.DataFrame(depth_rows).to_csv(out / "within_model_depth.csv", index=False)

    print("\n=== Divergence profile ===")
    print(profile.round(4).to_string(index=False))
    print(f"\nWrote layer-wise results to {out}")


if __name__ == "__main__":
    main()
