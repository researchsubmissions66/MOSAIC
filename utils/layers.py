"""Phase IV: layer-wise comparison of foundation models.

Every other phase compares models by their **final** embedding. This one opens
them up and compares every transformer block, asking where in the network the
models are still doing the same thing and where they part ways.

Why this needs images
---------------------
The feature store holds only final pooled embeddings, so intermediate
activations cannot be recovered from it — the models have to be re-run on
actual patch images with forward hooks. ``scripts/download_slides.py`` fetches
a few whole-slide images for exactly this purpose, and patches are re-cropped
at the coordinates trident already recorded, which keeps the analysis on the
same tissue as every other phase.

Reading the output
------------------
The core object is an ``L_a x L_b`` matrix of CKA between every block of model A
and every block of model B.

* A bright **diagonal ridge** means the two networks progress through
  comparable representations at comparable depth.
* The ridge **falling away from the diagonal** in late layers means one model
  reaches a given representation earlier than the other.
* The ridge **breaking down entirely** past some depth is the divergence point:
  beyond it the models are doing different things, and only the shared early
  layers encode common morphology.

:func:`alignment_trajectory` reduces the matrix to one curve per model pair —
for each layer of A, which layer of B matches it best and how well — which is
the "alignment trajectory across depth" deliverable.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .cka import linear_cka

__all__ = [
    "vision_tower",
    "find_blocks",
    "extract_layer_activations",
    "layerwise_similarity",
    "alignment_trajectory",
    "divergence_point",
    "layer_depth_profile",
]


def vision_tower(model):
    """Return the image-encoding submodule of a possibly multimodal model.

    Vision-language encoders (CONCH is a CoCa, MUSK a BEiT-3) have a
    ``forward`` that demands text as well as images, and carry text blocks that
    would otherwise be hooked alongside the visual ones. This resolves the part
    that encodes images, so the same code path serves vision-only and
    vision-language models.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded encoder.

    Returns
    -------
    torch.nn.Module
        The image tower, or the model itself when it is already vision-only.
    """
    for attr in ("visual", "vision_model", "image_encoder", "vision_tower"):
        tower = getattr(model, attr, None)
        if tower is not None and hasattr(tower, "forward"):
            return tower
    return model


def find_blocks(model, pattern: str | None = None) -> list[tuple[str, object]]:
    """Locate the transformer blocks of a model.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded encoder.
    pattern : str, optional
        Substring that a module's qualified name must contain. Defaults to
        trying ``'blocks.'`` then ``'layer.'`` then ``'layers.'``, which covers
        timm ViTs, HF ViTs and Swin variants respectively.

    Returns
    -------
    list of tuple
        ``(name, module)`` for each block, in forward order.

    Raises
    ------
    ValueError
        If no blocks are found, which usually means the architecture needs an
        explicit ``pattern``.
    """
    import re

    named = list(model.named_modules())
    candidates = [pattern] if pattern else ["blocks.", "layer.", "layers."]

    for pat in candidates:
        hits = [
            (name, mod)
            for name, mod in named
            if pat in name and name.split(pat)[-1].isdigit()
        ]
        if hits:
            # Sort by the trailing block index so forward order is preserved.
            return sorted(hits, key=lambda kv: int(kv[0].split(pat)[-1]))

    if pattern is None:
        # Convolutional backbones (ResNet, ConvNeXt) have a handful of named
        # stages rather than uniform transformer blocks. Those stages are the
        # right unit of depth for them.
        stage_re = re.compile(r"^(layer|stage|stages\.|features\.)(\d+)$")
        stages = [(n, m) for n, m in named if stage_re.match(n)]
        if stages:
            return sorted(stages, key=lambda kv: int(stage_re.match(kv[0]).group(2)))

    raise ValueError(
        "no transformer blocks found; pass an explicit pattern. Available "
        f"module names include: {[n for n, _ in named[:20]]}"
    )


def _pool(tensor, mode: str):
    """Pool a block output to one vector per sample.

    Handles the ``(batch, tokens, dim)`` layout of ViTs and the
    ``(batch, dim)`` layout of already-pooled outputs.
    """
    import torch

    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"unexpected block output type {type(tensor)}")

    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 4:
        # Convolutional feature map (B, C, H, W): global average pool, which is
        # what these backbones do at their own output anyway. Flattening
        # instead would produce a spatially-ordered vector whose width depends
        # on input size and dwarfs the transformer widths.
        return tensor.mean(dim=(2, 3))
    if tensor.ndim != 3:
        return tensor.flatten(start_dim=1)

    if mode == "cls":
        return tensor[:, 0]
    if mode == "mean":
        return tensor.mean(dim=1)
    if mode == "cls_mean":
        # What Virchow does at its output: class token concatenated with the
        # mean of the patch tokens.
        return torch.cat([tensor[:, 0], tensor[:, 1:].mean(dim=1)], dim=-1)
    raise ValueError(f"unknown pooling {mode!r}")


def extract_layer_activations(
    model,
    images,
    pool: str = "cls",
    pattern: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    dtype=np.float32,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Run a model and capture the output of every transformer block.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded encoder, in eval mode.
    images : torch.Tensor or numpy.ndarray
        Preprocessed image batch of shape ``(n, 3, H, W)``, already normalised
        the way the model expects.
    pool : {'cls', 'mean', 'cls_mean'}, default 'cls'
        How to reduce each block's token sequence to one vector per patch. The
        choice matters: ``'cls'`` follows the classification pathway, ``'mean'``
        follows the spatial one, and models differ in which their final
        embedding uses. Sweep it rather than assuming.
    pattern : str, optional
        Block-name pattern, see :func:`find_blocks`.
    batch_size : int, default 32
        Inference batch size.
    device : str, optional
        Torch device; ``None`` selects CUDA when available.
    dtype : numpy dtype, default ``np.float32``
        Output dtype.
    verbose : bool, default False
        Print progress.

    Returns
    -------
    dict of str to numpy.ndarray
        ``{block_name: (n, dim)}`` in forward order. Widths may differ between
        blocks; the similarity metrics handle that.
    """
    import torch

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(dev).eval()

    # Hook and call the image tower, so vision-language models do not demand a
    # text input and do not contribute their text blocks to the comparison.
    tower = vision_tower(model)
    blocks = find_blocks(tower, pattern)
    if verbose:
        print(f"  hooking {len(blocks)} blocks")

    captured: dict[str, list] = {name: [] for name, _ in blocks}
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            captured[name].append(_pool(out, pool).detach().float().cpu())

        return hook

    for name, module in blocks:
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        images = torch.as_tensor(np.asarray(images)) if not torch.is_tensor(images) else images
        with torch.no_grad():
            for i in range(0, images.shape[0], batch_size):
                batch = images[i : i + batch_size].to(dev)
                tower(batch)
                if verbose and (i // batch_size) % 10 == 0:
                    print(f"    {min(i + batch_size, images.shape[0])}/{images.shape[0]}")
    finally:
        for h in handles:
            h.remove()

    return {
        name: torch.cat(parts, dim=0).numpy().astype(dtype)
        for name, parts in captured.items()
        if parts
    }


def layerwise_similarity(
    acts_a: Mapping[str, np.ndarray],
    acts_b: Mapping[str, np.ndarray],
    metric=linear_cka,
    max_samples: int | None = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare every block of one model against every block of another.

    Parameters
    ----------
    acts_a, acts_b : mapping of str to numpy.ndarray
        ``{block_name: (n, dim)}`` from :func:`extract_layer_activations`, over
        the *same* patches in the same order.
    metric : callable, default :func:`utils.cka.linear_cka`
        Similarity function taking two row-paired matrices.
    max_samples : int or None, default 4000
        Shared patch subsample, drawn once and reused for every layer pair so
        the cells are comparable.
    seed : int, default 0
        Subsampling seed.

    Returns
    -------
    pandas.DataFrame
        Rows are model A's blocks, columns model B's, in forward order.
    """
    names_a, names_b = list(acts_a), list(acts_b)
    n = acts_a[names_a[0]].shape[0]
    if acts_b[names_b[0]].shape[0] != n:
        raise ValueError("activations must cover the same patches in the same order")

    idx = None
    if max_samples is not None and n > max_samples:
        idx = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)

    def get(store, name):
        arr = store[name]
        return arr[idx] if idx is not None else arr

    out = np.empty((len(names_a), len(names_b)))
    for i, a in enumerate(names_a):
        Xa = get(acts_a, a)
        for j, b in enumerate(names_b):
            out[i, j] = metric(Xa, get(acts_b, b))

    return pd.DataFrame(out, index=names_a, columns=names_b)


def alignment_trajectory(matrix: pd.DataFrame) -> pd.DataFrame:
    """Reduce a layer-by-layer matrix to one best-match curve.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Output of :func:`layerwise_similarity`.

    Returns
    -------
    pandas.DataFrame
        One row per block of model A with:

        ``depth_a``, ``depth_b``
            Relative depth in ``[0, 1]`` of the A block and of its best match
            in B. Relative depth is used because models differ in block count,
            so absolute indices are not comparable.
        ``best_layer_b``
            Name of the best-matching B block.
        ``best_similarity``
            Its similarity.
        ``diagonal_similarity``
            Similarity at the same *relative* depth in B — what you would get
            assuming the two networks progress in lockstep.
        ``depth_offset``
            ``depth_b - depth_a``. Persistently negative means B reaches a
            comparable representation earlier than A.
    """
    n_a, n_b = matrix.shape
    values = matrix.values

    rows = []
    for i, name_a in enumerate(matrix.index):
        depth_a = i / max(n_a - 1, 1)
        j = int(np.argmax(values[i]))
        diag_j = int(round(depth_a * (n_b - 1)))
        rows.append(
            {
                "layer_a": name_a,
                "depth_a": depth_a,
                "best_layer_b": matrix.columns[j],
                "depth_b": j / max(n_b - 1, 1),
                "best_similarity": float(values[i, j]),
                "diagonal_similarity": float(values[i, diag_j]),
                "depth_offset": j / max(n_b - 1, 1) - depth_a,
            }
        )
    return pd.DataFrame(rows)


def divergence_point(
    matrix: pd.DataFrame, threshold: float = 0.5
) -> dict[str, float]:
    """Find the depth beyond which two models stop matching.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Output of :func:`layerwise_similarity`.
    threshold : float, default 0.5
        Best-match similarity below which the blocks are considered diverged.
        Report the value used — the depth this returns moves with it.

    Returns
    -------
    dict
        ``divergence_depth``
            Relative depth after which the best match stays below the
            threshold, or 1.0 if the models never sustainedly diverge. This is
            the *last sustained* crossing rather than the first dip: real
            trajectories are not monotone, and an early transient dip that
            recovers is not divergence.
        ``early_similarity`` / ``late_similarity``
            Mean best-match similarity over the first and last thirds of the
            network — the direct test of "do early layers encode universal
            morphology while late layers are model-specific".
        ``early_minus_late``
            Their difference. Positive supports the universal-early hypothesis;
            near zero or negative refutes it.
    """
    traj = alignment_trajectory(matrix)
    best = traj["best_similarity"].to_numpy()
    depth = traj["depth_a"].to_numpy()

    # The depth after which the models *stay* diverged, not merely the first
    # dip below the threshold: the trajectory is not monotone, and an early
    # transient dip that later recovers is not divergence.
    above = np.flatnonzero(best >= threshold)
    if above.size == 0:
        divergence = 0.0
    elif above[-1] == len(best) - 1:
        divergence = 1.0
    else:
        divergence = float(depth[above[-1] + 1])

    third = max(len(best) // 3, 1)
    early = float(best[:third].mean())
    late = float(best[-third:].mean())

    return {
        "divergence_depth": divergence,
        "early_similarity": early,
        "late_similarity": late,
        "early_minus_late": early - late,
        "threshold": threshold,
    }


def layer_depth_profile(
    matrices: Mapping[tuple[str, str], pd.DataFrame], threshold: float = 0.5
) -> pd.DataFrame:
    """Summarise divergence for every model pair.

    Parameters
    ----------
    matrices : mapping of tuple to pandas.DataFrame
        ``{(model_a, model_b): layerwise_matrix}``.
    threshold : float, default 0.5
        Divergence threshold.

    Returns
    -------
    pandas.DataFrame
        One row per pair with the divergence statistics, sorted by
        ``early_minus_late`` descending — the pairs that most strongly show
        "shared early, divergent late" first.
    """
    rows = []
    for (a, b), matrix in matrices.items():
        rows.append(
            {"model_a": a, "model_b": b, **divergence_point(matrix, threshold)}
        )
    return pd.DataFrame(rows).sort_values("early_minus_late", ascending=False)


def plot_layer_matrix(
    matrix: pd.DataFrame,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    ax=None,
    figsize: tuple[float, float] = (6.0, 5.0),
):
    """Draw a rectangular layer-by-layer similarity matrix.

    Separate from :func:`utils.visualization.plot_similarity_heatmap`, which
    assumes a square model-by-model matrix; here the two axes are different
    networks with different depths.

    Parameters
    ----------
    matrix : pandas.DataFrame
        Output of :func:`layerwise_similarity`.
    title : str, optional
        Axes title.
    cmap : str, default 'viridis'
        Colormap.
    vmin, vmax : float, optional
        Colour limits, shared across panels by default so depths compare.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    figsize : tuple of float, default (6.0, 5.0)
        Figure size when creating one.

    Returns
    -------
    tuple
        ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt

    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", aspect="auto")
    ax.set_xlabel("block index (model B)")
    ax.set_ylabel("block index (model A)")
    if title:
        ax.set_title(title)
    # The diagonal is the "lockstep" reference the ridge is read against.
    n_a, n_b = matrix.shape
    ax.plot([0, n_b - 1], [0, n_a - 1], color="white", lw=0.8, ls="--", alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("CKA")
    fig.tight_layout()
    return fig, ax


def plot_alignment_trajectories(
    trajectories: Mapping[str, pd.DataFrame],
    title: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (7.0, 5.0),
):
    """Plot best-match similarity against relative depth for several pairs.

    Parameters
    ----------
    trajectories : mapping of str to pandas.DataFrame
        ``{pair_label: alignment_trajectory(...)}``.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    figsize : tuple of float, default (7.0, 5.0)
        Figure size.

    Returns
    -------
    tuple
        ``(figure, axes)``.
    """
    import matplotlib.pyplot as plt

    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    for label, traj in trajectories.items():
        ax.plot(traj["depth_a"], traj["best_similarity"], marker="o",
                markersize=3, lw=1.6, label=label)
    ax.set_xlabel("relative depth in model A")
    ax.set_ylabel("best-match CKA in model B")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(title or "Alignment trajectory across depth")
    if len(trajectories) <= 14:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    return fig, ax


__all__ += ["plot_layer_matrix", "plot_alignment_trajectories"]
