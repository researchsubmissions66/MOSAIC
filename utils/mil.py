"""Phase VIII: slide-level downstream learning with multiple-instance learning.

A whole-slide image is a bag of thousands of patch embeddings with a single
slide-level label, which is the classic MIL setting. The head is kept
deliberately simple — the question is whether the *representation* helps, so
anything elaborate in the classifier would confound the comparison.

Three input conditions, which are the actual experiment:

1. **single** — one encoder's patches, the per-encoder baseline.
2. **concat** — every encoder's patches concatenated per patch. The strong
   baseline: it has strictly more information than any single encoder and than
   the shared space, so beating it is the real bar for Phase V.
3. **shared** — patches mapped through a fitted aligner into the shared latent
   space, optionally as the cross-model consensus.

Models
------
The study evaluates **two** heads and averages them per reported bar, so a
result is about the representation rather than one classifier:

``ABMIL`` (Ilse et al., ICML 2018) gated-attention pooling, and ``TransMIL``
-style self-attention pooling as the second baseline the plan asks for, to show
findings are not an ABMIL artefact.

``MeanMIL`` is implemented and tested but is **deliberately not part of the
evaluation**. It would be the no-attention control; it is kept available for
anyone who wants that comparison, and building it is a one-word change to
``--mil``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "ABMIL",
    "MeanMIL",
    "TransMIL",
    "build_mil_model",
    "MILConfig",
    "train_mil",
    "evaluate_mil",
]


# ---------------------------------------------------------------------
# models
# ---------------------------------------------------------------------


def _make_torch_modules():
    """Import torch lazily so the rest of the package works without it."""
    import torch
    import torch.nn as nn

    return torch, nn


class _MILBase:
    """Marker base so the builders can be documented without importing torch."""


def _abmil_cls():
    torch, nn = _make_torch_modules()

    class ABMIL(nn.Module, _MILBase):
        """Gated-attention MIL (Ilse et al., 2018).

        Each patch gets a scalar attention weight from a small gated network;
        the bag embedding is the attention-weighted mean of the patches. The
        attention weights are interpretable and are returned alongside the
        logits, which matters for the biological analysis in a later phase.

        Parameters
        ----------
        in_dim : int
            Patch embedding dimensionality.
        n_classes : int
            Number of output classes.
        hidden_dim : int, default 512
            Width of the instance encoder.
        attn_dim : int, default 256
            Width of the attention branches.
        dropout : float, default 0.25
            Dropout applied in the encoder and attention.
        """

        def __init__(
            self,
            in_dim: int,
            n_classes: int,
            hidden_dim: int = 512,
            attn_dim: int = 256,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.attn_v = nn.Sequential(
                nn.Linear(hidden_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout)
            )
            self.attn_u = nn.Sequential(
                nn.Linear(hidden_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout)
            )
            self.attn_w = nn.Linear(attn_dim, 1)
            self.head = nn.Linear(hidden_dim, n_classes)

        def forward(self, x, return_attention: bool = False):
            """Run one bag.

            Parameters
            ----------
            x : torch.Tensor
                Patch embeddings of shape ``(n_patches, in_dim)``.
            return_attention : bool, default False
                Also return the per-patch attention weights.

            Returns
            -------
            torch.Tensor or tuple
                Logits of shape ``(1, n_classes)``, optionally with attention
                of shape ``(n_patches,)``.
            """
            h = self.encoder(x)
            a = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
            a = torch.softmax(a, dim=0)
            bag = (a.unsqueeze(-1) * h).sum(dim=0, keepdim=True)
            logits = self.head(bag)
            return (logits, a) if return_attention else logits

    return ABMIL


def _meanmil_cls():
    torch, nn = _make_torch_modules()

    class MeanMIL(nn.Module, _MILBase):
        """Mean-pooling MIL — the control for what attention actually buys.

        Parameters
        ----------
        in_dim : int
            Patch embedding dimensionality.
        n_classes : int
            Number of output classes.
        hidden_dim : int, default 512
            Width of the instance encoder.
        dropout : float, default 0.25
            Dropout probability.
        """

        def __init__(
            self,
            in_dim: int,
            n_classes: int,
            hidden_dim: int = 512,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.head = nn.Linear(hidden_dim, n_classes)

        def forward(self, x, return_attention: bool = False):
            """Run one bag; see :meth:`ABMIL.forward`."""
            h = self.encoder(x).mean(dim=0, keepdim=True)
            logits = self.head(h)
            if return_attention:
                n = x.shape[0]
                return logits, torch.full((n,), 1.0 / n, device=x.device)
            return logits

    return MeanMIL


def _transmil_cls():
    torch, nn = _make_torch_modules()

    class TransMIL(nn.Module, _MILBase):
        """Self-attention MIL in the TransMIL spirit.

        A class token attends over the patch tokens through standard
        transformer layers. This is a simplified formulation — it omits the
        original's pyramid position encoding and Nystrom approximation, keeping
        full attention over a subsampled bag — but it serves the purpose the
        plan asks of it: an architecturally different second head, to show the
        findings are not an ABMIL artefact.

        Parameters
        ----------
        in_dim : int
            Patch embedding dimensionality.
        n_classes : int
            Number of output classes.
        hidden_dim : int, default 256
            Token width.
        n_heads : int, default 4
            Attention heads.
        n_layers : int, default 2
            Transformer layers.
        dropout : float, default 0.25
            Dropout probability.
        max_patches : int, default 4096
            Bags longer than this are randomly subsampled, since full attention
            is quadratic in bag size and slides can exceed 20k patches.
        """

        def __init__(
            self,
            in_dim: int,
            n_classes: int,
            hidden_dim: int = 256,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.25,
            max_patches: int = 4096,
        ):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, n_classes)
            self.max_patches = max_patches

        def forward(self, x, return_attention: bool = False):
            """Run one bag; see :meth:`ABMIL.forward`."""
            if x.shape[0] > self.max_patches:
                idx = torch.randperm(x.shape[0], device=x.device)[: self.max_patches]
                x = x[idx]

            h = self.proj(x).unsqueeze(0)
            h = torch.cat([self.cls_token.expand(1, -1, -1), h], dim=1)
            h = self.transformer(h)
            logits = self.head(self.norm(h[:, 0]))
            if return_attention:
                n = x.shape[0]
                return logits, torch.full((n,), 1.0 / n, device=x.device)
            return logits

    return TransMIL


def build_mil_model(name: str, in_dim: int, n_classes: int, **kwargs):
    """Instantiate a MIL head by name.

    Parameters
    ----------
    name : {'abmil', 'mean', 'transmil'}
        Model to build.
    in_dim : int
        Patch embedding dimensionality.
    n_classes : int
        Number of output classes.
    **kwargs
        Passed to the model constructor.

    Returns
    -------
    torch.nn.Module
        The MIL model.
    """
    builders = {"abmil": _abmil_cls, "mean": _meanmil_cls, "transmil": _transmil_cls}
    if name not in builders:
        raise KeyError(f"unknown MIL model {name!r}; available: {sorted(builders)}")
    return builders[name]()(in_dim=in_dim, n_classes=n_classes, **kwargs)


# Public aliases resolved lazily so importing this module never requires torch.
def ABMIL(*args, **kwargs):  # noqa: N802 - class-like factory
    """Construct a gated-attention MIL model. See :func:`build_mil_model`."""
    return _abmil_cls()(*args, **kwargs)


def MeanMIL(*args, **kwargs):  # noqa: N802
    """Construct a mean-pooling MIL model. See :func:`build_mil_model`."""
    return _meanmil_cls()(*args, **kwargs)


def TransMIL(*args, **kwargs):  # noqa: N802
    """Construct a transformer MIL model. See :func:`build_mil_model`."""
    return _transmil_cls()(*args, **kwargs)


# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------


@dataclass
class MILConfig:
    """Training configuration for a MIL head.

    Attributes
    ----------
    model : str
        ``'abmil'``, ``'mean'`` or ``'transmil'``.
    epochs : int
        Maximum epochs.
    lr : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    patience : int
        Epochs without validation improvement before stopping.
    val_fraction : float
        Fraction of *training patients* held out for early stopping.
    max_patches : int or None
        Random patch subsample per bag per epoch. Acts as augmentation and
        bounds memory; ``None`` uses the whole bag.
    class_weight : bool
        Weight the loss inversely to class frequency. On by default because
        every task here is imbalanced.
    grad_accum : int
        Bags per optimiser step. MIL bags vary in size so they are processed
        one at a time; this recovers a usable effective batch size.
    device : str or None
        Torch device; ``None`` selects CUDA when available.
    seed : int
        Seed.
    model_kwargs : dict
        Extra arguments for the model constructor.
    """

    model: str = "abmil"
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 10
    val_fraction: float = 0.15
    max_patches: int | None = 4096
    class_weight: bool = True
    grad_accum: int = 16
    device: str | None = None
    seed: int = 0
    model_kwargs: dict = field(default_factory=dict)


def _resolve_device(device: str | None):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mil(
    bags: Sequence[np.ndarray],
    labels: Sequence[int],
    config: MILConfig | None = None,
    groups: Sequence | None = None,
    verbose: bool = False,
):
    """Train a MIL head on slide bags.

    Parameters
    ----------
    bags : sequence of numpy.ndarray
        One ``(n_patches_i, dim)`` array per slide. Bag sizes may differ.
    labels : sequence of int
        Integer class label per slide.
    config : MILConfig, optional
        Training configuration. Defaults to :class:`MILConfig`.
    groups : sequence, optional
        Patient id per slide, used to keep the internal validation split
        patient-disjoint. Strongly recommended — without it the early-stopping
        signal is contaminated by the same leakage the outer split avoids.
    verbose : bool, default False
        Print per-epoch losses.

    Returns
    -------
    tuple
        ``(model, history)`` with the best weights restored.
    """
    import torch

    cfg = config or MILConfig()
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    y = np.asarray(labels, dtype=np.int64)
    n_classes = int(y.max()) + 1
    in_dim = bags[0].shape[1]

    # Patient-disjoint validation split for early stopping.
    idx = np.arange(len(bags))
    if cfg.val_fraction > 0:
        if groups is not None:
            g = np.asarray(groups)
            uniq = np.array(sorted(set(g.tolist())))
            rng.shuffle(uniq)
            val_groups = set(uniq[: max(1, int(round(len(uniq) * cfg.val_fraction)))])
            val_mask = np.array([gi in val_groups for gi in g])
        else:
            perm = rng.permutation(len(bags))
            n_val = max(1, int(round(len(bags) * cfg.val_fraction)))
            val_mask = np.zeros(len(bags), dtype=bool)
            val_mask[perm[:n_val]] = True
        train_idx, val_idx = idx[~val_mask], idx[val_mask]
    else:
        train_idx, val_idx = idx, np.array([], dtype=int)

    model = build_mil_model(cfg.model, in_dim, n_classes, **cfg.model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.class_weight:
        counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float64)
        w = len(train_idx) / (n_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        weight = None
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, stale = np.inf, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(train_idx)
        total, opt_steps = 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i, bi in enumerate(order):
            x = _bag_tensor(bags[bi], cfg.max_patches, rng, device)
            target = torch.tensor([y[bi]], device=device)
            loss = criterion(model(x), target) / cfg.grad_accum
            loss.backward()
            total += loss.item() * cfg.grad_accum

            if (i + 1) % cfg.grad_accum == 0 or i == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_steps += 1

        history["train_loss"].append(total / max(len(order), 1))

        if len(val_idx):
            model.eval()
            vtotal = 0.0
            with torch.no_grad():
                for bi in val_idx:
                    x = _bag_tensor(bags[bi], cfg.max_patches, rng, device)
                    target = torch.tensor([y[bi]], device=device)
                    vtotal += criterion(model(x), target).item()
            vloss = vtotal / len(val_idx)
            history["val_loss"].append(vloss)

            if vloss < best_val - 1e-5:
                best_val, stale = vloss, 0
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
            else:
                stale += 1
                if stale >= cfg.patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch}")
                    break

        if verbose:
            v = f" val={history['val_loss'][-1]:.4f}" if history["val_loss"] else ""
            print(f"  epoch {epoch:3d} train={history['train_loss'][-1]:.4f}{v}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _bag_tensor(bag: np.ndarray, max_patches: int | None, rng, device):
    """Convert one bag to a tensor, subsampling if it is too large."""
    import torch

    if max_patches is not None and bag.shape[0] > max_patches:
        sel = rng.choice(bag.shape[0], size=max_patches, replace=False)
        bag = bag[sel]
    return torch.as_tensor(np.ascontiguousarray(bag), dtype=torch.float32, device=device)


def evaluate_mil(
    model,
    bags: Sequence[np.ndarray],
    labels: Sequence[int],
    device: str | None = None,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Evaluate a trained MIL head.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    bags : sequence of numpy.ndarray
        Held-out slide bags.
    labels : sequence of int
        True labels.
    device : str, optional
        Torch device.
    class_names : sequence of str, optional
        Names for reporting.

    Returns
    -------
    dict
        ``auc`` (macro one-vs-rest, or binary), ``accuracy``, ``f1`` (macro),
        ``balanced_accuracy``, plus ``probs`` and ``preds``. Balanced accuracy
        and macro F1 are the ones to read — every task here is imbalanced, so
        plain accuracy flatters a majority-class predictor.
    """
    import torch
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    dev = _resolve_device(device)
    model = model.to(dev).eval()

    probs = []
    with torch.no_grad():
        for bag in bags:
            x = torch.as_tensor(
                np.ascontiguousarray(bag), dtype=torch.float32, device=dev
            )
            probs.append(torch.softmax(model(x), dim=-1).cpu().numpy()[0])
    probs = np.stack(probs)
    preds = probs.argmax(axis=1)
    y = np.asarray(labels, dtype=int)

    n_classes = probs.shape[1]
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y, probs[:, 1]))
        else:
            auc = float(
                roc_auc_score(y, probs, multi_class="ovr", average="macro")
            )
    except ValueError:
        auc = float("nan")  # a class missing from the test split

    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "n_test": int(len(y)),
        "probs": probs,
        "preds": preds,
        "class_names": list(class_names) if class_names else None,
    }
