"""Shared latent autoencoder: the only nonlinear aligner in the suite.

Every other Phase V method is restricted to linear maps. If models encode the
same morphology but under different *nonlinear* warpings of the manifold, no
linear aligner can recover the correspondence, and the linear methods would
report a low ceiling that is an artefact of their own restriction rather than a
fact about the models. This aligner exists to separate those two explanations:
if the autoencoder substantially beats GCCA, the shared structure is real but
nonlinear.

Architecture
------------
One MLP encoder and one MLP decoder per model, all encoders mapping into a
single shared latent space::

    X_m --enc_m--> z_m ---> (shared space) ---> dec_m --> X_m_hat

Objective
---------
.. math:: \\mathcal{L} = \\underbrace{\\sum_m \\|X_m - \\hat{X}_m\\|^2}
                          _{\\text{reconstruction}}
          + \\lambda \\underbrace{\\sum_m \\|z_m - \\bar{z}\\|^2}
                          _{\\text{alignment}}

The alignment term pulls every model's encoding of *the same patch* toward
their consensus. Without it each encoder is free to use its own private corner
of the latent space and nothing is shared; with it too strong, all encoders
collapse to a constant that is trivially aligned. ``align_weight`` controls
that trade-off and is the single most important hyperparameter here —
``reconstruction_r2`` and ``alignment_error`` in the evaluation module are what
you tune it against.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .base import BaseAligner

__all__ = ["SharedAutoencoderAligner"]


def _build_mlp(dims: Sequence[int], activation: str, final_activation: bool):
    """Build an MLP over the given layer widths."""
    import torch.nn as nn

    acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "elu": nn.ELU}
    if activation not in acts:
        raise ValueError(f"unknown activation {activation!r}; expected {list(acts)}")
    act = acts[activation]

    layers: list = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers.append(act())
    return nn.Sequential(*layers)


class SharedAutoencoderAligner(BaseAligner):
    """Nonlinear shared latent space via per-model encoders and decoders.

    Parameters
    ----------
    latent_dim : int, default 64
        Shared latent dimensionality.
    hidden_dims : sequence of int, default (512, 256)
        Hidden widths of each encoder; decoders mirror this.
    align_weight : float, default 1.0
        Weight on the latent alignment term. Sweep this — it trades
        reconstruction fidelity against cross-model agreement, and the useful
        range depends on how similar the models turn out to be.
    activation : {'relu', 'gelu', 'tanh', 'elu'}, default 'gelu'
        Hidden activation.
    dropout : float, default 0.0
        Dropout probability in the hidden layers.
    epochs : int, default 100
        Maximum training epochs.
    batch_size : int, default 256
        Minibatch size.
    lr : float, default 1e-3
        AdamW learning rate.
    weight_decay : float, default 1e-5
        AdamW weight decay.
    val_fraction : float, default 0.1
        Fraction of training patches held out for early stopping.
    patience : int, default 15
        Epochs without validation improvement before stopping.
    normalize_latent : bool, default True
        Apply LayerNorm at the encoder output. Prevents the degenerate
        solution where the alignment term is minimised by shrinking the latent
        space toward zero.
    device : str or None, default None
        Torch device. ``None`` selects CUDA when available.
    pca_dim : int or None, default None
        Per-view PCA pre-reduction. Useful to cut input width before the MLPs.
    scaling : {'rms', 'std', 'none'}, default 'rms'
        Per-view scaling.
    verbose : bool, default False
        Print per-epoch losses.
    random_state : int, default 0
        Seed for torch and numpy.

    Attributes
    ----------
    history_ : dict of str to list of float
        Per-epoch ``train_loss``, ``val_loss``, ``recon``, ``align``.
    best_epoch_ : int
        Epoch whose weights were restored.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] = (512, 256),
        align_weight: float = 1.0,
        activation: str = "gelu",
        dropout: float = 0.0,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.1,
        patience: int = 15,
        normalize_latent: bool = True,
        device: str | None = None,
        pca_dim: int | None = None,
        scaling: str = "rms",
        decoder_reg: float = 1e-6,
        verbose: bool = False,
        random_state: int = 0,
    ):
        super().__init__(
            latent_dim=latent_dim,
            pca_dim=pca_dim,
            scaling=scaling,
            decoder_reg=decoder_reg,
            random_state=random_state,
        )
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.align_weight = float(align_weight)
        self.activation = activation
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_fraction = float(val_fraction)
        self.patience = int(patience)
        self.normalize_latent = bool(normalize_latent)
        self.device = device
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------

    def _resolve_device(self):
        import torch

        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_modules(self, dims: dict[str, int]) -> None:
        """Instantiate the per-model encoder and decoder MLPs."""
        import torch.nn as nn

        k = self.latent_dim
        encoders, decoders = {}, {}
        for name, d in dims.items():
            enc = _build_mlp([d, *self.hidden_dims, k], self.activation, False)
            if self.normalize_latent:
                enc = nn.Sequential(enc, nn.LayerNorm(k))
            if self.dropout > 0:
                enc = nn.Sequential(nn.Dropout(self.dropout), enc)
            encoders[name] = enc
            decoders[name] = _build_mlp(
                [k, *reversed(self.hidden_dims), d], self.activation, False
            )

        self.encoders_ = nn.ModuleDict(encoders)
        self.decoders_nn_ = nn.ModuleDict(decoders)

    def _fit(self, prepared: dict[str, np.ndarray]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        names = list(prepared)
        device = self._resolve_device()
        self._build_modules({n: prepared[n].shape[1] for n in names})
        self.encoders_.to(device)
        self.decoders_nn_.to(device)

        tensors = [
            torch.as_tensor(prepared[n], dtype=torch.float32) for n in names
        ]
        n_samples = tensors[0].shape[0]

        n_val = int(round(n_samples * self.val_fraction))
        if n_val < 1 or n_samples - n_val < self.batch_size:
            n_val = 0
            warnings.warn(
                "too few samples for a validation split; early stopping is "
                "disabled and the final epoch's weights are kept.",
                RuntimeWarning,
                stacklevel=3,
            )

        g = torch.Generator().manual_seed(self.random_state)
        perm = torch.randperm(n_samples, generator=g)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        train_ds = TensorDataset(*[t[train_idx] for t in tensors])
        loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, generator=g
        )
        val_batch = (
            [t[val_idx].to(device) for t in tensors] if n_val > 0 else None
        )

        params = list(self.encoders_.parameters()) + list(
            self.decoders_nn_.parameters()
        )
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(self.patience // 3, 2)
        )

        self.history_ = {"train_loss": [], "val_loss": [], "recon": [], "align": []}
        best_val, best_state, best_epoch, stale = np.inf, None, 0, 0

        for epoch in range(self.epochs):
            self.encoders_.train()
            self.decoders_nn_.train()
            totals = np.zeros(3)
            n_batches = 0

            for batch in loader:
                batch = [b.to(device, non_blocking=True) for b in batch]
                opt.zero_grad(set_to_none=True)
                loss, recon, align = self._losses(names, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                totals += [loss.item(), recon.item(), align.item()]
                n_batches += 1

            totals /= max(n_batches, 1)
            self.history_["train_loss"].append(totals[0])
            self.history_["recon"].append(totals[1])
            self.history_["align"].append(totals[2])

            if val_batch is not None:
                self.encoders_.eval()
                self.decoders_nn_.eval()
                with torch.no_grad():
                    val_loss = self._losses(names, val_batch)[0].item()
                self.history_["val_loss"].append(val_loss)
                sched.step(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val, best_epoch, stale = val_loss, epoch, 0
                    best_state = {
                        "enc": {
                            k: v.detach().cpu().clone()
                            for k, v in self.encoders_.state_dict().items()
                        },
                        "dec": {
                            k: v.detach().cpu().clone()
                            for k, v in self.decoders_nn_.state_dict().items()
                        },
                    }
                else:
                    stale += 1
                    if stale >= self.patience:
                        if self.verbose:
                            print(f"early stop at epoch {epoch} (best {best_epoch})")
                        break

            if self.verbose and epoch % 10 == 0:
                val_str = (
                    f" val={self.history_['val_loss'][-1]:.4f}"
                    if val_batch is not None
                    else ""
                )
                print(
                    f"epoch {epoch:4d} loss={totals[0]:.4f} "
                    f"recon={totals[1]:.4f} align={totals[2]:.4f}{val_str}"
                )

        if best_state is not None:
            self.encoders_.load_state_dict(best_state["enc"])
            self.decoders_nn_.load_state_dict(best_state["dec"])
        self.best_epoch_ = best_epoch

        self.encoders_.to("cpu").eval()
        self.decoders_nn_.to("cpu").eval()
        self._device_used_ = str(device)

    def _losses(self, names: list[str], batch: list):
        """Reconstruction + alignment loss for one batch."""
        import torch

        zs = [self.encoders_[n](x) for n, x in zip(names, batch)]
        recon = sum(
            torch.nn.functional.mse_loss(self.decoders_nn_[n](z), x)
            for n, z, x in zip(names, zs, batch)
        ) / len(names)

        z_stack = torch.stack(zs, dim=0)
        consensus = z_stack.mean(dim=0, keepdim=True)
        # Mean over latent dimensions, not sum: a sum would make this term
        # latent_dim times larger than the per-element reconstruction MSE, so
        # align_weight=1.0 would silently mean "weight 64" and collapse the
        # latent space.
        align = ((z_stack - consensus) ** 2).mean()

        return recon + self.align_weight * align, recon, align

    def _encode(self, name: str, Xp: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = torch.as_tensor(np.asarray(Xp), dtype=torch.float32)
            return self.encoders_[name](x).numpy().astype(np.float64)

    def _decode(self, name: str, Z: np.ndarray) -> np.ndarray:
        """Use the trained neural decoder rather than the fitted linear one."""
        import torch

        with torch.no_grad():
            z = torch.as_tensor(np.asarray(Z), dtype=torch.float32)
            return self.decoders_nn_[name](z).numpy().astype(np.float64)

    def _fit_decoders(self, prepared: dict[str, np.ndarray]) -> None:
        """No-op: the decoders are trained jointly with the encoders."""
        self.decoders_ = {}
