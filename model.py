"""Factorized spatiotemporal convolutional autoencoder for masked LFP reconstruction.

Input is (batch, 2, time, channel):
    channel 0  masked LFP (hidden entries zeroed)
    channel 1  the binary mask itself

Channel 1 is not redundant. Real LFP is legitimately near zero much of the time, so
a network given only the zeroed signal would have to guess which zeros are data and
which are holes. Handing it the mask removes that ambiguity.

Two design constraints come straight from the data:

* **GroupNorm, never BatchNorm.** A batch mixes windows from different mice, probes
  and cortical states. BatchNorm would normalise each window by statistics pooled
  across unrelated recordings, and at eval time would apply running statistics that
  match no recording in particular. GroupNorm normalises within a sample.

* **Adaptive pooling.** Pretraining windows are 1250 samples; the downstream
  natural-scenes responses are 376. The pooled embedding must be the same size for
  both, so the encoder ends in AdaptiveAvgPool2d and the decoder interpolates back
  to whatever (T, C) came in. Verified at both lengths in synthetic_test.py.

Convolutions are factorized: (k_time, 1) then (1, k_channel), never a full 2-D
kernel. Time and depth are not interchangeable axes - one is 0.8 ms per step, the
other 40 um - and factorizing keeps the parameter count in budget while letting the
two receptive fields grow independently.

Sizes with the default config: 273,329 trainable parameters (191,344 encoder,
81,985 decoder), inside the 100k-500k budget from instructions.txt.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int, groups: int) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides `channels`."""
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class ResBlock(nn.Module):
    """Residual block: temporal (k_t, 1) conv then spatial (1, k_c) conv."""

    def __init__(self, channels: int, k_time: int, k_chan: int, groups: int) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            channels, channels, (k_time, 1), padding=(k_time // 2, 0)
        )
        self.norm1 = group_norm(channels, groups)
        self.spatial = nn.Conv2d(
            channels, channels, (1, k_chan), padding=(0, k_chan // 2)
        )
        self.norm2 = group_norm(channels, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, Ch) -> (B, C, T, Ch), shape preserved."""
        h = F.gelu(self.norm1(self.temporal(x)))
        h = self.norm2(self.spatial(h))
        return F.gelu(x + h)


class Pooling(nn.Module):
    """Latent (B, C, T', Ch') -> fixed-size embedding.

    The original `gap` mode is AdaptiveAvgPool2d(1): it averages the latent over the
    WHOLE window and the WHOLE shank, down to one number per feature channel. That
    measurably destroys signal - reading out from the un-pooled latent instead is worth
    +0.048 AUC on animacy - because the scene-evoked response is a transient (~40-160 ms
    post-onset) concentrated at particular depths, and averaging over 300 ms of mostly
    ongoing activity and 3.7 mm of shank dilutes it away.

    `grid` keeps a coarse spatiotemporal layout (an adaptive pool to a small T x Ch grid)
    instead of collapsing to a point. It still accepts any input length - which is what
    lets the same encoder take 1250-sample pretraining windows and 376-sample trials -
    but it no longer throws the transient away.

    `stat` keeps mean AND standard deviation per channel: cheap, and std carries the
    oscillatory power that a mean over a zero-centred signal cannot.
    """

    def __init__(self, n_features: int, cfg: dict) -> None:
        super().__init__()
        self.mode = str(cfg.get("pool_mode", "gap"))
        emb = int(cfg["embedding_dim"])

        if self.mode == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            pooled_dim = n_features
        elif self.mode == "grid":
            grid = tuple(int(g) for g in cfg.get("pool_grid", [4, 3]))
            self.pool = nn.AdaptiveAvgPool2d(grid)
            pooled_dim = n_features * grid[0] * grid[1]
        elif self.mode == "stat":
            self.pool = None
            pooled_dim = 2 * n_features
        else:
            raise ValueError(f"unknown pool_mode {self.mode!r}; use gap | grid | stat")

        self.project = nn.Linear(pooled_dim, emb)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self.mode == "stat":
            mean = latent.mean(dim=(2, 3))
            std = latent.std(dim=(2, 3))
            pooled = torch.cat([mean, std], dim=1)
        else:
            pooled = self.pool(latent).flatten(1)
        return self.project(pooled)


class Encoder(nn.Module):
    """Masked LFP -> (latent feature map, pooled embedding)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        c_in = int(cfg["in_channels"])
        c_stem = int(cfg["stem_channels"])
        c_mid = int(cfg["mid_channels"])
        c_out = int(cfg["out_channels"])
        emb = int(cfg["embedding_dim"])
        groups = int(cfg["groups"])
        kt = int(cfg["temporal_kernel"])
        ks = int(cfg["spatial_kernel"])
        rkt = int(cfg["res_temporal_kernel"])
        rks = int(cfg["res_spatial_kernel"])
        n_blocks = int(cfg["n_res_blocks"])

        # Temporal stem: stride 5 in time only. 1250 -> 250.
        self.stem = nn.Conv2d(c_in, c_stem, (kt, 1), stride=(5, 1), padding=(kt // 2, 0))
        self.stem_norm = group_norm(c_stem, groups)

        # Spatial compression: stride 2 in depth only. 93 -> 47.
        self.spatial = nn.Conv2d(
            c_stem, c_mid, (1, ks), stride=(1, 2), padding=(0, ks // 2)
        )
        self.spatial_norm = group_norm(c_mid, groups)

        self.blocks_mid = nn.Sequential(
            *[ResBlock(c_mid, rkt, rks, groups) for _ in range(n_blocks)]
        )

        # Final downsample in both axes. (250, 47) -> (125, 24).
        self.down = nn.Conv2d(
            c_mid, c_out, (rkt, rks), stride=(2, 2), padding=(rkt // 2, rks // 2)
        )
        self.down_norm = group_norm(c_out, groups)

        self.blocks_out = nn.Sequential(
            *[ResBlock(c_out, rkt, rks, groups) for _ in range(n_blocks)]
        )

        self.head = Pooling(c_out, cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 2, T, Ch) -> latent (B, 64, T/10, Ch/4), embedding (B, emb).

        The embedding is fixed-size for any T, which is what lets the same encoder
        take 1250-sample pretraining windows and 376-sample downstream responses.
        """
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, T, Ch), got {tuple(x.shape)}")

        latent = self.trunk(x)
        return latent, self.head(latent)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Everything up to the latent. These are the weights pretraining actually learns."""
        h = F.gelu(self.stem_norm(self.stem(x)))
        h = F.gelu(self.spatial_norm(self.spatial(h)))
        h = self.blocks_mid(h)
        h = F.gelu(self.down_norm(self.down(h)))
        return self.blocks_out(h)


class Decoder(nn.Module):
    """Latent -> reconstructed LFP, upsampled back to the exact input size.

    Interpolate-then-convolve rather than transposed convolution: transposed convs
    with stride 5 leave checkerboard artifacts, which here would be indistinguishable
    from oscillatory structure - the very thing we are trying to learn.

    Discarded after pretraining. It exists only to make the masked loss computable.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        c_mid = int(cfg["mid_channels"])
        c_out = int(cfg["out_channels"])
        c_stem = int(cfg["stem_channels"])
        groups = int(cfg["groups"])
        rkt = int(cfg["res_temporal_kernel"])
        rks = int(cfg["res_spatial_kernel"])

        self.conv1 = nn.Conv2d(c_out, c_mid, (rkt, rks), padding=(rkt // 2, rks // 2))
        self.norm1 = group_norm(c_mid, groups)
        self.block = ResBlock(c_mid, rkt, rks, groups)
        self.conv2 = nn.Conv2d(c_mid, c_stem, (rkt, rks), padding=(rkt // 2, rks // 2))
        self.norm2 = group_norm(c_stem, groups)
        self.out = nn.Conv2d(c_stem, 1, (5, 3), padding=(2, 1))

    def forward(self, latent: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """(B, 64, T', Ch') -> (B, 1, T, Ch) for the requested (T, Ch)."""
        n_time, n_chan = int(size[0]), int(size[1])

        # Two-stage upsample: latent -> half resolution -> exact input size.
        mid = (max(n_time // 5, 1), max(n_chan // 2, 1))
        h = F.interpolate(latent, size=mid, mode="bilinear", align_corners=False)
        h = self.block(F.gelu(self.norm1(self.conv1(h))))

        h = F.interpolate(h, size=(n_time, n_chan), mode="bilinear", align_corners=False)
        h = F.gelu(self.norm2(self.conv2(h)))
        return self.out(h)


class MaskedLFPAutoencoder(nn.Module):
    """Encoder + decoder. `forward` returns (reconstruction, latent, embedding)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_time, n_chan = int(x.shape[-2]), int(x.shape[-1])
        latent, embedding = self.encoder(x)
        recon = self.decoder(latent, (n_time, n_chan))
        return recon, latent, embedding


def build_input(window: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Assemble the 2-channel network input from a window and its mask.

    Args:
        window: (B, 1, T, Ch) unmasked LFP.
        mask:   (B, 1, T, Ch) True/1.0 where hidden.

    Returns:
        (B, 2, T, Ch): [masked LFP, mask].
    """
    visible = window * (1.0 - mask)
    return torch.cat([visible, mask], dim=1)


def masked_huber_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """L = sum(M * Huber(X, X_hat)) / sum(M), over masked entries only.

    Visible entries are excluded deliberately. If they contributed, the network
    could drive the loss down by learning to copy its input through the residual
    path - the reconstruction would look excellent and the encoder would have
    learned nothing. Only the masked entries carry signal about whether the model
    can actually infer LFP it was not shown.

    Huber rather than MSE because LFP carries sharp transients (sharp-wave ripples,
    movement artifacts) whose squared error would dominate the gradient.
    """
    per_element = F.huber_loss(recon, target, reduction="none", delta=delta)
    denom = mask.sum().clamp_min(1.0)
    return (per_element * mask).sum() / denom


def masked_mae(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over masked entries only (reported, not optimised)."""
    denom = mask.sum().clamp_min(1.0)
    return ((recon - target).abs() * mask).sum() / denom


def load_encoder_trunk(encoder: Encoder, state: dict, strict_head: bool = False) -> str:
    """Load a pretrained encoder, tolerating a different pooling head.

    Pretraining learns the trunk (stem, spatial compression, residual stages). The head
    is just a readout, and swapping it is the whole point of `pool_mode` - so a checkpoint
    trained with `gap` must still be usable with a `grid` head. Trunk weights are loaded;
    head weights are loaded only if they match, otherwise left at their init (they get
    trained downstream anyway).

    Returns a human-readable summary of what was loaded.
    """
    own = encoder.state_dict()
    trunk_keys = [k for k in own if not k.startswith("head.")]

    missing = [k for k in trunk_keys if k not in state]
    if missing:
        raise RuntimeError(
            f"checkpoint is missing trunk weights {missing[:4]}... - it does not match "
            f"this architecture"
        )

    to_load = {k: state[k] for k in trunk_keys}

    head_loaded = 0
    for k in own:
        if k.startswith("head.") and k in state and state[k].shape == own[k].shape:
            to_load[k] = state[k]
            head_loaded += 1

    encoder.load_state_dict(to_load, strict=False)
    head_note = "head reused" if head_loaded else "head re-initialised (new pool_mode)"
    return f"{len(trunk_keys)} trunk tensors loaded, {head_note}"


def count_parameters(module: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_model(cfg: dict) -> MaskedLFPAutoencoder:
    """Construct the autoencoder and check it against the size budget."""
    model = MaskedLFPAutoencoder(cfg)
    n = count_parameters(model)
    if not 100_000 <= n <= 500_000:
        raise ValueError(
            f"model has {n} trainable parameters, outside the 100k-500k budget "
            f"required by instructions.txt; adjust the `model` block in config.yaml"
        )
    return model


if __name__ == "__main__":
    import yaml

    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh)["model"]

    model = build_model(cfg)
    print(f"encoder   : {count_parameters(model.encoder):>8,d}")
    print(f"decoder   : {count_parameters(model.decoder):>8,d}")
    print(f"TOTAL     : {count_parameters(model):>8,d}")

    for n_time in (1250, 376):
        x = torch.randn(2, 2, n_time, 93)
        recon, latent, emb = model(x)
        print(
            f"T={n_time:>4d}: in {tuple(x.shape)} -> latent {tuple(latent.shape)} "
            f"embedding {tuple(emb.shape)} recon {tuple(recon.shape)}"
        )
