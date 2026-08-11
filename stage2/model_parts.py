"""Reusable Stage 2 attention and physiological reconstruction modules."""

from __future__ import annotations

import torch
import torch.nn as nn


def apply_rotary_position_embedding(x: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a [B,T,C] temporal feature sequence."""
    if x.ndim != 3:
        raise ValueError(f"Expected [B,T,C], got {tuple(x.shape)}")

    channels = x.shape[-1]
    rotary_channels = channels - (channels % 2)
    if rotary_channels == 0:
        return x

    x_rotary = x[..., :rotary_channels]
    x_tail = x[..., rotary_channels:]

    half = rotary_channels // 2
    dtype = x.dtype
    device = x.device
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=dtype) / max(half, 1))
    )
    positions = torch.arange(x.shape[1], device=device, dtype=dtype)
    angles = torch.einsum("t,d->td", positions, inv_freq)
    cos = angles.cos().unsqueeze(0)
    sin = angles.sin().unsqueeze(0)

    first = x_rotary[..., :half]
    second = x_rotary[..., half:]
    rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    return torch.cat((rotated, x_tail), dim=-1)


class MemoryAttentionLayer(nn.Module):
    """RoPE-enhanced temporal Transformer stack used in Stage 2."""

    def __init__(
        self,
        d_model: int,
        num_blocks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = apply_rotary_position_embedding(x)
        for block in self.blocks:
            x = block(x)
        return x


class TemporalProjector(nn.Module):
    """Project a hierarchical temporal stream to the common 64-D Stage 2 space."""

    def __init__(self, in_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RPPGReconstructionHead(nn.Module):
    """Map a refined temporal representation to one reconstructed rPPG waveform."""

    def __init__(self, in_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(1)


class VitalSignRegressor(nn.Module):
    """Predict direct HR and HRV from the final refined Stage 2 representation."""

    def __init__(self, feature_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_feature = x.mean(dim=1)
        max_feature = x.amax(dim=1)
        values = self.net(torch.cat((mean_feature, max_feature), dim=-1))
        return values[:, 0], values[:, 1]
