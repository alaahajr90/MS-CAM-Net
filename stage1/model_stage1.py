"""MS-CAM-Net Stage 1 encoder, projection head, and training objective."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """3D-CNN encoder used for Stage 1 spatiotemporal pre-training."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                16,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2)),
        )

        self.convblock1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2)),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(1, 2, 2)),
        )

        self.convblock3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    @staticmethod
    def _to_sequence(x: torch.Tensor) -> torch.Tensor:
        # [B, C, T, H, W] -> [B, T, C]
        x = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
        return x.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return the hierarchical P5, P6, P7, and Pout temporal features."""
        p5 = self.conv1(x)
        p6 = self.convblock1(p5)
        p7 = self.convblock2(p6)
        pout = self.convblock3(p7)
        return (
            self._to_sequence(p5),
            self._to_sequence(p6),
            self._to_sequence(p7),
            self._to_sequence(pout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p5 = self.conv1(x)
        p6 = self.convblock1(p5)
        p7 = self.convblock2(p6)
        pout = self.convblock3(p7)
        pooled = self.global_pool(pout)
        return pooled.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()


class ProjectionHead(nn.Module):
    """Two-layer temporal projection head used only during Stage 1."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = x.shape
        projected = self.mlp(x.reshape(batch_size * time_steps, channels))
        return projected.view(batch_size, time_steps, -1)


def temporal_consistency_loss(
    z: torch.Tensor,
    delta: int = 1,
) -> torch.Tensor:
    """Mean cosine distance between embeddings separated by ``delta`` steps."""
    if z.ndim != 3:
        raise ValueError(f"Expected [B, T, D] embeddings, got {tuple(z.shape)}")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if z.shape[1] <= delta:
        return z.new_zeros(())

    current = z[:, :-delta, :]
    future = z[:, delta:, :]
    similarity = F.cosine_similarity(current, future, dim=-1)
    return (1.0 - similarity).mean()


def contrastive_temporal_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
    margin: float = 0.3,
    temporal_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the margin-adjusted contrastive-temporal Stage 1 objective.

    The batch and temporal dimensions are flattened so a physical batch of B
    clips and T temporal embeddings produces N = B*T anchors per view. For each
    anchor from view 1, the denominator contains all representations from both
    views except the anchor itself and its corresponding positive pair. The
    negative logits are shifted by ``margin`` before exponentiation.
    """
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have identical shapes.")
    if z1.ndim != 3:
        raise ValueError(f"Expected [B, T, D] embeddings, got {tuple(z1.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    batch_size, time_steps, dim = z1.shape
    n_anchors = batch_size * time_steps

    view1 = F.normalize(z1.reshape(n_anchors, dim), dim=1)
    view2 = F.normalize(z2.reshape(n_anchors, dim), dim=1)
    all_embeddings = torch.cat((view1, view2), dim=0)

    # Anchors are the N vectors from view 1. Candidates include both views.
    logits = torch.matmul(view1, all_embeddings.T) / temperature

    candidate_mask = torch.ones(
        (n_anchors, 2 * n_anchors),
        dtype=torch.bool,
        device=logits.device,
    )

    anchor_indices = torch.arange(n_anchors, device=logits.device)
    positive_indices = anchor_indices + n_anchors
    candidate_mask[anchor_indices, anchor_indices] = False
    candidate_mask[anchor_indices, positive_indices] = False

    negative_logits = logits.masked_fill(~candidate_mask, float("-inf")) - margin
    log_negative_sum = torch.logsumexp(negative_logits, dim=1)

    positive_logits = torch.sum(view1 * view2, dim=1) / temperature
    contrastive = -(positive_logits - log_negative_sum).mean()

    temporal_view1 = temporal_consistency_loss(z1, delta=1)
    temporal_view2 = temporal_consistency_loss(z2, delta=1)
    temporal_unweighted = temporal_view1 + temporal_view2

    total = contrastive + temporal_weight * temporal_unweighted

    components = {
        "total": total.detach(),
        "contrastive": contrastive.detach(),
        "temporal_unweighted": temporal_unweighted.detach(),
        "temporal_view1": temporal_view1.detach(),
        "temporal_view2": temporal_view2.detach(),
    }
    return total, components


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
    margin: float = 0.3,
    temporal_weight: float = 0.25,
) -> torch.Tensor:
    """Compatibility wrapper returning only the total Stage 1 loss."""
    total, _ = contrastive_temporal_loss(
        z1,
        z2,
        temperature=temperature,
        margin=margin,
        temporal_weight=temporal_weight,
    )
    return total
