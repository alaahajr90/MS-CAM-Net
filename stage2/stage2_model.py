"""MS-CAM-Net Stage 2 supervised physiological refinement model."""

from __future__ import annotations

import torch
import torch.nn as nn

from model_parts import (
    MemoryAttentionLayer,
    RPPGReconstructionHead,
    TemporalProjector,
    VitalSignRegressor,
)


class Stage2PhysiologicalModel(nn.Module):
    """Refine Stage 1 hierarchical features and reconstruct physiological signals.

    The transferred Stage 1 projection head is not used. P5, P6, P7, and Pout
    are refined independently by Memory-Attention stacks, projected into a
    common 64-D space, integrated, and passed through a six-block long-range
    refinement module. One primary rPPG head produces the Stage 2 waveform.
    Auxiliary reconstruction heads are used only for the deep-supervision term
    L_aux described by the ablation analysis.
    """

    def __init__(self, encoder: nn.Module, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = encoder
        source_dims = (16, 32, 64, 64)

        self.attention_stacks = nn.ModuleList(
            [
                MemoryAttentionLayer(dim, num_blocks=4, num_heads=4, dropout=dropout)
                for dim in source_dims
            ]
        )
        self.projectors = nn.ModuleList([TemporalProjector(dim, 64) for dim in source_dims])

        self.feature_fusion = nn.Sequential(
            nn.Linear(64 * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        self.long_range_refinement = MemoryAttentionLayer(
            d_model=64,
            num_blocks=6,
            num_heads=8,
            dropout=dropout,
        )

        self.rppg_head = RPPGReconstructionHead(64, dropout=dropout)
        self.aux_rppg_heads = nn.ModuleList(
            [RPPGReconstructionHead(64, dropout=dropout) for _ in range(4)]
        )
        self.vitals_head = VitalSignRegressor(64, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        hierarchical = self.encoder.extract_features(x)
        if len(hierarchical) != 4:
            raise RuntimeError("Stage 1 encoder must return P5, P6, P7, and Pout.")

        refined_streams = []
        auxiliary_waveforms = []
        for feature, attention, projector, aux_head in zip(
            hierarchical,
            self.attention_stacks,
            self.projectors,
            self.aux_rppg_heads,
        ):
            refined = projector(attention(feature))
            refined_streams.append(refined)
            auxiliary_waveforms.append(aux_head(refined))

        fused = self.feature_fusion(torch.cat(refined_streams, dim=-1))
        refined = fused + self.long_range_refinement(fused)

        rppg = self.rppg_head(refined)
        hr, hrv = self.vitals_head(refined)

        return {
            "features": refined,
            "rppg": rppg,
            "rppg_aux": auxiliary_waveforms,
            "hr": hr,
            "hrv": hrv,
        }
