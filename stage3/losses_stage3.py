"""Loss function for MS-CAM-Net Stage 3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage3Loss(nn.Module):
    """Weighted Stage 3 objective reported in the manuscript.

    L_S3 = 2.0 * L_state + 5.0 * L_level + 0.1 * L_rPPG
    where both classification terms use cross-entropy with label smoothing 0.1
    and the auxiliary rPPG term uses mean squared error.
    """

    def __init__(
        self,
        state_weight: float = 2.0,
        level_weight: float = 5.0,
        rppg_weight: float = 0.1,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_weight = state_weight
        self.level_weight = level_weight
        self.rppg_weight = rppg_weight
        self.state_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.level_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        state_target: torch.Tensor,
        level_target: torch.Tensor,
        rppg_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state_loss = self.state_ce(outputs["state"], state_target)
        level_loss = self.level_ce(outputs["level"], level_target)

        pred_rppg = outputs["rppg"]
        if pred_rppg.shape[-1] != rppg_target.shape[-1]:
            pred_rppg = F.interpolate(
                pred_rppg.unsqueeze(1),
                size=rppg_target.shape[-1],
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        rppg_loss = F.mse_loss(pred_rppg, rppg_target)

        total = (
            self.state_weight * state_loss
            + self.level_weight * level_loss
            + self.rppg_weight * rppg_loss
        )
        return total, {
            "state": state_loss.detach(),
            "level": level_loss.detach(),
            "rppg": rppg_loss.detach(),
        }
