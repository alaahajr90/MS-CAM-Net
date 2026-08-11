"""Stage 2 objective exactly following the manuscript formulation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2Loss(nn.Module):
    """Weighted Stage 2 physiological objective.

    L_S2 = alpha_e * (5 L_HR + 2 L_HRV + 2 L_sig + 0.1 L_aux)
    alpha_e = min(1, e / 10), with one-based epoch e.

    L_sig and L_HRV use MSE. L_HR combines Huber loss (delta=10) with
    the manuscript's error-dependent absolute-error penalty. L_aux is the mean
    MSE of the auxiliary deep-supervision reconstruction heads.
    """

    def __init__(
        self,
        w_hr: float = 5.0,
        w_hrv: float = 2.0,
        w_sig: float = 2.0,
        w_aux: float = 0.1,
        huber_delta: float = 10.0,
    ) -> None:
        super().__init__()
        self.w_hr = float(w_hr)
        self.w_hrv = float(w_hrv)
        self.w_sig = float(w_sig)
        self.w_aux = float(w_aux)
        self.huber_delta = float(huber_delta)

    def forward(
        self,
        outputs: dict,
        targets: dict,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if epoch < 1:
            raise ValueError("epoch must be one-based and >= 1.")

        pred_rppg = outputs["rppg"]
        true_rppg = targets["rppg"]
        pred_hr = outputs["hr"].reshape(-1)
        pred_hrv = outputs["hrv"].reshape(-1)
        true_hr = targets["hr"].reshape(-1)
        true_hrv = targets["hrv"].reshape(-1)

        l_sig = F.mse_loss(pred_rppg, true_rppg)
        l_hrv = F.mse_loss(pred_hrv, true_hrv)

        l_huber = F.huber_loss(pred_hr, true_hr, delta=self.huber_delta)
        hr_abs_error = torch.abs(pred_hr - true_hr)
        hr_penalty = torch.where(
            hr_abs_error > 20.0,
            0.5 * hr_abs_error,
            0.05 * hr_abs_error,
        ).mean()
        l_hr = l_huber + hr_penalty

        aux_outputs = outputs.get("rppg_aux", [])
        if aux_outputs:
            l_aux = torch.stack([F.mse_loss(aux, true_rppg) for aux in aux_outputs]).mean()
        else:
            l_aux = pred_rppg.new_zeros(())

        alpha_e = min(1.0, float(epoch) / 10.0)
        weighted = (
            self.w_hr * l_hr
            + self.w_hrv * l_hrv
            + self.w_sig * l_sig
            + self.w_aux * l_aux
        )
        total = alpha_e * weighted

        return total, {
            "total": float(total.detach()),
            "alpha": alpha_e,
            "hr": float(l_hr.detach()),
            "hrv": float(l_hrv.detach()),
            "signal": float(l_sig.detach()),
            "aux": float(l_aux.detach()),
        }
