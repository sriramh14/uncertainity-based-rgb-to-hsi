from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    mrae: float = 1.0
    l1: float = 0.2
    mse: float = 0.0
    sam: float = 0.05
    uncertainty: float = 0.05
    rgb: float = 0.0
    spectral_smooth: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "LossWeights":
        valid = cls().__dict__.keys()
        return cls(**{k: values[k] for k in valid if k in values})


def mrae_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps))


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mrae",
    mrae_eps: float = 1e-6,
) -> torch.Tensor:
    loss_type = loss_type.lower()
    if loss_type == "mrae":
        return mrae_loss(pred, target, eps=mrae_eps)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown reconstruction loss: {loss_type}. Use 'mrae', 'l1', or 'mse'.")


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_f = pred.flatten(2)
    target_f = target.flatten(2)
    dot = torch.sum(pred_f * target_f, dim=1)
    pred_norm = torch.linalg.norm(pred_f, dim=1)
    target_norm = torch.linalg.norm(target_f, dim=1)
    cos = dot / (pred_norm * target_norm + eps)
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(torch.acos(cos))


def spectral_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    return torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))


def rgb_consistency_loss(
    pred_hsi: torch.Tensor,
    rgb: torch.Tensor,
    camera_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if camera_matrix is None:
        return pred_hsi.new_zeros(())
    matrix = camera_matrix.to(device=pred_hsi.device, dtype=pred_hsi.dtype)
    if matrix.ndim != 2 or matrix.shape[0] != 3 or matrix.shape[1] != pred_hsi.shape[1]:
        raise ValueError(
            "camera_matrix must have shape [3, num_bands]. "
            f"Got {tuple(matrix.shape)} for pred_hsi with {pred_hsi.shape[1]} bands."
        )
    rgb_hat = torch.einsum("cl,blhw->bchw", matrix, pred_hsi)
    return F.l1_loss(rgb_hat, rgb)


def uncertainty_target(
    x_det: torch.Tensor,
    target_hsi: torch.Tensor,
    uncertainty_channels: int = 1,
) -> torch.Tensor:
    err = torch.abs(x_det.detach() - target_hsi)
    if uncertainty_channels == 1:
        return err.mean(dim=1, keepdim=True)
    if uncertainty_channels == target_hsi.shape[1]:
        return err
    raise ValueError(
        "uncertainty_channels must be 1 or equal to the number of HSI bands. "
        f"Got {uncertainty_channels}, target has {target_hsi.shape[1]} bands."
    )


def uncertainty_l1_loss(
    pred_uncertainty: torch.Tensor,
    x_det: torch.Tensor,
    target_hsi: torch.Tensor,
) -> torch.Tensor:
    target_uncertainty = uncertainty_target(
        x_det=x_det,
        target_hsi=target_hsi,
        uncertainty_channels=pred_uncertainty.shape[1],
    )
    return F.l1_loss(pred_uncertainty, target_uncertainty)


class HSICompositeLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        mrae_eps: float = 1e-6,
        camera_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.weights = weights or LossWeights()
        self.mrae_eps = mrae_eps
        if camera_matrix is not None:
            self.register_buffer("camera_matrix", camera_matrix.float(), persistent=True)
        else:
            self.camera_matrix = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        x_det: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        weights = self.weights

        loss_mrae = mrae_loss(pred, target, eps=self.mrae_eps)
        loss_l1 = F.l1_loss(pred, target)
        loss_mse = F.mse_loss(pred, target)
        loss_sam = sam_loss(pred, target)
        loss_smooth = spectral_smoothness_loss(pred)

        if x_det is not None and uncertainty is not None:
            loss_uncertainty = uncertainty_l1_loss(uncertainty, x_det, target)
        else:
            loss_uncertainty = pred.new_zeros(())

        if rgb is not None and self.camera_matrix is not None:
            loss_rgb = rgb_consistency_loss(pred, rgb, self.camera_matrix)
        else:
            loss_rgb = pred.new_zeros(())

        total = (
            weights.mrae * loss_mrae
            + weights.l1 * loss_l1
            + weights.mse * loss_mse
            + weights.sam * loss_sam
            + weights.uncertainty * loss_uncertainty
            + weights.rgb * loss_rgb
            + weights.spectral_smooth * loss_smooth
        )

        return {
            "loss": total,
            "mrae": loss_mrae.detach(),
            "l1": loss_l1.detach(),
            "mse": loss_mse.detach(),
            "sam": loss_sam.detach(),
            "uncertainty": loss_uncertainty.detach(),
            "rgb": loss_rgb.detach(),
            "spectral_smooth": loss_smooth.detach(),
        }
