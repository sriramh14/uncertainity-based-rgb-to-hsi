from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .losses import mrae_loss


@torch.no_grad()
def rmse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2).clamp_min(1e-12))


@torch.no_grad()
def psnr_metric(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    return 20.0 * torch.log10(pred.new_tensor(data_range)) - 10.0 * torch.log10(mse)


@torch.no_grad()
def sam_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_f = pred.flatten(2)
    target_f = target.flatten(2)
    dot = torch.sum(pred_f * target_f, dim=1)
    pred_norm = torch.linalg.norm(pred_f, dim=1)
    target_norm = torch.linalg.norm(target_f, dim=1)
    cos = dot / (pred_norm * target_norm + eps)
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.mean(torch.acos(cos)) * (180.0 / torch.pi)


@torch.no_grad()
def ssim_metric(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return torch.mean(numerator / denominator.clamp_min(1e-12))


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mrae_eps: float = 1e-6) -> Dict[str, float]:
    pred = pred.detach().float()
    target = target.detach().float()
    return {
        "mrae": float(mrae_loss(pred, target, eps=mrae_eps).item()),
        "rmse": float(rmse_metric(pred, target).item()),
        "psnr": float(psnr_metric(pred, target).item()),
        "sam": float(sam_metric(pred, target).item()),
        "ssim": float(ssim_metric(pred, target).item()),
    }
