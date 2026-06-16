from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    rgb_channels: int = 3
    hsi_channels: int = 31
    uncertainty_channels: int = 1
    num_timesteps: int = 8
    base_channels: int = 96
    time_dim: int = 128
    kappa: float = 2.0
    min_noise: float = 0.20
    uncertainty_max: float = 0.10
    freeze_prior: bool = False
    residual_scale: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict) -> "ModelConfig":
        valid_keys = cls().__dict__.keys()
        return cls(**{key: values[key] for key in valid_keys if key in values})


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.view(-1)
        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.out_ch = out_ch
        self.norm1 = RMSNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = RMSNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.GELU()
        self.time_to_scale_shift = nn.Linear(time_dim, out_ch * 2)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale_shift = self.time_to_scale_shift(t_emb).view(t_emb.shape[0], 2 * self.out_ch, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode="nearest")
        return self.conv(x)


class HSIPriorNet(nn.Module):
    def __init__(self, rgb_channels: int = 3, hsi_channels: int = 31, width: int = 64):
        super().__init__()
        self.stem = nn.Conv2d(rgb_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(width, hsi_channels, 3, padding=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        feat = self.stem(rgb)
        feat = feat + self.blocks(feat)
        return self.out(feat)


class SpectralUncertaintyHead(nn.Module):
    def __init__(
        self,
        rgb_channels: int = 3,
        hsi_channels: int = 31,
        output_channels: int = 1,
        width: int = 48,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(rgb_channels + hsi_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, output_channels, 1),
        )
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, rgb: torch.Tensor, x_det: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.net(torch.cat([rgb, x_det], dim=1)))


class SpectralSpatialDenoiser(nn.Module):
    def __init__(
        self,
        rgb_channels: int = 3,
        hsi_channels: int = 31,
        uncertainty_channels: int = 1,
        base_channels: int = 96,
        time_dim: int = 128,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.hsi_channels = hsi_channels
        self.uncertainty_channels = uncertainty_channels
        self.residual_scale = residual_scale

        in_ch = hsi_channels + hsi_channels + rgb_channels + uncertainty_channels
        self.time_mlp = TimeMLP(time_dim)
        self.in_conv = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        self.enc1 = ResBlock(base_channels, base_channels, time_dim)
        self.down1 = Downsample(base_channels)
        self.enc2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = Downsample(base_channels * 2)

        self.mid1 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up1 = Upsample(base_channels * 4, base_channels * 2)
        self.dec1 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up2 = Upsample(base_channels * 2, base_channels)
        self.dec2 = ResBlock(base_channels * 2, base_channels, time_dim)

        self.out_norm = RMSNorm2d(base_channels)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(base_channels, hsi_channels, 3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        rgb: torch.Tensor,
        x_det: torch.Tensor,
        uncertainty: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if uncertainty.shape[1] == 1 and self.uncertainty_channels != 1:
            uncertainty = uncertainty.repeat(1, self.uncertainty_channels, 1, 1)

        cond = torch.cat([x_t, x_det, rgb, uncertainty], dim=1)
        t_emb = self.time_mlp(t)

        h = self.in_conv(cond)
        s1 = self.enc1(h, t_emb)
        h = self.down1(s1)
        s2 = self.enc2(h, t_emb)
        h = self.down2(s2)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        h = self.up1(h, target_size=s2.shape[-2:])
        h = torch.cat([h, s2], dim=1)
        h = self.dec1(h, t_emb)

        h = self.up2(h, target_size=s1.shape[-2:])
        h = torch.cat([h, s1], dim=1)
        h = self.dec2(h, t_emb)

        residual = self.out(self.out_norm(h))
        return x_det + self.residual_scale * residual


class UncertaintyGuidedHSIDiffusion(nn.Module):
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        prior_net: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        if cfg.uncertainty_channels not in (1, cfg.hsi_channels):
            raise ValueError("uncertainty_channels must be 1 or equal to hsi_channels.")

        self.prior_net = prior_net if prior_net is not None else HSIPriorNet(
            rgb_channels=cfg.rgb_channels,
            hsi_channels=cfg.hsi_channels,
        )
        self.uncertainty_head = SpectralUncertaintyHead(
            rgb_channels=cfg.rgb_channels,
            hsi_channels=cfg.hsi_channels,
            output_channels=cfg.uncertainty_channels,
        )
        self.denoiser = SpectralSpatialDenoiser(
            rgb_channels=cfg.rgb_channels,
            hsi_channels=cfg.hsi_channels,
            uncertainty_channels=cfg.uncertainty_channels,
            base_channels=cfg.base_channels,
            time_dim=cfg.time_dim,
            residual_scale=cfg.residual_scale,
        )

        if cfg.freeze_prior:
            self.freeze_prior()

        eta = torch.linspace(0.0, 1.0, steps=cfg.num_timesteps + 1)
        self.register_buffer("eta", eta, persistent=False)

    def freeze_prior(self) -> None:
        self.prior_net.eval()
        for parameter in self.prior_net.parameters():
            parameter.requires_grad_(False)

    def unfreeze_prior(self) -> None:
        self.prior_net.train()
        for parameter in self.prior_net.parameters():
            parameter.requires_grad_(True)

    def deterministic_prior(self, rgb: torch.Tensor, detach: bool = True) -> torch.Tensor:
        if detach or self.config.freeze_prior:
            with torch.no_grad():
                return self.prior_net(rgb)
        return self.prior_net(rgb)

    def predict_uncertainty(self, rgb: torch.Tensor, x_det: torch.Tensor) -> torch.Tensor:
        return self.uncertainty_head(rgb, x_det)

    def uncertainty_to_weight(self, uncertainty: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        u = uncertainty.clamp(min=0.0, max=cfg.uncertainty_max) / max(cfg.uncertainty_max, 1e-8)
        return cfg.min_noise + (1.0 - cfg.min_noise) * u

    def q_sample(
        self,
        x0: torch.Tensor,
        x_det: torch.Tensor,
        noise_weight: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        eta_t = self.eta[t].view(-1, 1, 1, 1).to(device=x0.device, dtype=x0.dtype)
        x_t = x0 + eta_t * (x_det - x0) + self.config.kappa * noise_weight * torch.sqrt(eta_t.clamp_min(1e-8)) * noise
        return x_t, noise

    def p_sample_ddim(
        self,
        x_t: torch.Tensor,
        rgb: torch.Tensor,
        x_det: torch.Tensor,
        uncertainty: torch.Tensor,
        noise_weight: torch.Tensor,
        t_value: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_t.shape[0]
        t = torch.full((batch_size,), t_value, device=x_t.device, dtype=torch.long)
        x0_pred = self.denoiser(x_t, rgb, x_det, uncertainty, t)

        eta_t = self.eta[t_value].to(device=x_t.device, dtype=x_t.dtype)
        eta_prev = self.eta[t_value - 1].to(device=x_t.device, dtype=x_t.dtype)

        mean_t = (1.0 - eta_t) * x0_pred + eta_t * x_det
        eps_pred = (x_t - mean_t) / (
            self.config.kappa * noise_weight * torch.sqrt(eta_t.clamp_min(1e-8))
        )
        mean_prev = (1.0 - eta_prev) * x0_pred + eta_prev * x_det
        x_prev = mean_prev + self.config.kappa * noise_weight * torch.sqrt(eta_prev.clamp_min(0.0)) * eps_pred
        return x_prev, x0_pred

    def forward(
        self,
        rgb: torch.Tensor,
        hsi: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        detach_prior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if hsi is None:
            return {"pred": self.sample(rgb)}

        batch_size = rgb.shape[0]
        device = rgb.device
        x_det = self.deterministic_prior(rgb, detach=detach_prior)
        uncertainty = self.predict_uncertainty(rgb, x_det.detach() if detach_prior else x_det)
        noise_weight = self.uncertainty_to_weight(uncertainty)

        if t is None:
            t = torch.randint(
                1,
                self.config.num_timesteps + 1,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

        x_t, noise = self.q_sample(hsi, x_det, noise_weight, t, noise=noise)
        pred = self.denoiser(x_t, rgb, x_det, uncertainty, t)

        return {
            "pred": pred,
            "x_det": x_det,
            "uncertainty": uncertainty,
            "noise_weight": noise_weight,
            "x_t": x_t,
            "t": t,
            "noise": noise,
        }

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        clamp: bool = True,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        was_training = self.training
        self.eval()

        steps = self.config.num_timesteps if steps is None else steps
        steps = min(steps, self.config.num_timesteps)
        batch_size, _, height, width = rgb.shape

        x_det = self.deterministic_prior(rgb, detach=True)
        uncertainty = self.predict_uncertainty(rgb, x_det)
        noise_weight = self.uncertainty_to_weight(uncertainty)

        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.hsi_channels,
                height,
                width,
                device=rgb.device,
                dtype=rgb.dtype,
            )

        eta_t = self.eta[steps].to(device=rgb.device, dtype=rgb.dtype)
        x_t = x_det + self.config.kappa * noise_weight * torch.sqrt(eta_t.clamp_min(1e-8)) * noise
        last_pred = x_det

        for t_value in range(steps, 0, -1):
            x_t, last_pred = self.p_sample_ddim(
                x_t=x_t,
                rgb=rgb,
                x_det=x_det,
                uncertainty=uncertainty,
                noise_weight=noise_weight,
                t_value=t_value,
            )

        pred = last_pred if steps == 0 else x_t
        if clamp:
            pred = pred.clamp(0.0, 1.0)

        if was_training:
            self.train()

        if return_all:
            return {
                "pred": pred,
                "x_det": x_det,
                "uncertainty": uncertainty,
                "noise_weight": noise_weight,
            }
        return pred

    @torch.no_grad()
    def sample_ensemble(
        self,
        rgb: torch.Tensor,
        num_samples: int = 4,
        steps: Optional[int] = None,
        clamp: bool = True,
    ) -> Dict[str, torch.Tensor]:
        samples = [self.sample(rgb, steps=steps, clamp=clamp) for _ in range(num_samples)]
        stack = torch.stack(samples, dim=0)
        return {
            "mean": stack.mean(dim=0),
            "std": stack.std(dim=0),
            "samples": stack,
        }

    def load_prior_checkpoint(
        self,
        path: str,
        key: Optional[str] = None,
        strict: bool = False,
        map_location: Union[str, torch.device] = "cpu",
    ) -> None:
        checkpoint = torch.load(path, map_location=map_location)
        if key is not None:
            state = checkpoint[key]
        else:
            state = checkpoint
            for possible_key in ("params_ema", "params", "state_dict", "model"):
                if isinstance(checkpoint, dict) and possible_key in checkpoint:
                    state = checkpoint[possible_key]
                    break
        clean_state = {name.replace("module.", "", 1): value for name, value in state.items()}
        self.prior_net.load_state_dict(clean_state, strict=strict)
