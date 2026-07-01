"""
Inference script for UncertaintyGuidedHSIDiffusion.

Loads an RGB image, runs the deterministic prior + uncertainty-guided
diffusion sampler, and writes the reconstructed hyperspectral cube
(plus a couple of preview images) to disk.

All settings are plain variables in the CONFIG block below -- edit them
directly instead of passing command line arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from models import ModelConfig, UncertaintyGuidedHSIDiffusion


# --------------------------------------------------------------------------- #
# Configuration -- edit these values directly, no CLI parsing.
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    # I/O
    input_image_path: str = "/kaggle/input/datasets/sriramhari14/ntire-2022"
    output_dir: str = "output"
    checkpoint_path: Optional[str] = "ug_hsi_diffusion_best.pth"
    checkpoint_key: Optional[str] = None  # e.g. "state_dict" / "model" / "ema"; None = auto-detect

    # Model architecture (must match the checkpoint being loaded)
    rgb_channels: int = 3
    hsi_channels: int = 31
    uncertainty_channels: int = 1
    num_timesteps: int = 8
    base_channels: int = 96
    time_dim: int = 128
    kappa: float = 2.0
    min_noise: float = 0.20
    uncertainty_max: float = 0.10
    residual_scale: float = 1.0

    # Sampling
    sampling_steps: Optional[int] = None  # None -> use num_timesteps
    use_ensemble: bool = False
    ensemble_samples: int = 4
    seed: Optional[int] = 0
    clamp_output: bool = True

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Preview rendering
    save_npy: bool = True
    save_previews: bool = True
    false_color_bands: Tuple[int, int, int] = (25, 15, 5)  # (R, G, B) band indices into the HSI cube


CONFIG = InferenceConfig()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_model(cfg: InferenceConfig) -> UncertaintyGuidedHSIDiffusion:
    model_cfg = ModelConfig(
        rgb_channels=cfg.rgb_channels,
        hsi_channels=cfg.hsi_channels,
        uncertainty_channels=cfg.uncertainty_channels,
        num_timesteps=cfg.num_timesteps,
        base_channels=cfg.base_channels,
        time_dim=cfg.time_dim,
        kappa=cfg.kappa,
        min_noise=cfg.min_noise,
        uncertainty_max=cfg.uncertainty_max,
        residual_scale=cfg.residual_scale,
    )
    model = UncertaintyGuidedHSIDiffusion(config=model_cfg)
    model.to(device=cfg.device, dtype=cfg.dtype)
    model.eval()
    return model


def load_checkpoint(model: UncertaintyGuidedHSIDiffusion, cfg: InferenceConfig) -> None:
    if cfg.checkpoint_path is None:
        print("No checkpoint_path set -- running with randomly initialized weights.")
        return
    if not os.path.isfile(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)

    if cfg.checkpoint_key is not None:
        state = checkpoint[cfg.checkpoint_key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "ema" in checkpoint:
        state = checkpoint["ema"]
    else:
        state = checkpoint

    clean_state = {name.replace("module.", "", 1): value for name, value in state.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[load_checkpoint] missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] unexpected keys: {unexpected}")


def load_rgb_image(path: str, device: str, dtype: torch.dtype) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load an RGB image and return a normalized [1, 3, H, W] tensor in [0, 1]."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input image not found: {path}")

    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0  # H, W, 3
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # 1, 3, H, W
    tensor = tensor.to(device=device, dtype=dtype)
    original_size = (array.shape[0], array.shape[1])  # (H, W)
    return tensor, original_size


def pad_to_multiple(x: torch.Tensor, multiple: int = 4) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Reflect-pad the last two dims of x up to the next multiple of `multiple`."""
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (height, width)


def unpad(x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    height, width = original_size
    return x[..., :height, :width]


def save_hsi_cube(cube: torch.Tensor, path: str) -> None:
    """cube: [C, H, W] tensor -> saved as .npy with shape [H, W, C]."""
    array = cube.detach().cpu().numpy().transpose(1, 2, 0)
    np.save(path, array)


def save_false_color_preview(cube: torch.Tensor, bands: Tuple[int, int, int], path: str) -> None:
    num_channels = cube.shape[0]
    bands = tuple(min(b, num_channels - 1) for b in bands)
    rgb = cube[list(bands)].detach().cpu().numpy()  # 3, H, W
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def save_uncertainty_preview(uncertainty: torch.Tensor, path: str) -> None:
    """uncertainty: [C, H, W] tensor, averaged across channels and normalized to 0-255."""
    heat = uncertainty.detach().cpu().numpy().mean(axis=0)
    min_val, max_val = heat.min(), heat.max()
    if max_val - min_val > 1e-8:
        heat = (heat - min_val) / (max_val - min_val)
    else:
        heat = np.zeros_like(heat)
    heat = (heat * 255.0).astype(np.uint8)
    Image.fromarray(heat, mode="L").save(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(cfg: InferenceConfig = CONFIG) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    print(f"Building model on device={cfg.device}, dtype={cfg.dtype} ...")
    model = build_model(cfg)
    load_checkpoint(model, cfg)

    print(f"Loading input image: {cfg.input_image_path}")
    rgb, original_size = load_rgb_image(cfg.input_image_path, cfg.device, cfg.dtype)
    rgb_padded, original_size = pad_to_multiple(rgb, multiple=4)

    with torch.no_grad():
        if cfg.use_ensemble:
            print(f"Sampling ensemble of {cfg.ensemble_samples} draws ...")
            result = model.sample_ensemble(
                rgb_padded,
                num_samples=cfg.ensemble_samples,
                steps=cfg.sampling_steps,
                clamp=cfg.clamp_output,
            )
            pred = unpad(result["mean"][0], original_size)
            uncertainty = unpad(result["std"][0], original_size)
        else:
            print("Sampling single draw ...")
            result = model.sample(
                rgb_padded,
                steps=cfg.sampling_steps,
                clamp=cfg.clamp_output,
                return_all=True,
            )
            pred = unpad(result["pred"][0], original_size)
            uncertainty = unpad(result["uncertainty"][0], original_size)

    base_name = os.path.splitext(os.path.basename(cfg.input_image_path))[0]

    if cfg.save_npy:
        npy_path = os.path.join(cfg.output_dir, f"{base_name}_hsi.npy")
        save_hsi_cube(pred, npy_path)
        print(f"Saved HSI cube: {npy_path}  shape={tuple(pred.shape)} (C, H, W)")

        unc_npy_path = os.path.join(cfg.output_dir, f"{base_name}_uncertainty.npy")
        save_hsi_cube(uncertainty, unc_npy_path)
        print(f"Saved uncertainty map: {unc_npy_path}")

    if cfg.save_previews:
        preview_path = os.path.join(cfg.output_dir, f"{base_name}_falsecolor.png")
        save_false_color_preview(pred, cfg.false_color_bands, preview_path)
        print(f"Saved false-color preview: {preview_path}")

        unc_preview_path = os.path.join(cfg.output_dir, f"{base_name}_uncertainty.png")
        save_uncertainty_preview(uncertainty, unc_preview_path)
        print(f"Saved uncertainty preview: {unc_preview_path}")

    print("Done.")


if __name__ == "__main__":
    main()
