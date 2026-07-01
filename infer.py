"""
Inference + evaluation script for UncertaintyGuidedHSIDiffusion.

Picks a random subset of RGB images from `rgb_dir`, runs the
deterministic prior + uncertainty-guided diffusion sampler on each,
and compares the reconstructed hyperspectral cube against the matching
ground-truth HSI file in `hsi_dir` (matched by filename stem).

Reports MRAE, RMSE, and SAM per image plus averages, and writes
predictions / previews / a metrics CSV to `output_dir`.

All settings are plain variables in the CONFIG block below -- edit them
directly instead of passing command line arguments.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from model import ModelConfig, UncertaintyGuidedHSIDiffusion


# --------------------------------------------------------------------------- #
# Configuration -- edit these values directly, no CLI parsing.
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    # I/O -- RGB inputs and ground-truth HSI cubes live in separate folders,
    # paired up by matching filename stem (e.g. "0001.png" <-> "0001.npy").
    rgb_dir: str = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_RGB/Train_RGB"
    hsi_dir: str = "/kaggle/input/datasets/sriramhari14/ntire-2022/Train_spectral/Train_spectral"
    output_dir: str = "outputs"
    rgb_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    hsi_extension: str = ".npy"  # ".npy" (array shaped H,W,C) or ".mat" (MATLAB struct)
    hsi_mat_key: Optional[str] = "cube"  # key inside the .mat file, only used if hsi_extension == ".mat"

    # Random sampling
    num_random_samples: int = 5
    sample_seed: Optional[int] = 42

    checkpoint_path: Optional[str] = "checkpoints/ug_hsi_diffusion_best.pth"
    checkpoint_key: Optional[str] = None  # e.g. "state_dict" / "model" / "ema"; None = auto-detect
    use_checkpoint_model_config: bool = True  # if the checkpoint stores a "model_config" dict, use it

    # Model architecture (only used if use_checkpoint_model_config=False, or as a fallback
    # when the checkpoint doesn't contain a "model_config" entry). Must match the trained model.
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
    torch_seed: Optional[int] = 0
    clamp_output: bool = True

    # Runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # Preview rendering
    save_npy: bool = True
    save_previews: bool = True
    false_color_bands: Tuple[int, int, int] = (25, 15, 5)  # (R, G, B) band indices into the HSI cube

    # Metrics
    mrae_eps: float = 1e-3
    metrics_csv_name: str = "metrics.csv"


CONFIG = InferenceConfig()


# --------------------------------------------------------------------------- #
# Model / checkpoint loading
# --------------------------------------------------------------------------- #
def default_model_config(cfg: InferenceConfig) -> ModelConfig:
    """Build a ModelConfig purely from the hand-set InferenceConfig fields."""
    return ModelConfig(
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


def load_checkpoint_file(cfg: InferenceConfig) -> Optional[dict]:
    if cfg.checkpoint_path is None:
        return None
    if not os.path.isfile(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")
    return torch.load(cfg.checkpoint_path, map_location=cfg.device)


def resolve_model_config(checkpoint: Optional[dict], cfg: InferenceConfig) -> ModelConfig:
    if (
        checkpoint is not None
        and cfg.use_checkpoint_model_config
        and isinstance(checkpoint, dict)
        and "model_config" in checkpoint
    ):
        print(f"Using model_config stored in checkpoint: {checkpoint['model_config']}")
        return ModelConfig.from_dict(checkpoint["model_config"])
    return default_model_config(cfg)


def build_model(model_cfg: ModelConfig, cfg: InferenceConfig) -> UncertaintyGuidedHSIDiffusion:
    model = UncertaintyGuidedHSIDiffusion(config=model_cfg)
    model.to(device=cfg.device, dtype=cfg.dtype)
    model.eval()
    return model


def load_state_dict_into_model(
    model: UncertaintyGuidedHSIDiffusion, checkpoint: Optional[dict], cfg: InferenceConfig
) -> None:
    if checkpoint is None:
        print("No checkpoint_path set -- running with randomly initialized weights.")
        return

    if cfg.checkpoint_key is not None:
        state = checkpoint[cfg.checkpoint_key]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
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
    if not missing and not unexpected:
        print("Checkpoint weights loaded with an exact key match.")


# --------------------------------------------------------------------------- #
# Dataset pairing (RGB folder <-> HSI folder, matched by filename stem)
# --------------------------------------------------------------------------- #
def find_paired_samples(cfg: InferenceConfig) -> List[Tuple[str, str]]:
    """Return a list of (rgb_path, hsi_path) pairs matched by filename stem."""
    if not os.path.isdir(cfg.rgb_dir):
        raise FileNotFoundError(f"rgb_dir not found: {cfg.rgb_dir}")
    if not os.path.isdir(cfg.hsi_dir):
        raise FileNotFoundError(f"hsi_dir not found: {cfg.hsi_dir}")

    rgb_files = sorted(
        f for f in os.listdir(cfg.rgb_dir) if os.path.splitext(f)[1].lower() in cfg.rgb_extensions
    )

    pairs: List[Tuple[str, str]] = []
    skipped: List[str] = []
    for rgb_name in rgb_files:
        stem = os.path.splitext(rgb_name)[0]
        hsi_path = os.path.join(cfg.hsi_dir, stem + cfg.hsi_extension)
        if os.path.isfile(hsi_path):
            pairs.append((os.path.join(cfg.rgb_dir, rgb_name), hsi_path))
        else:
            skipped.append(rgb_name)

    if skipped:
        print(f"Skipped {len(skipped)} RGB image(s) with no matching HSI file, e.g.: {skipped[:5]}")
    if not pairs:
        raise RuntimeError("No matching RGB/HSI pairs found -- check rgb_dir, hsi_dir, and hsi_extension.")

    return pairs


def pick_random_samples(pairs: List[Tuple[str, str]], cfg: InferenceConfig) -> List[Tuple[str, str]]:
    rng = random.Random(cfg.sample_seed)
    if cfg.num_random_samples >= len(pairs):
        print(
            f"num_random_samples ({cfg.num_random_samples}) >= available pairs ({len(pairs)}); "
            f"using all of them."
        )
        chosen = list(pairs)
        rng.shuffle(chosen)
        return chosen
    return rng.sample(pairs, cfg.num_random_samples)


# --------------------------------------------------------------------------- #
# Image / cube I/O
# --------------------------------------------------------------------------- #
def load_rgb_image(path: str, device: str, dtype: torch.dtype) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load an RGB image and return a normalized [1, 3, H, W] tensor in [0, 1]."""
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0  # H, W, 3
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # 1, 3, H, W
    tensor = tensor.to(device=device, dtype=dtype)
    original_size = (array.shape[0], array.shape[1])  # (H, W)
    return tensor, original_size


def load_hsi_cube(path: str, cfg: InferenceConfig) -> np.ndarray:
    """Load a ground-truth HSI cube and return it as a [C, H, W] float32 array in [0, 1]."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        array = np.load(path)
    elif ext == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError(
                "Reading .mat ground-truth files requires scipy. Install it with "
                "`pip install scipy --break-system-packages`."
            ) from exc
        mat = loadmat(path)
        if cfg.hsi_mat_key is None or cfg.hsi_mat_key not in mat:
            candidates = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(
                f"hsi_mat_key='{cfg.hsi_mat_key}' not found in {path}. Available keys: {candidates}"
            )
        array = mat[cfg.hsi_mat_key]
    else:
        raise ValueError(f"Unsupported hsi_extension '{ext}'. Use '.npy' or '.mat'.")

    array = np.asarray(array).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D HSI cube in {path}, got shape {array.shape}")

    # Accept either (H, W, C) or (C, H, W); assume the smallest dim is the channel dim.
    if array.shape[-1] <= array.shape[0] and array.shape[-1] <= array.shape[1]:
        array = array.transpose(2, 0, 1)  # H,W,C -> C,H,W
    return array  # C, H, W


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
# Metrics
# --------------------------------------------------------------------------- #
def compute_mrae(pred: np.ndarray, gt: np.ndarray, eps: float) -> float:
    return float(np.mean(np.abs(pred - gt) / (gt + eps)))


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def compute_sam(pred: np.ndarray, gt: np.ndarray) -> float:
    """Spectral Angle Mapper, averaged over pixels, in degrees. pred/gt: [C, H, W]."""
    pred_flat = pred.reshape(pred.shape[0], -1)  # C, N
    gt_flat = gt.reshape(gt.shape[0], -1)  # C, N
    dot = np.sum(pred_flat * gt_flat, axis=0)
    pred_norm = np.linalg.norm(pred_flat, axis=0)
    gt_norm = np.linalg.norm(gt_flat, axis=0)
    denom = np.clip(pred_norm * gt_norm, 1e-8, None)
    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    angles = np.arccos(cos_angle)
    return float(np.degrees(np.mean(angles)))


def compute_metrics(pred: np.ndarray, gt: np.ndarray, cfg: InferenceConfig) -> Dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch between prediction {pred.shape} and ground truth {gt.shape}")
    return {
        "mrae": compute_mrae(pred, gt, cfg.mrae_eps),
        "rmse": compute_rmse(pred, gt),
        "sam_deg": compute_sam(pred, gt),
    }


# --------------------------------------------------------------------------- #
# Per-image inference
# --------------------------------------------------------------------------- #
def run_inference_on_image(
    model: UncertaintyGuidedHSIDiffusion, rgb_path: str, cfg: InferenceConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (pred, uncertainty), each a [C, H, W] tensor at the original image size."""
    rgb, original_size = load_rgb_image(rgb_path, cfg.device, cfg.dtype)
    rgb_padded, original_size = pad_to_multiple(rgb, multiple=4)

    with torch.no_grad():
        if cfg.use_ensemble:
            result = model.sample_ensemble(
                rgb_padded,
                num_samples=cfg.ensemble_samples,
                steps=cfg.sampling_steps,
                clamp=cfg.clamp_output,
            )
            pred = unpad(result["mean"][0], original_size)
            uncertainty = unpad(result["std"][0], original_size)
        else:
            result = model.sample(
                rgb_padded,
                steps=cfg.sampling_steps,
                clamp=cfg.clamp_output,
                return_all=True,
            )
            pred = unpad(result["pred"][0], original_size)
            uncertainty = unpad(result["uncertainty"][0], original_size)

    return pred, uncertainty


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(cfg: InferenceConfig = CONFIG) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.torch_seed is not None:
        torch.manual_seed(cfg.torch_seed)

    print(f"Loading checkpoint: {cfg.checkpoint_path}")
    checkpoint = load_checkpoint_file(cfg)
    model_cfg = resolve_model_config(checkpoint, cfg)

    print(f"Building model on device={cfg.device}, dtype={cfg.dtype} ...")
    model = build_model(model_cfg, cfg)
    load_state_dict_into_model(model, checkpoint, cfg)

    pairs = find_paired_samples(cfg)
    print(f"Found {len(pairs)} matched RGB/HSI pairs.")
    samples = pick_random_samples(pairs, cfg)
    print(f"Selected {len(samples)} sample(s) for inference (seed={cfg.sample_seed}).")

    all_metrics: List[Dict[str, float]] = []

    for rgb_path, hsi_path in samples:
        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        print(f"\n--- {base_name} ---")
        print(f"RGB: {rgb_path}")
        print(f"GT HSI: {hsi_path}")

        pred, uncertainty = run_inference_on_image(model, rgb_path, cfg)
        gt = load_hsi_cube(hsi_path, cfg)

        pred_np = pred.detach().cpu().numpy()
        metrics = compute_metrics(pred_np, gt, cfg)
        metrics["name"] = base_name
        all_metrics.append(metrics)
        print(f"MRAE={metrics['mrae']:.4f}  RMSE={metrics['rmse']:.4f}  SAM={metrics['sam_deg']:.2f} deg")

        if cfg.save_npy:
            npy_path = os.path.join(cfg.output_dir, f"{base_name}_hsi.npy")
            save_hsi_cube(pred, npy_path)

        if cfg.save_previews:
            save_false_color_preview(
                pred, cfg.false_color_bands, os.path.join(cfg.output_dir, f"{base_name}_falsecolor.png")
            )
            save_uncertainty_preview(
                uncertainty, os.path.join(cfg.output_dir, f"{base_name}_uncertainty.png")
            )

    # Aggregate + write CSV
    csv_path = os.path.join(cfg.output_dir, cfg.metrics_csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "mrae", "rmse", "sam_deg"])
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)

    avg_mrae = float(np.mean([m["mrae"] for m in all_metrics]))
    avg_rmse = float(np.mean([m["rmse"] for m in all_metrics]))
    avg_sam = float(np.mean([m["sam_deg"] for m in all_metrics]))

    print("\n=== Summary over sampled images ===")
    print(f"n={len(all_metrics)}  MRAE={avg_mrae:.4f}  RMSE={avg_rmse:.4f}  SAM={avg_sam:.2f} deg")
    print(f"Per-image metrics saved to: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
