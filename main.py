#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset_loader import ARADDataset
from dataset.random_arad_loader import load_random_arad1k_samples
from loss import HSICompositeLoss, LossWeights, compute_metrics, reconstruction_loss
from models.ug_hsi_diffusion_model import ModelConfig, UncertaintyGuidedHSIDiffusion


# ==================================================
# CONFIG
# ==================================================

MODE = "train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_SEED = 1234

DATA_ROOT = "data"
HSI_KEY = "cube"
DOWNLOAD_DATA = True
TRAIN_IMAGES = 2
TOTAL_IMAGES = 4
EVAL_RANDOM_IMAGES = 50
EVAL_RANDOM_TOTAL_IMAGES = 1000

BATCH_SIZE = 4
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"

NUM_EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
USE_AMP = True

NUM_BANDS = 31
UNCERTAINTY_CHANNELS = 1
DIFFUSION_TIMESTEPS = 8
SAMPLE_STEPS = 8
BASE_CHANNELS = 96
TIME_DIM = 128
KAPPA = 2.0
MIN_NOISE = 0.20
UNCERTAINTY_MAX = 0.10
RESIDUAL_SCALE = 1.0

PRIOR_CHECKPOINT: Optional[Union[str, Path]] = None
PRIOR_CHECKPOINT_KEY: Optional[str] = None
FREEZE_PRIOR = False

RECONSTRUCTION_LOSS = "mrae"
MRAE_EPS = 1e-6
LAMBDA_MRAE = 1.0
LAMBDA_L1 = 0.2
LAMBDA_MSE = 0.0
LAMBDA_SAM = 0.05
LAMBDA_UNCERTAINTY = 0.05
LAMBDA_RGB = 0.0
LAMBDA_SPECTRAL_SMOOTH = 0.0
CAMERA_MATRIX_PATH: Optional[Union[str, Path]] = None

EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 4
LR_FACTOR = 0.5
MIN_LR = 1e-7

CHECKPOINT_DIR = Path("checkpoints")
BEST_PATH = CHECKPOINT_DIR / "ug_hsi_diffusion_best.pth"
BEST_LOSS_PATH = CHECKPOINT_DIR / "ug_hsi_diffusion_best_loss.pth"
LATEST_PATH = CHECKPOINT_DIR / "ug_hsi_diffusion_latest.pth"
RESUME_CHECKPOINT: Optional[Union[str, Path]] = None
EVAL_CHECKPOINT: Optional[Union[str, Path]] = None



def ensure_checkpoint_dir() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


ensure_checkpoint_dir()


# ==================================================
# COMMAND LINE OVERRIDES
# ==================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train uncertainty-guided RGB-to-HSI diffusion")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default=MODE)
    return parser.parse_args()


args = parse_args()
MODE = args.mode


# ==================================================
# REPRODUCIBILITY
# ==================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ==================================================
# SMALL UTILITIES
# ==================================================


def make_model_config() -> ModelConfig:
    return ModelConfig(
        rgb_channels=3,
        hsi_channels=NUM_BANDS,
        uncertainty_channels=UNCERTAINTY_CHANNELS,
        num_timesteps=DIFFUSION_TIMESTEPS,
        base_channels=BASE_CHANNELS,
        time_dim=TIME_DIM,
        kappa=KAPPA,
        min_noise=MIN_NOISE,
        uncertainty_max=UNCERTAINTY_MAX,
        freeze_prior=FREEZE_PRIOR,
        residual_scale=RESIDUAL_SCALE,
    )


def make_loss_weights() -> LossWeights:
    return LossWeights(
        mrae=LAMBDA_MRAE,
        l1=LAMBDA_L1,
        mse=LAMBDA_MSE,
        sam=LAMBDA_SAM,
        uncertainty=LAMBDA_UNCERTAINTY,
        rgb=LAMBDA_RGB,
        spectral_smooth=LAMBDA_SPECTRAL_SMOOTH,
    )


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def load_camera_matrix(path: Optional[Union[str, Path]]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera matrix not found: {path}")
    if path.suffix.lower() in {".pt", ".pth"}:
        matrix = torch.load(path, map_location="cpu")
    else:
        matrix = np.load(path)
        matrix = torch.from_numpy(matrix)
    matrix = matrix.float()
    if matrix.shape != (3, NUM_BANDS):
        raise ValueError(f"Expected camera matrix [3,{NUM_BANDS}], got {tuple(matrix.shape)}")
    return matrix


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Any, Optional[torch.Tensor]]:
    name = None
    orig_hw = None

    if isinstance(batch, dict):
        rgb = batch.get("rgb", batch.get("lq"))
        hsi = batch.get("hsi", batch.get("gt"))
        name = batch.get("name", batch.get("filename"))
        orig_hw = batch.get("orig_hw", batch.get("original_hw"))
        if rgb is None or hsi is None:
            raise KeyError(
                "Dictionary batch must contain ('rgb','hsi') or ('lq','gt'). "
                f"Available keys: {list(batch.keys())}"
            )
    elif isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("List/tuple batch must contain at least [rgb, hsi].")
        rgb = batch[0]
        hsi = batch[1]
        if len(batch) >= 3:
            name = batch[2]
        if len(batch) >= 4:
            orig_hw = batch[3]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch).__name__}")

    if not torch.is_tensor(rgb) or not torch.is_tensor(hsi):
        raise TypeError(f"RGB and HSI must be tensors, got rgb={type(rgb)}, hsi={type(hsi)}")
    if rgb.ndim != 4 or hsi.ndim != 4:
        raise ValueError(f"Expected rgb=[B,3,H,W], hsi=[B,L,H,W], got {rgb.shape}, {hsi.shape}")
    if rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB to have 3 channels, got {rgb.shape[1]}")
    if hsi.shape[1] != NUM_BANDS:
        raise ValueError(f"Expected HSI to have {NUM_BANDS} bands, got {hsi.shape[1]}")
    return rgb, hsi, name, orig_hw


def make_orig_hw_tensor(orig_hw: Optional[Any], hsi: torch.Tensor) -> torch.Tensor:
    batch_size = hsi.shape[0]
    if orig_hw is None:
        return torch.tensor([[hsi.shape[-2], hsi.shape[-1]]] * batch_size, dtype=torch.long)

    if torch.is_tensor(orig_hw):
        value = orig_hw.detach().cpu()
    else:
        value = torch.as_tensor(orig_hw, dtype=torch.long)

    if value.ndim == 1:
        if value.numel() == 2:
            value = value.view(1, 2).repeat(batch_size, 1)
        else:
            raise ValueError(f"orig_hw must contain H,W, got shape {tuple(value.shape)}")
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"orig_hw must have shape [B,2], got {tuple(value.shape)}")
    if value.shape[0] != batch_size:
        if value.shape[0] == 1:
            value = value.repeat(batch_size, 1)
        else:
            raise ValueError(f"orig_hw batch size {value.shape[0]} != tensor batch {batch_size}")
    return value.long()


def crop_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    orig_hw: torch.Tensor,
    sample_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    original_h = int(orig_hw[sample_index, 0].item())
    original_w = int(orig_hw[sample_index, 1].item())
    return (
        pred[sample_index : sample_index + 1, :, :original_h, :original_w],
        target[sample_index : sample_index + 1, :, :original_h, :original_w],
    )


# ==================================================
# DATA
# ==================================================


def make_dataloaders(device: torch.device) -> Tuple[Optional[DataLoader], DataLoader]:
    set_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader: Optional[DataLoader] = None
    if MODE == "train":
        train_dataset = ARADDataset(
            root_dir=DATA_ROOT,
            train=True,
            train_images=TRAIN_IMAGES,
            total_images=TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )
        if len(train_dataset) == 0:
            raise RuntimeError("Training dataset is empty. Check DATA_ROOT and ARAD file pairing.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda" and PIN_MEMORY),
            worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
            generator=generator,
            drop_last=False,
        )

    if MODE == "eval":
        val_dataset, _ = load_random_arad1k_samples(
            root_dir=DATA_ROOT,
            num_samples=EVAL_RANDOM_IMAGES,
            seed=VAL_SEED,
            total_images=EVAL_RANDOM_TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )
    else:
        val_dataset = ARADDataset(
            root_dir=DATA_ROOT,
            train=False,
            train_images=TRAIN_IMAGES,
            total_images=TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=False,
        )

    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Check TRAIN_IMAGES/TOTAL_IMAGES and file pairing.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda" and PIN_MEMORY),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader


# ==================================================
# CHECKPOINTS
# ==================================================


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    config: ModelConfig,
    loss_weights: LossWeights,
    best_val_mrae: float,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> None:
    ensure_checkpoint_dir()
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "model_config": config.to_dict(),
        "loss_weights": loss_weights.to_dict(),
        "best_val_mrae": best_val_mrae,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Union[str, Path], device: torch.device) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path} is not a valid checkpoint dictionary containing a 'model' state dict.")
    return checkpoint


# ==================================================
# MODEL CONSTRUCTION
# ==================================================


def build_model(device: torch.device, config: Optional[ModelConfig] = None) -> Tuple[UncertaintyGuidedHSIDiffusion, ModelConfig]:
    config = config or make_model_config()
    model = UncertaintyGuidedHSIDiffusion(config=config).to(device)

    if PRIOR_CHECKPOINT is not None:
        model.load_prior_checkpoint(
            str(PRIOR_CHECKPOINT),
            key=PRIOR_CHECKPOINT_KEY,
            strict=False,
            map_location=device,
        )
        if config.freeze_prior:
            model.freeze_prior()

    return model, config


def build_evaluation_model(checkpoint: Dict, device: torch.device) -> Tuple[UncertaintyGuidedHSIDiffusion, ModelConfig]:
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = UncertaintyGuidedHSIDiffusion(config=config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config


def build_criterion(device: torch.device, checkpoint: Optional[Dict] = None) -> Tuple[HSICompositeLoss, LossWeights]:
    if checkpoint is not None and "loss_weights" in checkpoint:
        weights = LossWeights.from_dict(checkpoint["loss_weights"])
    else:
        weights = make_loss_weights()
    camera_matrix = load_camera_matrix(CAMERA_MATRIX_PATH)
    if camera_matrix is not None:
        camera_matrix = camera_matrix.to(device)
    criterion = HSICompositeLoss(weights=weights, mrae_eps=MRAE_EPS, camera_matrix=camera_matrix).to(device)
    return criterion, weights


# ==================================================
# VALIDATION
# ==================================================


@torch.no_grad()
def validate(
    model: UncertaintyGuidedHSIDiffusion,
    criterion: HSICompositeLoss,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "psnr": 0.0,
        "sam": 0.0,
        "ssim": 0.0,
    }
    count = 0

    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(VAL_SEED)

    for batch in val_loader:
        rgb, hsi, _, orig_hw = unpack_batch(batch)
        orig_hw_tensor = make_orig_hw_tensor(orig_hw, hsi)
        rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
        hsi = hsi.to(device, non_blocking=(device.type == "cuda"))

        try:
            noise = torch.randn(
                rgb.shape[0],
                model.config.hsi_channels,
                rgb.shape[-2],
                rgb.shape[-1],
                device=device,
                generator=generator,
            )
        except TypeError:
            noise = torch.randn(
                rgb.shape[0],
                model.config.hsi_channels,
                rgb.shape[-2],
                rgb.shape[-1],
                device=device,
            )
        sample_output = model.sample(rgb, steps=SAMPLE_STEPS, noise=noise, return_all=True)
        pred_hsi = sample_output["pred"]

        for sample_index in range(rgb.shape[0]):
            sample_pred, sample_hsi = crop_sample(pred_hsi, hsi, orig_hw_tensor, sample_index)
            sample_loss = reconstruction_loss(
                sample_pred,
                sample_hsi,
                loss_type=RECONSTRUCTION_LOSS,
                mrae_eps=MRAE_EPS,
            )
            sample_metrics = compute_metrics(sample_pred, sample_hsi, mrae_eps=MRAE_EPS)
            totals["loss"] += float(sample_loss.item())
            for metric_name in ("mrae", "rmse", "psnr", "sam", "ssim"):
                totals[metric_name] += float(sample_metrics[metric_name])
            count += 1

    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return {name: value / count for name, value in totals.items()}


# ==================================================
# TRAINING
# ==================================================


def train() -> None:
    ensure_checkpoint_dir()
    set_seed(SEED)
    device = torch.device(DEVICE)
    train_loader, val_loader = make_dataloaders(device)
    if train_loader is None:
        raise RuntimeError("Training requested but train_loader is None")

    config = make_model_config()
    model, config = build_model(device, config=config)
    criterion, loss_weights = build_criterion(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found. Set FREEZE_PRIOR=False or enable denoiser training.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )
    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    start_epoch = 1
    best_val_mrae = math.inf
    best_val_loss = math.inf
    epochs_without_improvement = 0

    if RESUME_CHECKPOINT is not None:
        resume = load_checkpoint(RESUME_CHECKPOINT, device)
        resume_config = ModelConfig.from_dict(resume["model_config"])
        if resume_config.to_dict() != config.to_dict():
            raise ValueError("Resume checkpoint architecture differs from current CONFIG")
        model.load_state_dict(resume["model"], strict=True)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            lr_scheduler.load_state_dict(resume["scheduler"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        best_val_mrae = float(resume.get("best_val_mrae", math.inf))
        best_val_loss = float(resume.get("best_val_loss", math.inf))
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
        print(f"Resumed from epoch {start_epoch}")

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("Batch format is normalized with unpack_batch(); tuple/list/dict batches are supported.")
    print(f"Model configuration: {config.to_dict()}")
    print(f"Loss weights: {loss_weights.to_dict()}")
    if PRIOR_CHECKPOINT is not None:
        print(f"Loaded prior checkpoint: {PRIOR_CHECKPOINT}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        if config.freeze_prior:
            model.prior_net.eval()

        running_total = 0.0
        running_mrae = 0.0
        running_l1 = 0.0
        running_sam = 0.0
        running_uncertainty = 0.0
        running_rgb = 0.0
        running_smooth = 0.0
        train_count = 0

        for batch in train_loader:
            rgb, hsi, _, _ = unpack_batch(batch)
            rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
            hsi = hsi.to(device, non_blocking=(device.type == "cuda"))
            batch_size = rgb.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                outputs = model(rgb, hsi, detach_prior=config.freeze_prior)
                loss_dict = criterion(
                    pred=outputs["pred"],
                    target=hsi,
                    rgb=rgb,
                    x_det=outputs["x_det"],
                    uncertainty=outputs["uncertainty"],
                )
                total_loss = loss_dict["loss"]

            scaler.scale(total_loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_total += float(total_loss.item()) * batch_size
            running_mrae += float(loss_dict["mrae"].item()) * batch_size
            running_l1 += float(loss_dict["l1"].item()) * batch_size
            running_sam += float(loss_dict["sam"].item()) * batch_size
            running_uncertainty += float(loss_dict["uncertainty"].item()) * batch_size
            running_rgb += float(loss_dict["rgb"].item()) * batch_size
            running_smooth += float(loss_dict["spectral_smooth"].item()) * batch_size
            train_count += batch_size

        train_total = running_total / max(train_count, 1)
        train_mrae = running_mrae / max(train_count, 1)
        train_l1 = running_l1 / max(train_count, 1)
        train_sam = running_sam / max(train_count, 1)
        train_uncertainty = running_uncertainty / max(train_count, 1)
        train_rgb = running_rgb / max(train_count, 1)
        train_smooth = running_smooth / max(train_count, 1)

        val_results = validate(model, criterion, val_loader, device)
        lr_scheduler.step(val_results["mrae"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} "
            f"| Train Total {train_total:.6f} "
            f"| Train MRAE {train_mrae:.6f} "
            f"| Train L1 {train_l1:.6f} "
            f"| Train SAM {train_sam:.6f} "
            f"| Train Uncertainty {train_uncertainty:.6f} "
            f"| Train RGB {train_rgb:.6f} "
            f"| Train Smooth {train_smooth:.6f} "
            f"| Val Loss {val_results['loss']:.6f} "
            f"| Val MRAE {val_results['mrae']:.6f} "
            f"| Val RMSE {val_results['rmse']:.6f} "
            f"| Val SAM {val_results['sam']:.4f} "
            f"| Val PSNR {val_results['psnr']:.4f} "
            f"| Val SSIM {val_results['ssim']:.6f} "
            f"| LR {current_lr:.2e}"
        )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            save_checkpoint(
                BEST_LOSS_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                config=config,
                loss_weights=loss_weights,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )

        if val_results["mrae"] < best_val_mrae:
            best_val_mrae = val_results["mrae"]
            epochs_without_improvement = 0
            save_checkpoint(
                BEST_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                config=config,
                loss_weights=loss_weights,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )
            print(f"Saved best model (Val MRAE: {best_val_mrae:.6f})")
        else:
            epochs_without_improvement += 1
            print(f"No validation MRAE improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs")

        save_checkpoint(
            LATEST_PATH,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            config=config,
            loss_weights=loss_weights,
            best_val_mrae=best_val_mrae,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered. Best validation MRAE: {best_val_mrae:.6f}")
            break


# ==================================================
# EVALUATION
# ==================================================


def evaluate() -> None:
    ensure_checkpoint_dir()
    set_seed(SEED)
    device = torch.device(DEVICE)
    _, val_loader = make_dataloaders(device)
    selected_path = Path(EVAL_CHECKPOINT) if EVAL_CHECKPOINT is not None else BEST_PATH
    checkpoint = load_checkpoint(selected_path, device)
    model, config = build_evaluation_model(checkpoint, device)
    criterion, loss_weights = build_criterion(device, checkpoint=checkpoint)
    results = validate(model, criterion, val_loader, device)

    print(f"Evaluated checkpoint: {selected_path}")
    print(f"Model configuration: {config.to_dict()}")
    print(f"Loss weights: {loss_weights.to_dict()}")
    print(
        f"MRAE {results['mrae']:.6f} "
        f"| RMSE {results['rmse']:.6f} "
        f"| SAM {results['sam']:.4f} "
        f"| PSNR {results['psnr']:.4f} "
        f"| SSIM {results['ssim']:.6f}"
    )


# ==================================================
# MAIN
# ==================================================


def main() -> None:
    if MODE == "train":
        train()
    elif MODE == "eval":
        evaluate()
    else:
        raise ValueError("MODE must be 'train' or 'eval'")


if __name__ == "__main__":
    main()
