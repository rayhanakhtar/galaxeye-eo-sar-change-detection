"""Training script for EO-SAR binary change detection."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import ChangeDetectionDataset, JointTransform
from src.models.siamese_unet import build_model
from src.utils.losses import build_loss
from src.utils.metrics import (
    average_loss,
    compute_confusion_matrix,
    logits_to_prediction,
    metrics_from_confusion_matrix,
)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_datasets(
    config: dict,
) -> tuple[ChangeDetectionDataset, ChangeDetectionDataset]:
    """Build train and validation datasets from config."""
    data_config = config["data"]
    norm_config = config.get("normalization", {})
    aug_config = config.get("augmentation", {})
    train_aug = aug_config.get("train", {})

    train_transform = JointTransform(
        random_flip=train_aug.get("random_flip", False),
        random_rotate=train_aug.get("random_rotate", False),
        color_jitter=train_aug.get("color_jitter", False),
        brightness=train_aug.get("brightness", 0.0),
        contrast=train_aug.get("contrast", 0.0),
    )

    common_kwargs = {
        "image_size": data_config["image_size"],
        "eo_mean": norm_config.get("eo_mean"),
        "eo_std": norm_config.get("eo_std"),
        "sar_mean": norm_config.get("sar_mean"),
        "sar_std": norm_config.get("sar_std"),
    }

    train_dataset = ChangeDetectionDataset(
        data_path=f"{data_config['dataset_path']}/{data_config['train_split']}",
        transform=train_transform,
        **common_kwargs,
    )
    val_dataset = ChangeDetectionDataset(
        data_path=f"{data_config['dataset_path']}/{data_config['val_split']}",
        transform=None,
        **common_kwargs,
    )
    return train_dataset, val_dataset


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    scaler: GradScaler | None,
    amp_enabled: bool,
    desc: str,
) -> dict:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    conf_matrix = torch.zeros(2, 2, dtype=torch.long)

    progress = tqdm(loader, desc=desc)
    for batch in progress:
        pre_img = batch["pre_img"].to(device, non_blocking=True)
        post_img = batch["post_img"].to(device, non_blocking=True)
        target = batch["target"].squeeze(1).to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(pre_img, post_img)
            loss = criterion(logits, target)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        pred = logits_to_prediction(logits.detach())
        conf_matrix += compute_confusion_matrix(pred.cpu(), target.detach().cpu())
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = metrics_from_confusion_matrix(conf_matrix)
    metrics["loss"] = average_loss(total_loss, len(loader))
    return metrics


def train(config: dict, logger: logging.Logger) -> Path | None:
    """Main training routine."""
    requested_device = config.get("device", "cuda")
    device = torch.device(
        requested_device
        if requested_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    train_dataset, val_dataset = build_datasets(config)
    training_config = config["training"]
    num_workers = training_config.get("num_workers", 0)
    pin_memory = device.type == "cuda"

    debug_samples = training_config.get("debug_samples")
    if debug_samples:
        train_dataset = Subset(
            train_dataset, range(min(debug_samples, len(train_dataset)))
        )
        val_dataset = Subset(val_dataset, range(min(debug_samples, len(val_dataset))))
        logger.info("Debug mode enabled with %s samples per split", debug_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "Train samples: %s, Val samples: %s", len(train_dataset), len(val_dataset)
    )

    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config["num_epochs"],
        eta_min=1e-6,
    )

    amp_enabled = bool(training_config.get("amp", False) and device.type == "cuda")
    scaler = GradScaler("cuda") if amp_enabled else None

    patience = training_config["early_stopping"]["patience"]
    min_delta = training_config["early_stopping"]["min_delta"]
    best_val_iou = -1.0
    epochs_no_improve = 0

    checkpoint_dir = Path(config["checkpoint"]["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pth"

    for epoch in range(1, training_config["num_epochs"] + 1):
        logger.info("Epoch %s/%s", epoch, training_config["num_epochs"])
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            amp_enabled,
            f"Train {epoch}",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            None,
            False,
            "Validation",
        )
        scheduler.step()

        logger.info(
            "Train - Loss: %.4f | IoU: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            train_metrics["loss"],
            train_metrics["iou"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1"],
        )
        logger.info(
            "Val   - Loss: %.4f | IoU: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            val_metrics["loss"],
            val_metrics["iou"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
        )

        if val_metrics["iou"] > best_val_iou + min_delta:
            best_val_iou = val_metrics["iou"]
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                best_path,
            )
            logger.info("Saved best model to %s", best_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered after %s epochs", epoch)
            break

    logger.info("Training completed. Best Val IoU: %.4f", best_val_iou)
    return best_path if best_path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EO-SAR change detection model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(Path("logs"))
    set_seed(config.get("seed", 42))
    train(config, logger)


if __name__ == "__main__":
    main()
