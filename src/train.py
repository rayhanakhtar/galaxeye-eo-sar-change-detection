"""Training script for EO-SAR binary change detection."""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm

from src.data.dataset import ChangeDetectionDataset
from src.models.siamese_unet import SiameseUNet, build_model
from src.utils.losses import FocalDiceLoss, build_loss


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_dir / "training.log")
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
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
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> dict:
    """Compute evaluation metrics for change detection.
    
    Args:
        pred: Predicted logits - shape (B, C, H, W)
        target: Ground truth - shape (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Dictionary with IoU, Precision, Recall, F1
    """
    pred_labels = pred.argmax(dim=1)
    
    # Initialize confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for t, p in zip(target.view(-1), pred_labels.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    
    # Calculate metrics for change class (class 1)
    tn = conf_matrix[0, 0].item()
    fp = conf_matrix[0, 1].item()
    fn = conf_matrix[1, 0].item()
    tp = conf_matrix[1, 1].item()
    
    # IoU for change class
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Precision
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    
    # Recall
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix.numpy()
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_metrics = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        pre_img = batch["pre_img"].to(device)
        post_img = batch["post_img"].to(device)
        target = batch["target"].squeeze(1).to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pre_img, post_img)
        loss = criterion(outputs, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(outputs, target)
            for k, v in total_metrics.items():
                if k != "confusion_matrix":
                    total_metrics[k] += v
        
        pbar.set_postfix({"loss": loss.item()})
    
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    
    for k in total_metrics:
        if k != "confusion_matrix":
            total_metrics[k] /= n_batches
    
    return {"loss": avg_loss, **total_metrics}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger
) -> dict:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_metrics = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            pre_img = batch["pre_img"].to(device)
            post_img = batch["post_img"].to(device)
            target = batch["target"].squeeze(1).to(device)
            
            outputs = model(pre_img, post_img)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
            
            metrics = compute_metrics(outputs, target)
            for k, v in total_metrics.items():
                if k != "confusion_matrix":
                    total_metrics[k] += v
    
    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    
    for k in total_metrics:
        if k != "confusion_matrix":
            total_metrics[k] /= n_batches
    
    return {"loss": avg_loss, **total_metrics}


def train(config: dict, logger: logging.Logger) -> None:
    """Main training function."""
    # Setup device
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets
    data_config = config["data"]
    train_dataset = ChangeDetectionDataset(
        data_path=f"{data_config['dataset_path']}/{data_config['train_split']}",
        image_size=data_config["image_size"]
    )
    val_dataset = ChangeDetectionDataset(
        data_path=f"{data_config['dataset_path']}/{data_config['val_split']}",
        image_size=data_config["image_size"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = build_model(config)
    model = model.to(device)
    logger.info(f"Model: {config['model']['name']}")
    
    # Loss and optimizer
    criterion = build_loss(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
        eta_min=1e-6
    )
    
    # Early stopping
    early_stop_config = config["training"]["early_stopping"]
    patience = early_stop_config["patience"]
    min_delta = early_stop_config["min_delta"]
    best_val_iou = 0.0
    epochs_no_improve = 0
    
    # Checkpointing
    checkpoint_config = config["checkpoint"]
    checkpoint_dir = Path(checkpoint_config["save_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch
        )
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"IoU: {train_metrics['iou']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, logger)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"IoU: {val_metrics['iou']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Check for improvement
        if val_metrics["iou"] > best_val_iou + min_delta:
            best_val_iou = val_metrics["iou"]
            epochs_no_improve = 0
            
            # Save best model
            best_path = checkpoint_dir / f"best_model_iou{best_val_iou:.4f}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": best_val_iou,
                "val_f1": val_metrics["f1"]
            }, best_path)
            logger.info(f"Saved best model to {best_path}")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    logger.info(f"Training completed. Best Val IoU: {best_val_iou:.4f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train EO-SAR change detection model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    log_dir = Path("logs")
    logger = setup_logging(log_dir)
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Train
    train(config, logger)


if __name__ == "__main__":
    main()