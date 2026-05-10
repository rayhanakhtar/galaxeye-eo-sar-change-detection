"""Evaluation script for EO-SAR binary change detection."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.data.dataset import ChangeDetectionDataset
from src.models.siamese_unet import SiameseUNet


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> dict:
    """Compute evaluation metrics for change detection."""
    pred_labels = pred.argmax(dim=1)
    
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for t, p in zip(target.view(-1), pred_labels.view(-1)):
        conf_matrix[t.long(), p.long()] += 0
    
    tn = conf_matrix[0, 0].item()
    fp = conf_matrix[0, 1].item()
    fn = conf_matrix[1, 0].item()
    tp = conf_matrix[1, 1].item()
    
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conf_matrix": [[tn, fp], [fn, tp]]
    }


def evaluate(config: dict, data_path: str, weights_path: str) -> dict:
    """Evaluate model on test set."""
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    test_dataset = ChangeDetectionDataset(
        data_path=data_path,
        image_size=config["data"]["image_size"]
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    model = SiameseUNet(
        eo_channels=config["model"]["eo_channels"],
        sar_channels=config["model"]["sar_channels"],
        pretrained=False
    )
    
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_metrics = {"iou": [], "precision": [], "recall": [], "f1": []}
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pre_img = batch["pre_img"].to(device)
            post_img = batch["post_img"].to(device)
            target = batch["target"].squeeze(1).to(device)
            
            outputs = model(pre_img, post_img)
            
            metrics = compute_metrics(outputs, target)
            
            for k in ["iou", "precision", "recall", "f1"]:
                all_metrics[k].append(metrics[k])
            
            all_predictions.append(outputs.argmax(dim=1).cpu())
            all_targets.append(target.cpu())
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    print("\nEvaluation Results:")
    print(f"  IoU:       {avg_metrics['iou']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall:    {avg_metrics['recall']:.4f}")
    print(f"  F1 Score:  {avg_metrics['f1']:.4f}")
    
    return avg_metrics


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate EO-SAR change detection model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    evaluate(config, args.data_path, args.weights)


if __name__ == "__main__":
    main()