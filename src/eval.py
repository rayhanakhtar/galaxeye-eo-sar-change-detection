"""Evaluation script for EO-SAR binary change detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ChangeDetectionDataset
from src.models.siamese_unet import build_model
from src.utils.metrics import (
    compute_confusion_matrix,
    logits_to_prediction,
    metrics_from_confusion_matrix,
    metrics_to_numpy,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate(
    config: dict, data_path: str, weights_path: str, output_dir: str = "results"
) -> dict:
    requested_device = config.get("device", "cuda")
    device = torch.device(
        requested_device
        if requested_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    norm_config = config.get("normalization", {})
    dataset = ChangeDetectionDataset(
        data_path=data_path,
        image_size=config["data"]["image_size"],
        eo_mean=norm_config.get("eo_mean"),
        eo_std=norm_config.get("eo_std"),
        sar_mean=norm_config.get("sar_mean"),
        sar_std=norm_config.get("sar_std"),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    conf_matrix = torch.zeros(2, 2, dtype=torch.long)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pre_img = batch["pre_img"].to(device, non_blocking=True)
            post_img = batch["post_img"].to(device, non_blocking=True)
            target = batch["target"].squeeze(1)
            logits = model(pre_img, post_img)
            pred = logits_to_prediction(logits).cpu()
            conf_matrix += compute_confusion_matrix(pred, target)

    metrics = metrics_from_confusion_matrix(conf_matrix)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_file = output_path / "evaluation_metrics.json"
    metrics_file.write_text(
        json.dumps(metrics_to_numpy(metrics), indent=2, default=lambda x: x.tolist()),
        encoding="utf-8",
    )

    print("\nEvaluation Results:")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Saved:     {metrics_file}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EO-SAR change detection model"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.data_path, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
