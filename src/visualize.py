"""Visualization script for change detection predictions."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import tifffile

from src.data.dataset import ChangeDetectionDataset
from src.models.siamese_unet import SiameseUNet


def visualize_predictions(
    config: dict,
    data_path: str,
    weights_path: str,
    output_dir: str,
    num_samples: int = 10
) -> None:
    """Visualize predictions on test set."""
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    test_dataset = ChangeDetectionDataset(
        data_path=data_path,
        image_size=config["data"]["image_size"]
    )
    
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
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Visualize samples
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    
    if len(indices) == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = test_dataset[idx]
            
            pre_img = sample["pre_img"].unsqueeze(0).to(device)
            post_img = sample["post_img"].unsqueeze(0).to(device)
            
            # Get prediction
            output = model(pre_img, post_img)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            target = sample["target"].squeeze().cpu().numpy()
            
            # Denormalize for visualization
            pre_vis = sample["pre_img"].cpu().numpy()
            if pre_vis.shape[0] == 3:
                # RGB
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                pre_vis = (pre_vis * std + mean).transpose(1, 2, 0)
                pre_vis = np.clip(pre_vis, 0, 1)
            else:
                # Grayscale
                pre_vis = pre_vis[0]
            
            # Visualize
            axes[i, 0].imshow(pre_vis, cmap="gray")
            axes[i, 0].set_title("Pre-event (EO)")
            axes[i, 0].axis("off")
            
            post_vis = sample["post_img"].cpu().numpy()
            if post_vis.shape[0] == 1:
                post_vis = post_vis[0]
            axes[i, 1].imshow(post_vis, cmap="gray")
            axes[i, 1].set_title("Post-event (SAR)")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(target, cmap="gray")
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis("off")
            
            axes[i, 3].imshow(pred, cmap="gray")
            axes[i, 3].set_title("Prediction")
            axes[i, 3].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path / "predictions.png", dpi=150, bbox_inches="tight")
    print(f"Saved visualizations to {output_path / 'predictions.png'}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize change detection predictions")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    visualize_predictions(config, args.data_path, args.weights, args.output, args.num_samples)


if __name__ == "__main__":
    main()