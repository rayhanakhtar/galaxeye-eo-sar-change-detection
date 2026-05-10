"""Visualization script for change detection predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.dataset import ChangeDetectionDataset
from src.models.siamese_unet import build_model
from src.utils.metrics import logits_to_prediction


def denormalize_eo(
    image: np.ndarray, mean: list[float], std: list[float]
) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    image = image * std_arr + mean_arr
    return np.clip(image.transpose(1, 2, 0), 0.0, 1.0)


def denormalize_sar(
    image: np.ndarray, mean: list[float], std: list[float]
) -> np.ndarray:
    sar = image[0] * std[0] + mean[0]
    return np.clip(sar, 0.0, 1.0)


def visualize_predictions(
    config: dict,
    data_path: str,
    weights_path: str,
    output_dir: str,
    num_samples: int = 10,
) -> None:
    requested_device = config.get("device", "cuda")
    device = torch.device(
        requested_device
        if requested_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    norm_config = config.get("normalization", {})

    dataset = ChangeDetectionDataset(
        data_path=data_path,
        image_size=config["data"]["image_size"],
        eo_mean=norm_config.get("eo_mean"),
        eo_std=norm_config.get("eo_std"),
        sar_mean=norm_config.get("sar_mean"),
        sar_std=norm_config.get("sar_std"),
    )

    model = build_model(config).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_count = min(num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=int)
    fig, axes = plt.subplots(sample_count, 4, figsize=(16, 4 * sample_count))
    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[int(idx)]
            pre_img = sample["pre_img"].unsqueeze(0).to(device)
            post_img = sample["post_img"].unsqueeze(0).to(device)
            logits = model(pre_img, post_img)
            pred = logits_to_prediction(logits).squeeze(0).cpu().numpy()
            target = sample["target"].squeeze(0).cpu().numpy()

            pre_vis = denormalize_eo(
                sample["pre_img"].cpu().numpy(),
                norm_config.get("eo_mean", [0.485, 0.456, 0.406]),
                norm_config.get("eo_std", [0.229, 0.224, 0.225]),
            )
            post_vis = denormalize_sar(
                sample["post_img"].cpu().numpy(),
                norm_config.get("sar_mean", [0.5]),
                norm_config.get("sar_std", [0.25]),
            )

            axes[row, 0].imshow(pre_vis)
            axes[row, 0].set_title("Pre-event EO")
            axes[row, 1].imshow(post_vis, cmap="gray")
            axes[row, 1].set_title("Post-event SAR")
            axes[row, 2].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title("Ground Truth")
            axes[row, 3].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].set_title("Prediction")

            for col in range(4):
                axes[row, col].axis("off")

    plt.tight_layout()
    save_path = output_path / "predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualizations to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize change detection predictions"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset split"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to visualize"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    visualize_predictions(
        config, args.data_path, args.weights, args.output, args.num_samples
    )


if __name__ == "__main__":
    main()
