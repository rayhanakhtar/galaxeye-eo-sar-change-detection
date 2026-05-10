"""Dataset classes for EO-SAR change detection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


class JointTransform:
    """Apply aligned augmentations to EO, SAR, and mask tensors."""

    def __init__(
        self,
        random_flip: bool = False,
        random_rotate: bool = False,
        color_jitter: bool = False,
        brightness: float = 0.0,
        contrast: float = 0.0,
    ) -> None:
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.eo_jitter = (
            ColorJitter(brightness=brightness, contrast=contrast)
            if color_jitter
            else None
        )

    def __call__(
        self, pre_img: Tensor, post_img: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.random_flip and random.random() < 0.5:
            pre_img = torch.flip(pre_img, dims=[2])
            post_img = torch.flip(post_img, dims=[2])
            target = torch.flip(target, dims=[2])

        if self.random_flip and random.random() < 0.5:
            pre_img = torch.flip(pre_img, dims=[1])
            post_img = torch.flip(post_img, dims=[1])
            target = torch.flip(target, dims=[1])

        if self.random_rotate:
            k = random.randint(0, 3)
            if k:
                pre_img = torch.rot90(pre_img, k=k, dims=[1, 2])
                post_img = torch.rot90(post_img, k=k, dims=[1, 2])
                target = torch.rot90(target, k=k, dims=[1, 2])

        if self.eo_jitter is not None:
            pre_img = self.eo_jitter(pre_img)

        return pre_img.contiguous(), post_img.contiguous(), target.contiguous()


class ChangeDetectionDataset(Dataset):
    """PyTorch dataset for EO-SAR binary change detection."""

    def __init__(
        self,
        data_path: str,
        image_size: int = 256,
        transform: JointTransform | None = None,
        eo_mean: list[float] | None = None,
        eo_std: list[float] | None = None,
        sar_mean: list[float] | None = None,
        sar_std: list[float] | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.transform = transform

        self.eo_mean = torch.tensor(
            eo_mean or [0.485, 0.456, 0.406], dtype=torch.float32
        )
        self.eo_std = torch.tensor(eo_std or [0.229, 0.224, 0.225], dtype=torch.float32)
        self.sar_mean = torch.tensor(sar_mean or [0.5], dtype=torch.float32)
        self.sar_std = torch.tensor(sar_std or [0.25], dtype=torch.float32)

        self.pre_dir = self.data_path / "pre-event"
        self.post_dir = self.data_path / "post-event"
        self.target_dir = self.data_path / "target"

        self.file_names = sorted(f.name for f in self.pre_dir.glob("*.tif"))
        if not self.file_names:
            raise FileNotFoundError(f"No .tif files found in {self.pre_dir}")

        post_names = {f.name for f in self.post_dir.glob("*.tif")}
        target_names = {f.name for f in self.target_dir.glob("*.tif")}
        missing_post = sorted(set(self.file_names) - post_names)
        missing_target = sorted(set(self.file_names) - target_names)
        if missing_post or missing_target:
            raise ValueError(
                "Dataset split has mismatched filenames: "
                f"missing_post={missing_post[:3]}, missing_target={missing_target[:3]}"
            )

    def __len__(self) -> int:
        return len(self.file_names)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        return tifffile.imread(str(path))

    @staticmethod
    def _remap_labels(mask: np.ndarray) -> np.ndarray:
        return np.where(mask >= 2, 1, 0).astype(np.int64)

    def _resize_tensor(self, tensor: Tensor, mode: str) -> Tensor:
        if tensor.shape[-2:] == (self.image_size, self.image_size):
            return tensor

        kwargs: dict[str, Any] = {
            "size": (self.image_size, self.image_size),
            "mode": mode,
        }
        if mode != "nearest":
            kwargs["align_corners"] = False

        return F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)

    def _normalize(self, img: Tensor, is_eo: bool) -> Tensor:
        if is_eo:
            mean = self.eo_mean.view(-1, 1, 1)
            std = self.eo_std.view(-1, 1, 1)
        else:
            mean = self.sar_mean.view(-1, 1, 1)
            std = self.sar_std.view(-1, 1, 1)
        return (img - mean) / std

    def __getitem__(self, idx: int) -> dict[str, Any]:
        filename = self.file_names[idx]

        pre_img = self._load_image(self.pre_dir / filename)
        post_img = self._load_image(self.post_dir / filename)
        target = self._load_image(self.target_dir / filename)

        if pre_img.ndim != 3 or pre_img.shape[2] != 3:
            raise ValueError(
                f"Expected RGB EO image for {filename}, got shape {pre_img.shape}"
            )
        if post_img.ndim != 2:
            raise ValueError(
                f"Expected grayscale SAR image for {filename}, got shape {post_img.shape}"
            )
        if target.ndim != 2:
            raise ValueError(
                f"Expected 2D mask for {filename}, got shape {target.shape}"
            )

        target = self._remap_labels(target)

        pre_tensor = torch.from_numpy(pre_img).permute(2, 0, 1).float() / 255.0
        post_tensor = torch.from_numpy(post_img).unsqueeze(0).float() / 255.0
        target_tensor = torch.from_numpy(target).unsqueeze(0).float()

        pre_tensor = self._resize_tensor(pre_tensor, mode="bilinear")
        post_tensor = self._resize_tensor(post_tensor, mode="bilinear")
        target_tensor = self._resize_tensor(target_tensor, mode="nearest").long()

        invalid_mask = (
            (pre_tensor.sum(dim=0, keepdim=True) == 0.0) | (post_tensor == 0.0)
        ).float()

        if self.transform is not None:
            pre_tensor, post_tensor, target_tensor = self.transform(
                pre_tensor, post_tensor, target_tensor
            )

        pre_tensor = self._normalize(pre_tensor, is_eo=True)
        post_tensor = self._normalize(post_tensor, is_eo=False)

        return {
            "pre_img": pre_tensor,
            "post_img": post_tensor,
            "target": target_tensor,
            "invalid_mask": invalid_mask,
            "filename": filename,
        }
