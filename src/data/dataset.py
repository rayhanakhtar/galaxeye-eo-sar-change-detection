"""Dataset classes for EO-SAR change detection."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import tifffile
from PIL import Image


class ChangeDetectionDataset(Dataset):
    """PyTorch Dataset for EO-SAR binary change detection.
    
    Loads pre-event (EO), post-event (SAR) image pairs with binary change masks.
    
    Args:
        data_path: Path to dataset split (train/val/test)
        image_size: Target size for resizing
        transform: Optional transform to apply
        eo_mean: Mean values for EO (RGB) normalization
        eo_std: Std values for EO (RGB) normalization
        sar_mean: Mean value for SAR normalization
        sar_std: Std value for SAR normalization
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 256,
        transform: Callable | None = None,
        eo_mean: list[float] | None = None,
        eo_std: list[float] | None = None,
        sar_mean: list[float] | None = None,
        sar_std: list[float] | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.transform = transform
        
        # Default normalization values (ImageNet for EO)
        self.eo_mean = torch.tensor(eo_mean or [0.485, 0.456, 0.406])
        self.eo_std = torch.tensor(eo_std or [0.229, 0.224, 0.225])
        self.sar_mean = torch.tensor(sar_mean or [0.5])
        self.sar_std = torch.tensor(sar_std or [0.25])
        
        # Get file lists
        self.pre_dir = self.data_path / "pre-event"
        self.post_dir = self.data_path / "post-event"
        self.target_dir = self.data_path / "target"
        
        # Get sorted list of files
        self.file_names = sorted([
            f.name for f in self.pre_dir.glob("*.tif")
        ])
        
    def __len__(self) -> int:
        return len(self.file_names)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image and convert to float32."""
        img = tifffile.imread(str(path))
        if len(img.shape) == 2:
            # Grayscale - expand to 3 channels if needed
            img = img.astype(np.float32)
        else:
            img = img.astype(np.float32)
        return img
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img = np.array(pil_img)
        return img
    
    def _normalize(self, img: Tensor, is_eo: bool) -> Tensor:
        """Normalize image tensor."""
        if is_eo:
            # EO: 3-channel, use ImageNet stats
            mean = self.eo_mean.view(3, 1, 1)
            std = self.eo_std.view(3, 1, 1)
        else:
            # SAR: 1-channel
            mean = self.sar_mean.view(1, 1, 1)
            std = self.sar_std.view(1, 1, 1)
        
        return (img - mean) / std
    
    def _remap_labels(self, mask: np.ndarray) -> np.ndarray:
        """Remap 4-class labels to binary.
        
        Original: 0=Background, 1=Intact, 2=Damaged, 3=Destroyed
        Remapped: 0=No-Change (0,1), 1=Change (2,3)
        """
        return np.where(mask >= 2, 1, 0).astype(np.int64)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.
        
        Returns:
            dict with keys: 'pre_img', 'post_img', 'target', 'filename'
        """
        filename = self.file_names[idx]
        
        # Load images
        pre_img = self._load_image(self.pre_dir / filename)
        post_img = self._load_image(self.post_dir / filename)
        target = self._load_image(self.target_dir / filename)
        
        # Resize
        pre_img = self._resize_image(pre_img)
        post_img = self._resize_image(post_img)
        target = self._resize_image(target)
        
        # Remap labels
        target = self._remap_labels(target)
        
        # Convert to tensors
        # Handle channels
        if len(pre_img.shape) == 2:
            # Grayscale - expand to 3 for EO branch
            pre_tensor = torch.from_numpy(pre_img).unsqueeze(0).float()
        else:
            # RGB
            pre_tensor = torch.from_numpy(pre_img).permute(2, 0, 1).float()
        
        if len(post_img.shape) == 2:
            post_tensor = torch.from_numpy(post_img).unsqueeze(0).float()
        else:
            post_tensor = torch.from_numpy(post_img).permute(2, 0, 1).float()
        
        target_tensor = torch.from_numpy(target).unsqueeze(0).long()
        
        # Normalize
        if pre_tensor.shape[0] == 3:
            pre_tensor = self._normalize(pre_tensor, is_eo=True)
        else:
            pre_tensor = self._normalize(pre_tensor, is_eo=False)
            
        post_tensor = self._normalize(post_tensor, is_eo=False)
        
        # Apply transforms if any
        if self.transform:
            pre_tensor, post_tensor, target_tensor = self.transform(
                pre_tensor, post_tensor, target_tensor
            )
        
        return {
            "pre_img": pre_tensor,
            "post_img": post_tensor,
            "target": target_tensor,
            "filename": filename
        }