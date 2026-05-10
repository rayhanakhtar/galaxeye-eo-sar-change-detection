"""Loss functions for class-imbalanced change detection.

Combines Focal Loss and Dice Loss to handle severe class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal loss down-weights well-classified examples and focuses on hard examples.
    Reference: Lin et al., ICCV 2017

    Args:
        alpha: Weighting factor for each class
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction mode ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float | torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predictions (before softmax) - shape (B, C, H, W)
            targets: Ground truth - shape (B, H, W)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for binary/multi-class segmentation.

    Directly optimizes for IoU/Dice coefficient.

    Args:
        smooth: Smoothing constant to avoid division by zero
        reduction: Reduction mode ('mean', 'sum', 'none')
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            inputs: Predictions (before softmax) - shape (B, C, H, W)
            targets: Ground truth - shape (B, H, W)

        Returns:
            Dice loss value
        """
        # Get predictions for positive class (change)
        probs = F.softmax(inputs, dim=1)

        # One-hot encode targets
        num_classes = inputs.shape[1]
        targets_one_hot = (
            F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        )

        # Compute Dice coefficient for each class
        dims = (0, 2, 3)

        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = (probs + targets_one_hot).sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        dice_loss = 1.0 - dice_score

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice Loss for class-imbalanced change detection.

    Combines benefits of both losses:
    - Focal: Focuses on hard/misclassified pixels
    - Dice: Directly optimizes for IoU

    Args:
        focal_weight: Weight for focal loss component
        dice_weight: Weight for dice loss component
        focal_gamma: Focusing parameter for focal loss
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal = FocalLoss(gamma=focal_gamma, reduction="mean")
        self.dice = DiceLoss(reduction="mean")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            inputs: Predictions - shape (B, C, H, W)
            targets: Ground truth - shape (B, H, W)

        Returns:
            Combined loss value
        """
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


def build_loss(config: dict) -> FocalDiceLoss:
    """Build loss function from config.

    Args:
        config: Configuration dictionary

    Returns:
        FocalDiceLoss instance
    """
    loss_config = config.get("training", {}).get("loss", {})

    return FocalDiceLoss(
        focal_weight=loss_config.get("focal_weight", 0.5),
        dice_weight=loss_config.get("dice_weight", 0.5),
        focal_gamma=loss_config.get("focal_gamma", 2.0),
    )
