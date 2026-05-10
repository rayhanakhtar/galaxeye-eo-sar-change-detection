"""Siamese UNet model for EO-SAR binary change detection.

Architecture: Dual-encoder Siamese network with UNet-style decoder
- EO encoder: Pretrained ResNet34 for 3-channel RGB input
- SAR encoder: Pretrained ResNet34 for 1-channel grayscale input
- Fusion: Feature concatenation at each scale
- Decoder: Upsampling with attention gates and skip connections
- Output: Binary change mask
"""

from typing import list

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


class AttentionGate(nn.Module):
    """Attention Gate module for feature refinement.
    
    Args:
        g_channels: Number of gates (decoder) channels
        x_channels: Number of skip connection (encoder) channels
        inter_channels: Number of intermediate channels
    """
    
    def __init__(self, g_channels: int, x_channels: int, inter_channels: int) -> None:
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply attention gate.
        
        Args:
            g: Gating signal from decoder
            x: Skip connection from encoder
            
        Returns:
            Attention-weighted features
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ConvBlock(nn.Module):
    """Basic convolutional block with BN and activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        activation: Activation function
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu"
    ) -> None:
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if activation == "relu" else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and attention gate.
    
    Args:
        in_channels: Number of input channels
        skip_channels: Number of skip connection channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.attn = AttentionGate(out_channels, skip_channels, out_channels // 2)
        
        self.conv = nn.Sequential(
            ConvBlock(out_channels + skip_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input from previous layer
            skip: Skip connection from encoder
            
        Returns:
            Output features
        """
        x = self.up(x)
        
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        
        x = self.attn(x, skip)
        
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class SiameseUNet(nn.Module):
    """Siamese UNet for binary change detection.
    
    Dual-encoder architecture with separate encoders for EO (RGB) and SAR (grayscale).
    
    Args:
        eo_channels: Number of channels in EO input (default: 3)
        sar_channels: Number of channels in SAR input (default: 1)
        pretrained: Use pretrained ImageNet weights (default: True)
    """
    
    def __init__(
        self,
        eo_channels: int = 3,
        sar_channels: int = 1,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        
        # Create encoders with shared architecture but separate weights
        self.eo_encoder = self._create_encoder(eo_channels, pretrained)
        self.sar_encoder = self._create_encoder(sar_channels, pretrained)
        
        # Store encoder output channels for decoder
        encoder_channels = [64, 128, 256, 512]
        
        # Decoder
        self.decoder4 = DecoderBlock(512 + 512, 256 + 256, 256)
        self.decoder3 = DecoderBlock(256, 128 + 128, 128)
        self.decoder2 = DecoderBlock(128, 64 + 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)
        
        # Final output layer
        self.output = nn.Conv2d(32, 2, kernel_size=1)
        
    def _create_encoder(self, in_channels: int, pretrained: bool) -> nn.Module:
        """Create ResNet34 encoder.
        
        Args:
            in_channels: Number of input channels
            pretrained: Use pretrained weights
            
        Returns:
            Encoder network
        """
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)
        
        # Modify first conv layer for custom input channels
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize new conv with mean of original
        if pretrained and in_channels != 3:
            with torch.no_grad():
                resnet.conv1.weight[:, :3] = original_conv.weight
        
        # Remove final FC and avgpool
        features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,           # 1/4
            resnet.layer1,            # 1/4 -> 64
            resnet.layer2,            # 1/8 -> 128
            resnet.layer3,            # 1/16 -> 256
            resnet.layer4,            # 1/32 -> 512
        )
        
        return features
    
    def _get_encoder_features(self, encoder: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features from encoder.
        
        Args:
            encoder: Encoder network
            x: Input tensor
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # After conv1 + bn1 + relu + maxpool
        x = encoder[0](x)
        x = encoder[1](x)
        x = encoder[2](x)
        x = encoder[3](x)
        features.append(x)
        
        # layer1
        x = encoder[4](x)
        features.append(x)
        
        # layer2
        x = encoder[5](x)
        features.append(x)
        
        # layer3
        x = encoder[6](x)
        features.append(x)
        
        # layer4
        x = encoder[7](x)
        features.append(x)
        
        return features
    
    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pre_img: Pre-event image (EO, RGB) - shape (B, 3, H, W)
            post_img: Post-event image (SAR, grayscale) - shape (B, 1, H, W)
            
        Returns:
            Binary change mask - shape (B, 2, H, W)
        """
        # Extract features from both encoders
        eo_feats = self._get_encoder_features(self.eo_encoder, pre_img)
        sar_feats = self._get_encoder_features(self.sar_encoder, post_img)
        
        # Concatenate features at each scale
        fused = [
            torch.cat([eo, sar], dim=1)
            for eo, sar in zip(eo_feats, sar_feats)
        ]
        
        # Decoder
        # f4: 512+512 -> 256
        x = self.decoder4(fused[4], fused[3])
        
        # f3: 256 -> 128
        x = self.decoder3(x, fused[2])
        
        # f2: 128 -> 64
        x = self.decoder2(x, fused[1])
        
        # f1: 64 -> 32
        x = self.decoder1(x, fused[0])
        
        # Output
        x = self.output(x)
        
        return x


def build_model(config: dict) -> SiameseUNet:
    """Build SiameseUNet model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SiameseUNet model
    """
    model_config = config.get("model", {})
    
    return SiameseUNet(
        eo_channels=model_config.get("eo_channels", 3),
        sar_channels=model_config.get("sar_channels", 1),
        pretrained=model_config.get("pretrained", True)
    )