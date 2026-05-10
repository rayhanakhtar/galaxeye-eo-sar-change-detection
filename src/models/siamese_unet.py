"""Dual-encoder EO-SAR UNet model for binary change detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


class AttentionGate(nn.Module):
    """Attention gate for refining skip features."""

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.relu(self.w_g(g) + self.w_x(x)))


class ConvBlock(nn.Module):
    """Two-layer conv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample, attend to skip features, and fuse."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionGate(
            out_channels, skip_channels, max(out_channels // 2, 1)
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        attn_skip = self.attn(x, skip)
        return self.conv(torch.cat([x, attn_skip], dim=1))


class OutputHead(nn.Module):
    """Final full-resolution prediction head."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, 32, kernel_size=2, stride=2)
        self.refine = ConvBlock(32, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.up(x)
        x = self.refine(x)
        if x.shape[2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.out(x)


class DualEncoderEOSARUNet(nn.Module):
    """Dual-encoder EO-SAR UNet with separate EO and SAR backbones."""

    def __init__(
        self,
        eo_channels: int = 3,
        sar_channels: int = 1,
        pretrained: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.eo_encoder = self._create_encoder(eo_channels, pretrained)
        self.sar_encoder = self._create_encoder(sar_channels, pretrained)

        self.decoder4 = DecoderBlock(1024, 512, 256)
        self.decoder3 = DecoderBlock(256, 256, 128)
        self.decoder2 = DecoderBlock(128, 128, 64)
        self.decoder1 = DecoderBlock(64, 128, 64)
        self.output_head = OutputHead(64, num_classes)

    @staticmethod
    def _create_encoder(in_channels: int, pretrained: bool) -> nn.ModuleDict:
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            if pretrained and in_channels == 3:
                resnet.conv1.weight.copy_(original_conv.weight)
            elif pretrained and in_channels == 1:
                resnet.conv1.weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
            else:
                nn.init.kaiming_normal_(
                    resnet.conv1.weight, mode="fan_out", nonlinearity="relu"
                )

        return nn.ModuleDict(
            {
                "conv1": resnet.conv1,
                "bn1": resnet.bn1,
                "relu": resnet.relu,
                "maxpool": resnet.maxpool,
                "layer1": resnet.layer1,
                "layer2": resnet.layer2,
                "layer3": resnet.layer3,
                "layer4": resnet.layer4,
            }
        )

    @staticmethod
    def _get_encoder_features(
        encoder: nn.ModuleDict, x: torch.Tensor
    ) -> list[torch.Tensor]:
        x = encoder["conv1"](x)
        x = encoder["bn1"](x)
        stem = encoder["relu"](x)
        x = encoder["maxpool"](stem)
        feat1 = encoder["layer1"](x)
        feat2 = encoder["layer2"](feat1)
        feat3 = encoder["layer3"](feat2)
        feat4 = encoder["layer4"](feat3)
        return [stem, feat1, feat2, feat3, feat4]

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        eo_feats = self._get_encoder_features(self.eo_encoder, pre_img)
        sar_feats = self._get_encoder_features(self.sar_encoder, post_img)
        fused = [torch.cat([eo, sar], dim=1) for eo, sar in zip(eo_feats, sar_feats)]

        x = self.decoder4(fused[4], fused[3])
        x = self.decoder3(x, fused[2])
        x = self.decoder2(x, fused[1])
        x = self.decoder1(x, fused[0])
        return self.output_head(x, pre_img.shape[2:])


SiameseUNet = DualEncoderEOSARUNet


def build_model(config: dict) -> DualEncoderEOSARUNet:
    """Build the EO-SAR model from config."""
    model_config = config.get("model", {})
    return DualEncoderEOSARUNet(
        eo_channels=model_config.get("eo_channels", 3),
        sar_channels=model_config.get("sar_channels", 1),
        pretrained=model_config.get("pretrained", True),
        num_classes=model_config.get("num_classes", 2),
    )
