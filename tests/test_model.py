import torch

from src.models.siamese_unet import DualEncoderEOSARUNet


def test_model_forward_shape() -> None:
    model = DualEncoderEOSARUNet(pretrained=False)
    model.eval()

    pre = torch.randn(2, 3, 64, 64)
    post = torch.randn(2, 1, 64, 64)

    with torch.no_grad():
        output = model(pre, post)

    assert output.shape == (2, 2, 64, 64)
