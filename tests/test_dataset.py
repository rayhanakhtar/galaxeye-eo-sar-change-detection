from pathlib import Path

import tifffile
import torch

from src.data.dataset import ChangeDetectionDataset


def _write_sample_split(root: Path) -> None:
    for subdir in ("pre-event", "post-event", "target"):
        (root / subdir).mkdir(parents=True, exist_ok=True)

    pre = torch.zeros(8, 8, 3, dtype=torch.uint8)
    pre[..., 0] = 64
    pre[..., 1] = 128
    pre[..., 2] = 255
    post = torch.full((8, 8), 32, dtype=torch.uint8)
    target = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0, 1, 2, 3],
        ],
        dtype=torch.uint8,
    )

    tifffile.imwrite(root / "pre-event" / "sample.tif", pre.numpy())
    tifffile.imwrite(root / "post-event" / "sample.tif", post.numpy())
    tifffile.imwrite(root / "target" / "sample.tif", target.numpy())


def test_dataset_shapes_and_remap(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    _write_sample_split(split_dir)

    dataset = ChangeDetectionDataset(str(split_dir), image_size=16)
    sample = dataset[0]

    assert sample["pre_img"].shape == (3, 16, 16)
    assert sample["post_img"].shape == (1, 16, 16)
    assert sample["target"].shape == (1, 16, 16)
    assert sample["target"].dtype == torch.int64
    assert set(torch.unique(sample["target"]).tolist()) == {0, 1}
    assert sample["filename"] == "sample.tif"
