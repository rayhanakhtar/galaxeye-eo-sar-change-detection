import pytest
import torch

from src.utils.metrics import (
    compute_confusion_matrix,
    logits_to_prediction,
    metrics_from_confusion_matrix,
)


def test_logits_to_prediction_and_metrics() -> None:
    logits = torch.tensor(
        [
            [
                [[4.0, -4.0], [4.0, -4.0]],
                [[-4.0, 4.0], [-4.0, 4.0]],
            ]
        ]
    )
    target = torch.tensor([[[0, 1], [0, 1]]])

    pred = logits_to_prediction(logits)
    conf = compute_confusion_matrix(pred, target)
    metrics = metrics_from_confusion_matrix(conf)

    assert pred.shape == (1, 2, 2)
    assert conf.tolist() == [[2, 0], [0, 2]]
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
