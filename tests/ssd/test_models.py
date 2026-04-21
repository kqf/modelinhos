from functools import partial

import pytest
import torch
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)

from modelinhos.ssd.anchors import anchors
from modelinhos.ssd.load import (
    load_with_mismatch_from_weights,
)
from modelinhos.ssd.retinanet import RetinaNetPure


@pytest.fixture
def batch(resolution: tuple[int, int]) -> torch.Tensor:
    return torch.rand(1, 3, *resolution)


@pytest.mark.parametrize(
    "build_model, build_anchors, load_weights",
    [
        (
            RetinaNetPure,
            partial(
                anchors,
                min_sizes=[[16, 32], [64, 128], [256, 512]],
                steps=[8, 16, 32],
                clip=False,
            ),
            partial(
                load_with_mismatch_from_weights,
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
                progress=False,
            ),
        )
    ],
)
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 480),
    ],
)
def test_rentinanet(
    build_model,
    build_anchors,
    load_weights,
    resolution,
    batch,
    n_classes=2,
):
    priors = build_anchors(image_size=resolution[::-1])
    print(priors.shape)
    model = build_model(resolution, n_classes=n_classes)
    model = load_weights(model)
    boxes, classes = model(batch)
    assert boxes.shape == (1, *priors.shape)
    assert classes.shape == (1, priors.shape[0], n_classes)
