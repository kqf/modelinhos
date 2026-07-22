from functools import partial

import pytest
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)

from modelinhos.models.retinanet import (
    build_torchvision_retinanet,
    bulid_retinanet,
    retina_anchors,
    torchvision_retina_anchors,
)
from modelinhos.models.ssdlite import (
    build_torchvision_ssdlite,
    torchvision_ssdlite_anchors,
)


@pytest.fixture
def batch(resolution: tuple[int, int]) -> torch.Tensor:
    return torch.rand(1, 3, *resolution)


@pytest.mark.skip
@pytest.mark.parametrize(
    "build_model, build_anchors",
    [
        (
            partial(
                bulid_retinanet,
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            ),
            retina_anchors,
        ),
        (
            partial(
                build_torchvision_retinanet,
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            ),
            torchvision_retina_anchors,
        ),
        (
            partial(
                build_torchvision_ssdlite,
                weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            ),
            torchvision_ssdlite_anchors,
        ),
    ],
)
def test_ssd(
    build_model,
    build_anchors,
    resolution,
    batch,
    n_classes=91,
):
    model = build_model(n_classes=n_classes, resolution=resolution)
    priors = build_anchors(resolution)
    boxes, classes = model(batch)
    assert boxes.shape == (1, *priors.shape)
    assert classes.shape == (1, priors.shape[0], n_classes)
