from functools import partial

import pytest
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)

from modelinhos.ssd.anchors import anchors
from modelinhos.ssd.retinanet import RetinaNetPure, load_with_mismatch


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
                load_with_mismatch,
                pretrained_state_dict=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1.get_state_dict(
                    progress=True,
                ),
            ),
        )
    ],
)
def test_rentinanet(
    build_model,
    build_anchors,
    load_weights,
    resolution=(640, 480),
):
    priors = build_anchors(image_size=resolution[::-1])
    print(priors.shape)
    model = build_model(resolution, n_classes=2)
    model = load_weights(model)
