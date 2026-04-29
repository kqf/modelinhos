import math
from functools import partial

import numpy as np
import pytest
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

from modelinhos.ssd.anchors import (
    anchors,
    anchors2,
    retinanet_anchors,
    retinanet_anchors_,
)


def _default_anchorgen():
    anchor_sizes = tuple(
        (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
        for x in [32, 64, 128, 256, 512]
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [
            (x1 + x2) / 2,  # cx
            (y1 + y2) / 2,  # cy
            (x2 - x1),  # w
            (y2 - y1),  # h
        ],
        dim=-1,
    )


@pytest.fixture
def tv_anchors(resolution):
    ag = _default_anchorgen()
    H, W = resolution

    features = [
        torch.zeros(1, 256, math.ceil(H / s), math.ceil(W / s))
        for s in [8, 16, 32, 64, 128]
    ]

    images = ImageList(
        torch.zeros(1, 3, H, W),
        image_sizes=[(H, W)],
    )

    tv_anchors = ag(images, features)
    tv_anchors = xyxy_to_cxcywh(torch.cat(tv_anchors, dim=1))
    return tv_anchors


@pytest.mark.parametrize(
    "build_anchors",
    [
        partial(
            retinanet_anchors,
            steps=[8, 16, 32, 64, 128],
            aspect_ratios=[0.5, 1.0, 2.0],
            scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
            base_sizes=[32, 64, 128, 256, 512],
        ),
        partial(
            retinanet_anchors_,
            steps=[8, 16, 32, 64, 128],
            aspect_ratios=[0.5, 1.0, 2.0],
            scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
            base_sizes=[32, 64, 128, 256, 512],
        ),
    ],
)
def test_matches_torchvision_anchors(build_anchors, tv_anchors, resolution):
    H, W = resolution
    priors = build_anchors(resolution)
    custom = priors * torch.tensor([W, H, W, H], dtype=priors.dtype)
    print("mean abs diff:", (custom - tv_anchors).abs().mean())
    print("mean abs diff:", (custom - tv_anchors).abs().max())
    print("Original:")
    print(tv_anchors[:10])
    print("Current")
    print(custom[:10])
    np.testing.assert_almost_equal(
        custom.cpu().numpy(),
        tv_anchors.cpu().numpy(),
        decimal=4,
    )


# @pytest.mark.skip()
def test_original_ssd_anchors(resolution):
    priors = anchors(
        image_size=resolution,
        sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
    )

    priors2 = anchors2(
        image_size=resolution,
        sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        aspect_ratios=[1.0],
        clip=False,
    )

    np.testing.assert_almost_equal(
        priors.cpu().numpy(),
        priors2.cpu().numpy(),
        decimal=4,
    )
