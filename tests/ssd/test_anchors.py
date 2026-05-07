from functools import partial
from math import ceil

import numpy as np
import pytest
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

from modelinhos.ssd.anchors import anchors, tvison_anchors


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)],
        dim=-1,
    )


def build_tv_anchors(ag, resolution, strides):
    h, w = resolution
    features = [torch.zeros(1, 256, ceil(h / s), ceil(w / s)) for s in strides]
    images = ImageList(torch.zeros(1, 3, h, w), image_sizes=[(h, w)])
    return torch.cat(ag(images, features), dim=1)


def _default_anchorgen():
    sizes = tuple(
        (x, int(x * 2 ** (1 / 3)), int(x * 2 ** (2 / 3)))
        for x in [32, 64, 128, 256, 512]
    )
    return AnchorGenerator(sizes, ((0.5, 1.0, 2.0),) * len(sizes))


def _ssd_anchorgen():
    return AnchorGenerator(
        ((16, 32), (64, 128), (256, 512)),
        ((1,), (1,), (1,)),
    )


@pytest.mark.parametrize(
    "build_tv, build_custom",
    [
        pytest.param(
            partial(
                build_tv_anchors,
                _default_anchorgen(),
                strides=[8, 16, 32, 64, 128],
            ),
            partial(
                tvison_anchors,
                steps=[8, 16, 32, 64, 128],
                aspect_ratios=[0.5, 1.0, 2.0],
                scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
                base_sizes=[32, 64, 128, 256, 512],
            ),
            id="default",
        ),
        pytest.param(
            partial(build_tv_anchors, _ssd_anchorgen(), strides=[8, 16, 32]),
            partial(
                anchors,
                sizes=[[16, 32], [64, 128], [256, 512]],
                steps=[8, 16, 32],
                aspect_ratios=[1.0],
                clip=False,
                offset=0,
            ),
            id="ssd",
        ),
    ],
)
def test_anchors_match_torchvision(resolution, build_tv, build_custom):
    # Convert to cx, cy, w, h (pixels) -- for easier debugging
    expected = xyxy_to_cxcywh(build_tv(resolution))

    # Convert to pixel coordinates, for easier debugging
    h, w = resolution
    actual = build_custom(resolution) * torch.tensor([w, h, w, h])
    np.testing.assert_almost_equal(
        actual.cpu().numpy(), expected.cpu().numpy(), decimal=4
    )
