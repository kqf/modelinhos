import math

import numpy as np
import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

from modelinhos.ssd.retinanet import build_retinanet_torchvision


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


def test_retinanet_tv_anchors(resolution=(800, 1066)):
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

    _, priors = build_retinanet_torchvision(resolution)

    modelinhos = priors * torch.tensor([W, H, W, H], dtype=priors.dtype)
    print("mean abs diff:", (modelinhos - tv_anchors).abs().mean())
    print("mean abs diff:", (modelinhos - tv_anchors).abs().max())
    print("Original:")
    print(tv_anchors[:10])
    print("Current")
    print(modelinhos[:10])
    np.testing.assert_almost_equal(
        modelinhos.cpu().numpy(),
        tv_anchors.cpu().numpy(),
        decimal=4,
    )
