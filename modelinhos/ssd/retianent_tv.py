from itertools import product
from math import ceil

import torch
from torchvision.models.detection.retinanet import (
    LastLevelP6P7,
)

from modelinhos.ssd.retinanet import RetinaNetPure


def retinanet_anchors(image_size, steps, min_sizes, aspect_ratios):
    anchors = []

    feature_maps = [
        [ceil(image_size[0] / s), ceil(image_size[1] / s)] for s in steps
    ]

    for k, (fm_h, fm_w) in enumerate(feature_maps):
        for i, j in product(range(fm_h), range(fm_w)):
            cx = (j + 0.5) * steps[k] / image_size[1]
            cy = (i + 0.5) * steps[k] / image_size[0]

            for size in min_sizes[k]:
                area = size * size
                for ar in aspect_ratios:
                    w = (area / ar) ** 0.5
                    h = w * ar

                    anchors.append(
                        [
                            cx,
                            cy,
                            w / image_size[1],
                            h / image_size[0],
                        ]
                    )

    return torch.tensor(anchors).view(-1, 4)


def build_retinanet_torchvision(resolution: tuple[int, int]):
    model = RetinaNetPure(
        91,
        extra_blocks=LastLevelP6P7(2048, 256),
        num_anchors=9,
    )
    priors = retinanet_anchors(
        min_sizes=[
            [32, 40, 50],  # P3
            [64, 80, 100],  # P4
            [128, 160, 200],
            [256, 320, 400],
            [512, 640, 800],
        ],
        steps=[
            8,
            16,
            32,
            64,
            128,
        ],
        image_size=resolution,
        aspect_ratios=[0.5, 1.0, 2.0],
    )

    return model, priors
