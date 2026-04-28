from itertools import product
from math import ceil, sqrt

import torch


def anchors(
    min_sizes: list[list[int]],
    steps: list[int],
    clip: bool,
    image_size: tuple[int, int],  # (height, width)
) -> torch.Tensor:
    feature_maps = [
        [ceil(image_size[0] / step), ceil(image_size[1] / step)]
        for step in steps
    ]

    anchors: list[float] = []
    for k, f in enumerate(feature_maps):
        t_min_sizes = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in t_min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = torch.Tensor(anchors).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output


def anchors2(
    sizes: list[list[int]],
    steps: list[int],
    image_size: tuple[int, int],
    aspect_ratios: list[float],
    scales: list[float],
    clip=False,
):
    H, W = image_size
    feature_maps = [[ceil(H / step), ceil(W / step)] for step in steps]
    out = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            cx = (j + 0.5) * steps[k] / W
            cy = (i + 0.5) * steps[k] / H

            for base_size in sizes[k]:
                for ar in aspect_ratios:
                    for scale in scales:
                        s = base_size * scale
                        w = round((s / sqrt(ar)) / 2) * 2 / W
                        h = round((s * sqrt(ar)) / 2) * 2 / H
                        out += [cx, cy, w, h]

    output = torch.tensor(out).view(-1, 4)

    if clip:
        output.clamp_(0, 1)

    return output
