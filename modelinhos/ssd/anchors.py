import math
from itertools import product
from math import ceil, sqrt

import torch


def anchors(
    image_size: tuple[int, int],  # (height, width)
    sizes: list[list[int]],
    steps: list[int],
    clip: bool,
) -> torch.Tensor:
    H, W = image_size
    feature_maps = [[ceil(H / step), ceil(W / step)] for step in steps]

    anchors: list[float] = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for size in sizes[k]:
                s_kx = size / W
                s_ky = size / H
                cx = (j + 0.5) * steps[k] / W
                cy = (i + 0.5) * steps[k] / H
                anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = torch.Tensor(anchors).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output


def retinanet_anchors(
    image_size,
    steps,
    aspect_ratios,
    scales,
    base_sizes,
):
    anchors = []
    H, W = image_size
    sizes_per_level = [[int(base * s) for s in scales] for base in base_sizes]
    feature_maps = [[math.ceil(H / s), math.ceil(W / s)] for s in steps]
    for k, (fm_h, fm_w) in enumerate(feature_maps):
        stride_x = W // fm_w
        stride_y = H // fm_h

        for i, j in product(range(fm_h), range(fm_w)):
            cx = j * stride_x
            cy = i * stride_y
            for ar in aspect_ratios:
                w_ratio = 1.0 / math.sqrt(ar)
                h_ratio = math.sqrt(ar)
                for size in sizes_per_level[k]:
                    w = round(w_ratio * size / 2) * 2
                    h = round(h_ratio * size / 2) * 2
                    anchors.append([cx / W, cy / H, w / W, h / H])

    return torch.tensor(anchors, dtype=torch.float32).view(-1, 4)


def anchors2(
    image_size: tuple[int, int],
    sizes: list[list[int]],
    steps: list[int],
    # New, to match RetinaNet
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
