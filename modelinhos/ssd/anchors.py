import math
from itertools import product
from math import ceil, sqrt

import torch


def anchors(
    resolution: tuple[int, int],  # h, w
    sizes: list[list[int]],
    steps: list[int],
    # New, to match RetinaNet, min_sizes * ratios I calculate manually
    aspect_ratios: tuple[float] | None = None,
    clip=False,
    offset=0.5,
):
    """Generates anchros according to the most common conventions.

    The offset parameter is needed to keep compatibility with torchvision.
    Mental model:
    n_anchors =
        └── Feature maps   (len(feature_maps) chunks)
                └── Rows   (fm_h[i])
                └── Cols   (fm_w[i])
                        └── Anchors per cell  (len(aspect_ratios) * len(sizes))

    The flat index:
        mem_loc(fm=i) + row*fm_w[i]*A + col*A + ar_idx*len(sizes) + size_idx
    """
    aspect_ratios = aspect_ratios or (1.0,)
    H, W = resolution
    feature_maps = [[ceil(H / step), ceil(W / step)] for step in steps]
    out = []
    for k, (fm_h, fm_w) in enumerate(feature_maps):
        # Actual stride instead of steps[k] to avoid rounding errors
        stride_x = W // fm_w
        stride_y = H // fm_h

        for i, j in product(range(fm_h), range(fm_w)):
            # This fails for some anchors with rounding errors
            # cx = (j + 0.5) * steps[k] / W ~
            # cy = (i + 0.5) * steps[k] / H ~
            # NB: This actually works in al the cases
            cx = (j + offset) * stride_x / W
            cy = (i + offset) * stride_y / H

            for ar in aspect_ratios:
                w_ratio = 1.0 / math.sqrt(ar)
                h_ratio = math.sqrt(ar)
                for size in sizes[k]:
                    w = round(w_ratio * size / 2) * 2 / W
                    h = round(h_ratio * size / 2) * 2 / H
                    out.append([cx, cy, w, h])

    output = torch.tensor(out).view(-1, 4)

    if clip:
        output.clamp_(0, 1)

    return output


def tvison_anchors(
    resolution,
    base_sizes,
    steps,
    aspect_ratios,
    scales,
):
    sizes = [[int(base * s) for s in scales] for base in base_sizes]
    return anchors(resolution, sizes, steps, aspect_ratios, offset=0)
