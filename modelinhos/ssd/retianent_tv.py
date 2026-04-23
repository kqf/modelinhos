import math
from itertools import product
from math import ceil

import torch
import torchvision
from torchvision.models.detection.retinanet import (
    LastLevelP6P7,
)

from modelinhos.ssd.retinanet import RetinaNetPure


def retinanet_anchors(image_size, steps, min_sizes, aspect_ratios):
    anchors = []

    h, w = image_size
    feature_maps = [[ceil(h / s), ceil(w / s)] for s in steps]

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


def decode_boxes(
    rel_codes: torch.Tensor,  # (B, N, 4)
    priors: torch.Tensor,  # (N, 4)  normalised cxcywh
    image_size: tuple[int, int],  # (H, W)
    weights: tuple = (1.0, 1.0, 1.0, 1.0),
    bbox_xform_clip: float = math.log(1000.0 / 16),
) -> torch.Tensor:  # (B, N, 4)  pixel xyxy
    H, W = image_size
    pcx, pcy, pw, ph = (
        priors.to(rel_codes)
        * priors.new_tensor(
            [
                W,
                H,
                W,
                H,
            ]
        )
    ).unbind(-1)

    dx, dy = rel_codes[..., 0] / weights[0], rel_codes[..., 1] / weights[1]
    dw = torch.clamp(rel_codes[..., 2] / weights[2], max=bbox_xform_clip)
    dh = torch.clamp(rel_codes[..., 3] / weights[3], max=bbox_xform_clip)

    cx = dx * pw + pcx
    cy = dy * ph + pcy
    w = torch.exp(dw) * pw
    h = torch.exp(dh) * ph

    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        dim=-1,
    )


def postprocess(preds, priors, image_size, score_thresh=0.4, iou_thresh=0.5):
    raw_deltas, raw_logits = preds
    boxes = decode_boxes(raw_deltas, priors.to(raw_deltas.device), image_size)
    scores, labels = torch.sigmoid(raw_logits).max(dim=-1)

    results = []
    for b in range(scores.shape[0]):
        s, l, bx = scores[b], labels[b], boxes[b]  # noqa
        keep = s > score_thresh
        s, l, bx = s[keep], l[keep], bx[keep]  # noqa
        keep = torchvision.ops.batched_nms(bx, s, l, iou_thresh)
        results.append(
            {
                "boxes": bx[keep],
                "scores": s[keep],
                "labels": l[keep],
            }
        )
    return results
