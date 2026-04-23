import math
from itertools import product

import torch
import torchvision
from torchvision.models.detection.retinanet import (
    LastLevelP6P7,
)

from modelinhos.ssd.retinanet import RetinaNetPure


def retinanet_anchors(image_size, steps, aspect_ratios):
    anchors = []
    H, W = image_size
    scales = [1.0, 2 ** (1 / 3), 2 ** (2 / 3)]
    base_sizes = [32, 64, 128, 256, 512]

    # int() truncation matches torchvision's _default_anchorgen
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


def build_retinanet_torchvision(resolution: tuple[int, int]):
    model = RetinaNetPure(
        91,
        extra_blocks=LastLevelP6P7(2048, 256),
        num_anchors=9,
    )
    priors = retinanet_anchors(
        image_size=resolution,
        steps=[8, 16, 32, 64, 128],
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


def normalize(
    image: torch.Tensor,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
):
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(image_std, dtype=dtype, device=device)
    return (image - mean[:, None, None]) / std[:, None, None]
