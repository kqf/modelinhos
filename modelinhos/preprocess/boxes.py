import math

import torch


def decode_boxes(
    rel_codes: torch.Tensor,  # (B, N, 4)
    priors: torch.Tensor,  # (N, 4)  normalised cxcywh
    weights: tuple = (1.0, 1.0, 1.0, 1.0),
    bbox_xform_clip: float = math.log(1000.0 / 16),
) -> torch.Tensor:  # (B, N, 4)  normalised xyxy
    pcx, pcy, pw, ph = priors.to(rel_codes).unbind(-1)

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


def encode_boxes(
    boxes: torch.Tensor,  # (..., 4) normalised xyxy ground-truth boxes
    priors: torch.Tensor,  # (..., 4) normalised cxcywh anchors
    weights: tuple = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:  # (..., 4) regression targets, inverse of decode_boxes
    pcx, pcy, pw, ph = priors.to(boxes).unbind(-1)

    gx1, gy1, gx2, gy2 = boxes.unbind(-1)
    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2
    gw = gx2 - gx1
    gh = gy2 - gy1

    dx = weights[0] * (gcx - pcx) / pw
    dy = weights[1] * (gcy - pcy) / ph
    dw = weights[2] * torch.log(gw / pw)
    dh = weights[3] * torch.log(gh / ph)

    return torch.stack([dx, dy, dw, dh], dim=-1)
