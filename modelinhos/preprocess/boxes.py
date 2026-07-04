import math

import torch


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
