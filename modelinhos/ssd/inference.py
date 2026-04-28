import math
from functools import wraps

import cv2
import numpy as np
import torch
import torchvision


def to_blob(frame: np.ndarray, weights) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    preprocess = weights.transforms()
    return preprocess(tensor).unsqueeze(0)


def normalize(
    image: torch.Tensor,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
):
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(image_std, dtype=dtype, device=device)
    return (image - mean[:, None, None]) / std[:, None, None]


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


def postprocess(preds, priors, resolution, score_thresh=0.4, iou_thresh=0.5):
    raw_deltas, raw_logits = preds
    boxes = decode_boxes(raw_deltas, priors.to(raw_deltas.device), resolution)
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


def build_inference_model(
    build_model,
    weights,
    th=0.4,
    postprocess=postprocess,
    normalize=normalize,
):
    @wraps(build_inference_model)
    def build_inference(resolution):
        model, priors = build_model(resolution=resolution, weights=weights)
        model.eval()

        def infer(frame: np.ndarray):
            blob = to_blob(frame, weights)
            return postprocess(
                model(normalize(blob)),
                priors,
                resolution=resolution,
                score_thresh=th,
            )

        return infer

    return build_inference
