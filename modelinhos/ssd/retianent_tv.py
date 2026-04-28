import torch
import torchvision
from torchvision.models.detection.retinanet import (
    LastLevelP6P7,
)

from modelinhos.ssd.anchors import tvison_anchors
from modelinhos.ssd.inference import decode_boxes
from modelinhos.ssd.retinanet import RetinaNetPure


def build_retinanet_torchvision(
    n_classes=91,
    resolution: tuple[int, int] = (800, 1088),
    weights=None,
):
    model = RetinaNetPure(
        n_classes,
        extra_blocks=LastLevelP6P7(2048, 256),
        num_anchors=9,
    )
    priors = tvison_anchors(
        resolution=resolution,
        steps=[8, 16, 32, 64, 128],
        aspect_ratios=[0.5, 1.0, 2.0],
        scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
        base_sizes=[32, 64, 128, 256, 512],
    )
    if weights is not None:
        model.load_state_dict(weights.get_state_dict())
    return model, priors


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
