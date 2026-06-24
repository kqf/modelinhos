import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torchvision

from modelinhos.sample import Annotation, Sample


@dataclass(frozen=True)
class BatchedTensors:
    boxes: torch.Tensor  # (B, N, 4)
    scores: torch.Tensor  # (B, N) or (B, N, C)
    labels: torch.Tensor  # (B, N)


@dataclass(frozen=True)
class ImageTensors:
    boxes: torch.Tensor  # (K, 4)
    scores: torch.Tensor  # (K,)
    labels: torch.Tensor  # (K,)


def sample_to_image_tensors(sample: Sample) -> ImageTensors:
    if not sample.annotations:
        return ImageTensors(
            boxes=torch.empty((0, 4), dtype=torch.float32),
            scores=torch.empty((0,), dtype=torch.float32),
            labels=torch.empty((0,), dtype=torch.long),
        )

    return ImageTensors(
        boxes=torch.tensor(
            [ann.bbox for ann in sample.annotations], dtype=torch.float32
        ),
        scores=torch.tensor(
            [ann.score for ann in sample.annotations], dtype=torch.float32
        ),
        labels=torch.tensor(
            [ann.label for ann in sample.annotations], dtype=torch.long
        ),
    )


def image_tensors_to_sample(
    img_tensors: ImageTensors,
    file_name: Path,
) -> Sample:
    annotations = [
        Annotation(
            bbox=tuple(b.tolist()),  # type: ignore
            score=s.item(),
            label=l.item(),
        )
        for b, s, l in zip(  # noqa
            img_tensors.boxes, img_tensors.scores, img_tensors.labels
        )
    ]
    return Sample(file_name=file_name, annotations=annotations)


def collate_image_tensors(
    tensors_list: list[ImageTensors], pad_value: float = -1.0
) -> BatchedTensors:
    max_len = max((len(t.labels) for t in tensors_list), default=0)
    B = len(tensors_list)

    b_boxes = torch.full((B, max_len, 4), pad_value, dtype=torch.float32)
    b_scores = torch.full((B, max_len), pad_value, dtype=torch.float32)
    b_labels = torch.full((B, max_len), int(pad_value), dtype=torch.long)

    for i, t in enumerate(tensors_list):
        n = len(t.labels)
        if n > 0:
            b_boxes[i, :n] = t.boxes
            b_scores[i, :n] = t.scores
            b_labels[i, :n] = t.labels

    return BatchedTensors(boxes=b_boxes, scores=b_scores, labels=b_labels)


def uncollate_batched_tensors(
    batched: BatchedTensors, pad_value: float = -1.0
) -> list[ImageTensors]:
    results = []
    for i in range(batched.boxes.shape[0]):
        # Find valid elements by checking labels against the pad_value
        valid_mask = batched.labels[i] != pad_value

        results.append(
            ImageTensors(
                boxes=batched.boxes[i][valid_mask],
                scores=batched.scores[i][valid_mask],
                labels=batched.labels[i][valid_mask],
            )
        )
    return results


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


def decode_standard(
    preds: tuple,
    priors: torch.Tensor,
    resolution: tuple[int, int],
) -> BatchedTensors:
    raw_deltas, raw_logits = preds
    boxes = decode_boxes(raw_deltas, priors.to(raw_deltas.device), resolution)
    scores, labels = torch.sigmoid(raw_logits).max(dim=-1)
    return BatchedTensors(boxes=boxes, scores=scores, labels=labels)


def decode_ssd(
    preds: tuple,
    priors: torch.Tensor,
    resolution: tuple[int, int],
) -> BatchedTensors:
    raw_deltas, raw_logits = preds
    boxes = decode_boxes(
        raw_deltas,
        priors.to(raw_deltas.device),
        resolution,
        weights=(10.0, 10.0, 5.0, 5.0),
    )
    scores_all = torch.softmax(raw_logits, dim=-1)
    # Return empty labels, as SSD derives labels during multi-class NMS
    return BatchedTensors(
        boxes=boxes, scores=scores_all, labels=torch.empty(0)
    )


def nms_unbatch_standard(
    batched: BatchedTensors, score_thresh: float, iou_thresh: float
) -> list[ImageTensors]:
    results = []
    for i in range(batched.boxes.shape[0]):
        b, s, ll = batched.boxes[i], batched.scores[i], batched.labels[i]

        keep = s > score_thresh
        b, s, ll = b[keep], s[keep], ll[keep]  # noqa

        keep_nms = torchvision.ops.batched_nms(b, s, ll, iou_thresh)
        results.append(
            ImageTensors(
                boxes=b[keep_nms],
                scores=s[keep_nms],
                labels=ll[keep_nms],
            )
        )
    return results


def nms_unbatch_ssd(
    batched: BatchedTensors, score_thresh: float, iou_thresh: float
) -> list[ImageTensors]:
    results = []
    for i in range(batched.boxes.shape[0]):
        boxes_b = batched.boxes[i]
        scores_b = batched.scores[i]  # (N, C)

        img_boxes, img_scores, img_labels = [], [], []

        for cls in range(1, scores_b.shape[1]):  # Skip background
            cls_scores = scores_b[:, cls]
            keep = cls_scores > score_thresh
            if keep.any():
                img_boxes.append(boxes_b[keep])
                img_scores.append(cls_scores[keep])
                img_labels.append(
                    torch.full_like(cls_scores[keep], cls, dtype=torch.int64)
                )

        if not img_boxes:
            results.append(
                ImageTensors(torch.empty(0, 4), torch.empty(0), torch.empty(0))
            )
            continue

        all_b = torch.cat(img_boxes, dim=0)
        all_s = torch.cat(img_scores, dim=0)
        all_l = torch.cat(img_labels, dim=0)

        keep_nms = torchvision.ops.batched_nms(all_b, all_s, all_l, iou_thresh)
        results.append(
            ImageTensors(
                boxes=all_b[keep_nms],
                scores=all_s[keep_nms],
                labels=all_l[keep_nms],
            )
        )
    return results


def run_postprocess_pipeline(
    preds: tuple,
    priors: torch.Tensor,
    resolution: tuple[int, int],
    file_names: list[Path],
    decode_fn: Callable,
    unbatch_fn: Callable,
) -> list[Sample]:
    """Generic pipeline: Decode -> Unbatch -> Map to Sample"""

    # 1. Batched mathematical operations
    batched_data = decode_fn(preds, priors, resolution)

    # 2. Filter and split into a list of variable-length outputs
    unbatched_data = unbatch_fn(batched_data)

    # 3. Map back to domain objects
    return [
        image_tensors_to_sample(img_tensors, fname)
        for img_tensors, fname in zip(unbatched_data, file_names)
    ]


postprocess = partial(
    run_postprocess_pipeline,
    decode_fn=decode_standard,
    unbatch_fn=partial(
        nms_unbatch_standard,
        score_thresh=0.4,
        iou_thresh=0.5,
    ),
)

ssd_postprocess = partial(
    run_postprocess_pipeline,
    decode_fn=decode_ssd,
    unbatch_fn=partial(
        nms_unbatch_ssd,
        score_thresh=0.004,
        iou_thresh=0.5,
    ),
)
