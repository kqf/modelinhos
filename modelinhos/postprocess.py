from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path

import cv2
import torch
import torchvision

from modelinhos.preprocess.boxes import decode_boxes
from modelinhos.sample import Annotation, Sample


@dataclass(frozen=True)
class PerBatch:
    boxes: torch.Tensor  # (B, K, 4)
    scores: torch.Tensor  # (B, K, 1)
    labels: torch.Tensor  # (B, K, 1)


@dataclass(frozen=True)
class PerImage:
    boxes: torch.Tensor  # (K, 4)
    scores: torch.Tensor  # (K, 1)
    labels: torch.Tensor  # (K, 1)


@dataclass(frozen=True)
class Subloss:
    decode: Callable


@dataclass(frozen=True)
class Loss:
    boxes: Subloss
    scores: Subloss
    labels: Subloss


def sample_to_image_tensors(sample: Sample) -> PerImage:
    if not sample.annotations:
        return PerImage(
            boxes=torch.empty((0, 4), dtype=torch.float32),
            scores=torch.empty((0,), dtype=torch.float32),
            labels=torch.empty((0,), dtype=torch.long),
        )

    return PerImage(
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


def to_sample(
    unbatched: list[PerImage],
) -> list[Sample]:
    output = []
    for per_image in unbatched:
        annotations = [
            Annotation(
                bbox=tuple(b.tolist()),  # type: ignore
                score=s.item(),
                label=l.item(),
            )
            for b, s, l in zip(  # noqa
                per_image.boxes, per_image.scores, per_image.labels
            )
        ]
        output.append(
            Sample(
                file_name=Path("fake-file.png"),
                annotations=annotations,
            )
        )
    return output


def collate_image_tensors(
    tensors_list: list[PerImage],
    pad_value: float = -1.0,
) -> PerBatch:
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

    return PerBatch(
        boxes=b_boxes,
        scores=b_scores,
        labels=b_labels,
    )


def un_collate(batched: PerBatch, pad_value: float = -1.0) -> list[PerImage]:
    results = []
    for i in range(batched.boxes.shape[0]):
        # Find valid elements by checking labels against the pad_value
        valid_mask = batched.labels[i] != pad_value

        results.append(
            PerImage(
                boxes=batched.boxes[i][valid_mask],
                scores=batched.scores[i][valid_mask],
                labels=batched.labels[i][valid_mask],
            )
        )
    return results


def nms_unbatch(
    batched: PerBatch,
    iou_thresh: float,
    pad_value: float = -1.0,
) -> list[PerImage]:
    results = []
    for b in un_collate(batched, pad_value=pad_value):
        keep_nms = torchvision.ops.batched_nms(
            b.boxes,
            b.scores,
            b.labels,
            iou_thresh,
        )
        update = {f.name: getattr(b, f.name)[keep_nms] for f in fields(b)}
        results.append(replace(b, **update))
    return results


def decode(
    predictions: PerBatch,
    loss: Loss,
) -> PerBatch:
    update = {}
    for f in fields(predictions):
        subloss = getattr(loss, f.name)
        predict = getattr(predictions, f.name)
        update[f.name] = subloss.decode(predict)
    return replace(predictions, **update)


def run_postprocess_pipeline(
    predictions: PerBatch,
    loss: Loss,
    unbatch_fn: Callable,
) -> list[Sample]:
    """Generic pipeline: Decode -> Unbatch -> Map to Sample"""

    # 1. Batched mathematical operations
    batched_data = decode(predictions, loss)

    # 2. Filter and split into a list of variable-length outputs
    unbatched = unbatch_fn(batched_data)

    # 3. Map back to domain objects
    return to_sample(unbatched)


def sample_collate_fn(batch: list[tuple]) -> tuple:
    # 1. Standard stacking for the uniform image tensors
    images = torch.stack([item[0] for item in batch])

    # 2. Pad and batch the variable-length ground truth tensors
    gt_batched = collate_image_tensors([item[1] for item in batch])

    return images, gt_batched


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[Sample],
        transform,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, PerImage]:
        sample = self.samples[idx]
        bgr = cv2.imread(str(sample.file_name))
        image = self.transform(bgr)
        batch = sample_to_image_tensors(sample)
        return image, batch


def postprocess(
    resolution: tuple[int, int],
    priors: torch.Tensor,
    score_thresh: float,
) -> Callable:
    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        scores, labels = torch.sigmoid(raw_logits).max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels

    return partial(
        run_postprocess_pipeline,
        loss=Loss(
            boxes=Subloss(
                decode=partial(
                    decode_boxes,
                    priors=priors,
                    resolution=resolution,
                )
            ),
            scores=Subloss(decode=lambda x: torch.sigmoid(x).max(dim=-1)[0]),
            labels=Subloss(decode=decode_labels),
        ),
        unbatch_fn=partial(
            nms_unbatch,
            iou_thresh=0.5,
        ),
    )


def ssd_postprocess(
    resolution: tuple[int, int],
    priors: torch.Tensor,
    score_thresh: float,
) -> Callable:
    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        probs = torch.softmax(raw_logits, dim=-1)
        probs[..., 0] = 0.0  # exclude background class before taking max
        scores, labels = probs.max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels

    def decode_scores(raw_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(raw_logits, dim=-1)
        probs[..., 0] = 0.0
        return probs.max(dim=-1)[0]

    return partial(
        run_postprocess_pipeline,
        loss=Loss(
            boxes=Subloss(
                decode=partial(
                    decode_boxes,
                    priors=priors,
                    resolution=resolution,
                    weights=(10.0, 10.0, 5.0, 5.0),
                )
            ),
            scores=Subloss(decode=decode_scores),
            labels=Subloss(decode=decode_labels),
        ),
        unbatch_fn=partial(
            nms_unbatch,
            iou_thresh=0.5,
        ),
    )


def to_preds(preds: tuple[torch.Tensor, torch.Tensor]) -> PerBatch:
    boxes, classes = preds
    return PerBatch(
        boxes=boxes,
        scores=classes,
        labels=classes,
    )


def torchvision_to_samples(
    predictions,
    priors,
    resolution,
    score_thresh,
):
    return [
        Sample(
            file_name=Path("fake-file.png"),
            annotations=[
                Annotation(
                    bbox=b.tolist(),
                    label=ll.item(),
                    score=s.item(),
                )
                for b, s, ll in zip(
                    pred["boxes"].numpy(),
                    pred["scores"].numpy(),
                    pred["labels"].numpy(),
                )
                if s > score_thresh
            ],
        )
        for pred in predictions
    ]
