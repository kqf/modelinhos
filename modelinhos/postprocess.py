from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import torch
import torchvision

from modelinhos.preprocess.boxes import decode_boxes
from modelinhos.sample import Annotation, Sample


@dataclass(frozen=True)
class PerBatch:
    boxes: torch.Tensor  # (B, N, 4)
    scores: torch.Tensor  # (B, N) or (B, N, C)
    labels: torch.Tensor  # (B, N)
    file_names: list[Path]


@dataclass(frozen=True)
class PerImage:
    boxes: torch.Tensor  # (K, 4)
    scores: torch.Tensor  # (K,)
    labels: torch.Tensor  # (K,)
    file_name: Path


def sample_to_image_tensors(sample: Sample) -> PerImage:
    if not sample.annotations:
        return PerImage(
            boxes=torch.empty((0, 4), dtype=torch.float32),
            scores=torch.empty((0,), dtype=torch.float32),
            labels=torch.empty((0,), dtype=torch.long),
            file_name=sample.file_name,
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
        file_name=sample.file_name,
    )


def image_tensors_to_sample(
    img_tensors: PerImage,
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
    return Sample(file_name=img_tensors.file_name, annotations=annotations)


def collate_image_tensors(
    tensors_list: list[PerImage], pad_value: float = -1.0
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

    b_file_names = [t.file_name for t in tensors_list]

    return PerBatch(
        boxes=b_boxes,
        scores=b_scores,
        labels=b_labels,
        file_names=b_file_names,
    )


def uncollate_batched_tensors(
    batched: PerBatch, pad_value: float = -1.0
) -> list[PerImage]:
    results = []
    for i in range(batched.boxes.shape[0]):
        # Find valid elements by checking labels against the pad_value
        valid_mask = batched.labels[i] != pad_value

        results.append(
            PerImage(
                boxes=batched.boxes[i][valid_mask],
                scores=batched.scores[i][valid_mask],
                labels=batched.labels[i][valid_mask],
                file_name=batched.file_names[i],
            )
        )
    return results


def decode_standard(
    predictions: PerBatch,
    priors: torch.Tensor,
    resolution: tuple[int, int],
) -> PerBatch:
    raw_deltas, raw_logits = predictions.boxes, predictions.scores
    boxes = decode_boxes(raw_deltas, priors.to(raw_deltas.device), resolution)
    scores, labels = torch.sigmoid(raw_logits).max(dim=-1)
    return PerBatch(
        boxes=boxes,
        scores=scores,
        labels=labels,
        file_names=predictions.file_names,
    )


def decode_ssd(
    predictions: PerBatch,
    priors: torch.Tensor,
    resolution: tuple[int, int],
) -> PerBatch:
    raw_deltas, raw_logits = predictions.boxes, predictions.scores
    boxes = decode_boxes(
        raw_deltas,
        priors.to(raw_deltas.device),
        resolution,
        weights=(10.0, 10.0, 5.0, 5.0),
    )
    scores_all = torch.softmax(raw_logits, dim=-1)  # (B, N, C)

    # Exclude background (class 0) by zeroing it out before taking the max.
    # This lets us share nms_unbatch_standard with the standard pipeline.
    scores_all[..., 0] = 0.0
    scores, labels = scores_all.max(dim=-1)  # both (B, N)

    return PerBatch(
        boxes=boxes,
        scores=scores,
        labels=labels,
        file_names=predictions.file_names,
    )


def nms_unbatch(
    batched: PerBatch,
    score_thresh: float,
    iou_thresh: float,
) -> list[PerImage]:
    results = []
    for i in range(batched.boxes.shape[0]):
        b, s, ll = batched.boxes[i], batched.scores[i], batched.labels[i]

        keep = s > score_thresh
        b, s, ll = b[keep], s[keep], ll[keep]  # noqa

        keep_nms = torchvision.ops.batched_nms(b, s, ll, iou_thresh)
        results.append(
            PerImage(
                boxes=b[keep_nms],
                scores=s[keep_nms],
                labels=ll[keep_nms],
                file_name=batched.file_names[i],
            )
        )
    return results


def run_postprocess_pipeline(
    predictions: PerBatch,
    priors: torch.Tensor,
    resolution: tuple[int, int],
    score_thresh: float,
    decode_fn: Callable,
    unbatch_fn: Callable,
) -> list[Sample]:
    """Generic pipeline: Decode -> Unbatch -> Map to Sample"""

    # 1. Batched mathematical operations
    batched_data = decode_fn(predictions, priors, resolution)

    # 2. Filter and split into a list of variable-length outputs
    unbatched = unbatch_fn(batched_data, score_thresh=score_thresh)

    # 3. Map back to domain objects
    return [image_tensors_to_sample(img_tensors) for img_tensors in unbatched]


postprocess = partial(
    run_postprocess_pipeline,
    decode_fn=decode_standard,
    unbatch_fn=partial(
        nms_unbatch,
        iou_thresh=0.5,
    ),
)

ssd_postprocess = partial(
    run_postprocess_pipeline,
    decode_fn=decode_ssd,
    unbatch_fn=partial(
        nms_unbatch,
        iou_thresh=0.5,
    ),
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


def to_preds(preds: tuple[torch.Tensor, torch.Tensor]) -> PerBatch:
    boxes, classes = preds
    return PerBatch(
        boxes=boxes,
        scores=classes,
        labels=torch.empty_like(classes),
        file_names=[Path("fake-file.png") for _ in range(len(boxes))],
    )
