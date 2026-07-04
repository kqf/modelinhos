import typing
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path

import cv2
import torch
import torch.nn.utils.rnn as rnn_utils
import torchvision

from modelinhos.preprocess.boxes import decode_boxes
from modelinhos.sample import Sample, TrainAnnotation

_ANNOTATION_HINTS = typing.get_type_hints(TrainAnnotation)


def _field_spec(name: str) -> tuple[int, torch.dtype]:
    args = typing.get_args(_ANNOTATION_HINTS[name])
    dtype = torch.tensor([args[0]()]).dtype
    return len(args), dtype


@dataclass(frozen=True)
class PerImage:
    bboxes: torch.Tensor  # (K, 4)
    scores: torch.Tensor  # (K, 1)
    labels: torch.Tensor  # (K, 1)


@dataclass(frozen=True)
class PerBatch:
    bboxes: torch.Tensor  # (B, K, 4)
    scores: torch.Tensor  # (B, K, 1)
    labels: torch.Tensor  # (B, K, 1)


@dataclass(frozen=True)
class PerBatchEncoded:
    bboxes: torch.Tensor  # (B, K, 4)
    scores: torch.Tensor  # (B, K, 1)
    labels: torch.Tensor  # (B, K, 1)


@dataclass(frozen=True)
class Subloss:
    decode: Callable


@dataclass(frozen=True)
class Loss:
    bboxes: Subloss
    scores: Subloss
    labels: Subloss


def sample_to_image_tensors(annotations: list[TrainAnnotation]) -> PerImage:
    kwargs = {}
    for f in fields(PerImage):
        if values := [getattr(a, f.name) for a in annotations]:
            kwargs[f.name] = torch.tensor(values)
        else:
            width, dtype = _field_spec(f.name)
            kwargs[f.name] = torch.empty((0, width), dtype=dtype)
    return PerImage(**kwargs)


def collate_image_tensors(
    tensors_list: list[PerImage],
    pad_value: float = -1.0,
) -> PerBatch:
    return PerBatch(
        **{
            f.name: rnn_utils.pad_sequence(
                [getattr(t, f.name) for t in tensors_list],
                batch_first=True,
                padding_value=pad_value,
            )
            for f in fields(PerImage)
        }
    )


def un_collate(batched: PerBatch, pad_value: float = -1.0) -> list[PerImage]:
    mask = batched.labels[..., 0] != pad_value
    return [
        PerImage(
            **{
                f.name: getattr(
                    batched,
                    f.name,
                )[i][mask[i]]
                for f in fields(batched)
            },
        )
        for i in range(batched.labels.shape[0])
    ]


def to_sample(unbatched: list[PerImage]) -> list[Sample[TrainAnnotation]]:
    samples = []
    for per_image in unbatched:
        fnames = [f.name for f in fields(per_image)]
        rows = zip(*(getattr(per_image, name) for name in fnames))
        annotations = [
            TrainAnnotation(
                **{n: tuple(v.tolist()) for n, v in zip(fnames, row)},  # type: ignore
            )
            for row in rows
        ]
        samples.append(
            Sample(
                file_name=Path("fake-file.png"),
                annotations=annotations,
            )
        )
    return samples


def nms_unbatch(
    batched: PerBatch,
    iou_thresh: float,
    pad_value: float = -1.0,
) -> list[PerImage]:
    results = []
    for b in un_collate(batched, pad_value=pad_value):
        keep_nms = torchvision.ops.batched_nms(
            b.bboxes,
            b.scores[:, 0],
            b.labels[:, 0],
            iou_thresh,
        )
        update = {f.name: getattr(b, f.name)[keep_nms] for f in fields(b)}
        results.append(replace(b, **update))
    return results


def decode(predictions: PerBatchEncoded, loss: Loss) -> PerBatch:
    update = {}
    for f in fields(predictions):
        subloss = getattr(loss, f.name)
        predict = getattr(predictions, f.name)
        update[f.name] = subloss.decode(predict)
    return PerBatch(**update)


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample[TrainAnnotation]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, PerImage]:
        sample = self.samples[idx]
        bgr = cv2.imread(str(sample.file_name))
        image = self.transform(bgr)
        batch = sample_to_image_tensors(sample.annotations)
        return image, batch


@dataclass(frozen=True)
class Collate:
    pad_value: float = -1.0
    i2b: Callable = collate_image_tensors
    unc: Callable = un_collate
    nms: Callable = partial(nms_unbatch, iou_thresh=0.5)
    to_samples: Callable = to_sample

    def collate(
        self,
        collected: list[tuple[torch.Tensor, PerImage]],
    ) -> tuple[torch.Tensor, PerBatch]:
        images, labels = zip(*collected)
        return torch.stack(images), self.i2b(labels)

    def un_batch(self, batch: PerBatch) -> list[Sample]:
        return self.to_samples(self.unc(batch, pad_value=self.pad_value))

    def un_batch_nms(self, batch: PerBatch) -> list[Sample]:
        return self.to_samples(self.nms(batch, pad_value=self.pad_value))


def postprocess(
    resolution: tuple[int, int],
    priors: torch.Tensor,
    score_thresh: float,
) -> Callable:
    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        scores, labels = torch.sigmoid(raw_logits).max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels.unsqueeze(-1)

    return partial(
        decode,
        loss=Loss(
            bboxes=Subloss(
                decode=partial(
                    decode_boxes,
                    priors=priors,
                    resolution=resolution,
                )
            ),
            scores=Subloss(
                decode=lambda x: torch.sigmoid(x).max(dim=-1)[0].unsqueeze(-1)
            ),
            labels=Subloss(decode=decode_labels),
        ),
    )


def ssd_postprocess(
    resolution: tuple[int, int], priors: torch.Tensor, score_thresh: float
) -> Callable:
    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        probs = torch.softmax(raw_logits, dim=-1)
        probs[..., 0] = 0.0  # exclude background class before taking max
        scores, labels = probs.max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels.unsqueeze(-1)

    def decode_scores(raw_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(raw_logits, dim=-1)
        probs[..., 0] = 0.0
        return probs.max(dim=-1)[0].unsqueeze(-1)

    return partial(
        decode,
        loss=Loss(
            bboxes=Subloss(
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
    )


def to_preds(preds: tuple[torch.Tensor, torch.Tensor]) -> PerBatchEncoded:
    boxes, classes = preds
    return PerBatchEncoded(
        bboxes=boxes,
        scores=classes,
        labels=classes,
    )


def torchvision_to_samples(
    predictions,
    priors,
    resolution,
    score_thresh,
) -> list[Sample[TrainAnnotation]]:
    return [
        Sample(
            file_name=Path("fake-file.png"),
            annotations=[
                TrainAnnotation(
                    bboxes=tuple(b.tolist()),  # type: ignore
                    labels=(ll.item(),),
                    scores=(s.item(),),
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
