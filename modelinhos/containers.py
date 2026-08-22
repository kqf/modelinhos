from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path

import torch
import torch.nn.utils.rnn as rnn_utils
import torchvision

from modelinhos.sample import Sample, TrainAnnotation
from modelinhos.tasks.standard import (
    PerBatch,
    PerBatchEncoded,
    PerImage,
    map_fields,
)

Specs = dict[str, tuple[torch.dtype, int]]


def columns(annotations: list[TrainAnnotation]) -> dict[str, torch.Tensor]:
    return {
        f.name: torch.tensor([getattr(a, f.name) for a in annotations])
        for f in fields(PerImage)
    }


def field_specs(samples: list[Sample[TrainAnnotation]]) -> Specs:
    """Field dtypes and widths, taken from the first annotated sample.

    An annotation-less image carries no dtype or width of its own, and
    a batch of nothing but those has none to lend it -- a background
    image at batch_size 1, or an augmentation that cropped every box
    away, and there is no sample left to ask. So the tie is broken once
    per dataset instead of once per batch, which is the scope where an
    annotated sample really is guaranteed: a detection dataset without a
    single box is the thing worth exploding over, and this explodes at
    construction rather than at some random batch hours into training.
    """
    for sample in samples:
        if sample.annotations:
            return {
                name: (column.dtype, column.shape[1])
                for name, column in columns(sample.annotations).items()
            }
    raise ValueError(
        "no annotated sample in the dataset -- there is nothing to "
        "detect, and nothing to take field dtypes and widths from"
    )


def anno2tensors(
    annotations: list[TrainAnnotation],
    specs: Specs,
) -> PerImage:
    if annotations:
        return PerImage(**columns(annotations))

    return PerImage(
        **{
            name: torch.empty((0, width), dtype=dtype)
            for name, (dtype, width) in specs.items()
        }
    )


def ensure_correct_shapes(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Assert the batch is stackable. anno2tensors gives every sample
    the dataset's dtypes and widths, annotated or not, so disagreement
    here means samples from different datasets met in one batch."""
    specs = {(t.shape[1:], t.dtype) for t in tensors}
    if len(specs) > 1:
        raise ValueError(f"Expected a single shape and dtype, got {specs}")

    return tensors


def collate_labels(
    tensors: list[PerImage],
    pad_value: float = -1.0,
) -> PerBatch:
    if not tensors:
        return PerBatch(
            bboxes=torch.empty(0),
            scores=torch.empty(0),
            labels=torch.empty(0),
        )

    return map_fields(
        lambda *ts: rnn_utils.pad_sequence(
            ensure_correct_shapes(list(ts)),
            batch_first=True,
            padding_value=pad_value,
        ),
        *tensors,
        into=PerBatch,
    )


def un_collate(batched: PerBatch, pad_value: float = -1.0) -> list[PerImage]:
    mask = batched.labels[..., 0] != pad_value
    return [
        map_fields(lambda t, i=i: t[i][mask[i]], batched, into=PerImage)
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


@dataclass(frozen=True)
class Collate:
    pad_value: float = -1.0
    i2b: Callable = collate_labels
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


def to_preds(preds: tuple[torch.Tensor, torch.Tensor]) -> PerBatchEncoded:
    boxes, classes = preds
    return PerBatchEncoded(
        bboxes=boxes,
        scores=classes,
        labels=classes,
    )
