from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path
from typing import get_args, get_type_hints

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


def field_specs(annotation: type) -> dict[str, tuple[torch.dtype, int]]:
    """Per-field tensor dtype and width, read off the type hints.

    An image without annotations carries no values to infer either from,
    so the schema stands in for them: TrainAnnotation declares labels as
    ints, boxes and scores as floats, and boxes as a 4-tuple. A batch
    then has the same dtypes and widths whether or not the images in it
    were annotated. Without this an annotation-less batch comes out as
    float32 label ids in (0, 1) boxes, and every consumer downstream
    papers over it -- Long class indices for F.cross_entropy, four
    coordinates to unbind.
    """
    hints = get_type_hints(annotation)
    specs = {}
    for f in fields(annotation):
        args = get_args(hints[f.name])
        elems = {a for a in args if a is not Ellipsis}
        # tuple[float, ...] is variadic: one value per annotation
        width = 1 if Ellipsis in args else len(args)
        dtype = torch.long if elems == {int} else torch.float32
        specs[f.name] = (dtype, width)
    return specs


SPECS = field_specs(TrainAnnotation)


def anno2tensors(annotations: list[TrainAnnotation]) -> PerImage:
    return PerImage(
        **{
            f.name: (
                torch.tensor(
                    [getattr(a, f.name) for a in annotations],
                    dtype=SPECS[f.name][0],
                )
                if annotations
                else torch.empty((0, SPECS[f.name][1]), dtype=SPECS[f.name][0])
            )
            for f in fields(PerImage)
        }
    )


def ensure_correct_shapes(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    specs = {(t.shape[1:], t.dtype) for t in tensors if t.numel() > 0}
    if len(specs) > 1:
        raise ValueError(f"Expected all andnd dtype, got {specs}")

    if not specs:
        return tensors

    (width,), dtype = specs.pop()

    # This is needed to reshape empty
    return [t.reshape(-1, width).to(dtype) for t in tensors]


def collate_labels(
    tensors: list[PerImage],
    pad_value: float = -1.0,
) -> PerBatch:
    if not tensors:
        return PerBatch(
            **{
                f.name: torch.empty(0, dtype=SPECS[f.name][0])
                for f in fields(PerBatch)
            }
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
