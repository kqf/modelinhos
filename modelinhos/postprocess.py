from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from pathlib import Path
from typing import Generic, Protocol, Tuple, TypeVar

import cv2
import torch
import torch.nn.utils.rnn as rnn_utils
import torchvision
from torch import nn

from modelinhos.loss.matching import match
from modelinhos.loss.subloss import Sublosses, WeightedLoss
from modelinhos.preprocess.boxes import decode_boxes
from modelinhos.sample import Sample, TrainAnnotation

C = TypeVar("C")


class HasBoxesAndClasses(Protocol, Generic[C]):
    bbox: C
    label: C
    score: C

    @classmethod
    def is_dataclass(cls) -> bool: ...


def select(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    anchors: torch.Tensor,
    use_negatives: bool,
    positives: torch.Tensor,
    negatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b_pos, a_pos, o_pos = torch.where(positives)
    pred_pos = y_pred[b_pos, a_pos]
    true_pos = y_true[b_pos, o_pos]
    anch_pos = anchors[a_pos]

    if not use_negatives:
        return pred_pos, true_pos, anch_pos

    b_neg, a_neg = torch.where(negatives)
    pred_neg = y_pred[b_neg, a_neg]
    true_neg = torch.zeros_like(pred_neg[:, 0], dtype=torch.long)
    anch_neg = anchors[a_neg]

    pred_all = torch.cat([pred_pos, pred_neg], dim=0)
    true_all = torch.cat([true_pos.view(-1), true_neg], dim=0).long()
    anch_all = torch.cat([anch_pos, anch_neg], dim=0)
    return pred_all, true_all, anch_all


T = TypeVar("T")
LossContainer = TypeVar(
    "LossContainer",
    bound="HasBoxesAndClasses[WeightedLoss]",
)


Matching = Callable[
    [
        HasBoxesAndClasses[torch.Tensor],
        HasBoxesAndClasses[torch.Tensor],
        torch.Tensor,
    ],
    Tuple[torch.Tensor, torch.Tensor],
]


class DetectionLoss(Generic[LossContainer], nn.Module):
    def __init__(
        self,
        priors: torch.Tensor,
        sublosses: LossContainer,
        match: Matching = partial(
            match,
            negpos_ratio=7,
            overalp=0.35,
        ),
    ) -> None:
        super().__init__()
        if not is_dataclass(sublosses):
            raise TypeError("sublosses must be a dataclass instance")
        self.sublosses = sublosses
        self.match = match
        self.register_buffer("priors", priors)

    def forward(
        self,
        y_true: HasBoxesAndClasses[torch.Tensor],
        y_pred: HasBoxesAndClasses[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        positives, negatives = self.match(
            y_pred,
            y_true,
            self.priors,
        )

        losses = {}
        for field in fields(self.sublosses):
            name = field.name
            subloss: WeightedLoss = getattr(self.sublosses, name)
            if subloss.loss is None:
                continue
            y_pred_, y_true_, anchor_ = select(
                getattr(y_pred, name),
                getattr(y_true, name),
                self.priors,
                use_negatives=subloss.needs_negatives,
                positives=positives,
                negatives=negatives,
            )
            losses[name] = subloss(y_pred_, y_true_, anchor_)

        losses["loss"] = torch.stack(tuple(losses.values())).sum()
        return losses


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

    def to(self, device) -> "PerBatch":
        return replace(
            self,
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)},
        )


@dataclass(frozen=True)
class PerBatchEncoded:
    bboxes: torch.Tensor  # (B, K, 4)
    scores: torch.Tensor  # (B, K, 1)
    labels: torch.Tensor  # (B, K, 1)


def anno2tensors(annotations: list[TrainAnnotation]) -> PerImage:
    return PerImage(
        **{
            f.name: (
                torch.tensor([getattr(a, f.name) for a in annotations])
                if annotations
                else torch.empty((0, 1))
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
            bboxes=torch.empty(0),
            scores=torch.empty(0),
            labels=torch.empty(0),
        )

    return PerBatch(
        **{
            f.name: rnn_utils.pad_sequence(
                ensure_correct_shapes([getattr(t, f.name) for t in tensors]),
                batch_first=True,
                padding_value=pad_value,
            )
            for f in fields(tensors[0])
        }
    )

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample[TrainAnnotation]], transform):
        self.samples = samples
        self.transform = transform

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

    def un_batch_nms(self, batch: PerBatch) -> list[Sample]:
        return self.to_samples(self.nms(batch, pad_value=self.pad_value))

def decode(predictions: PerBatchEncoded, loss: DetectionLoss) -> PerBatch:
    update = {}
    for f in fields(predictions):
        subloss = getattr(loss.sublosses, f.name)
        predict = getattr(predictions, f.name)
        update[f.name] = subloss.dec_pred(predict)
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
        batch = anno2tensors(sample.annotations)
        return image, batch


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


# Builds the retina loss
def build_ret_loss(
    priors: torch.Tensor,
    score_thresh: float,
) -> DetectionLoss:
    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        scores, labels = torch.sigmoid(raw_logits).max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels.unsqueeze(-1)

    return DetectionLoss(
        priors=priors,
        sublosses=Sublosses(
            bboxes=WeightedLoss(
                loss=None,
                dec_pred=partial(
                    decode_boxes,
                    priors=priors,
                ),
            ),
            scores=WeightedLoss(
                loss=None,
                dec_pred=lambda x: torch.sigmoid(x)
                .max(dim=-1)[0]
                .unsqueeze(-1),
            ),
            labels=WeightedLoss(
                loss=None,
                dec_pred=decode_labels,
            ),
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
