from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from modelinhos.containers import Collate, to_preds
from modelinhos.data import SampleDataset
from modelinhos.loss.loss import DetectionLoss
from modelinhos.preprocess.image import build_transform, normalize
from modelinhos.preprocess.lables import DoNothingEncoder, SampleEncoder
from modelinhos.sample import Sample, TrainAnnotation
from modelinhos.tasks.standard import PerBatchEncoded
from modelinhos.trainer.simple import SimpleTrainer, TrainConfig


class DetectionModel(torch.nn.Module):
    """Adapts a (boxes, classes) two-headed model to the pipeline's
    PerBatchEncoded container."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> PerBatchEncoded:
        return to_preds(self.model(x))


class Detector:
    """Thin wrapper around an already-built trainer: converts between
    Sample objects and whatever the trainer's dataset/decode expect.
    Built exclusively by build_detector()/TorchvisionDetector -- do not
    construct directly elsewhere."""

    def __init__(
        self,
        trainer: SimpleTrainer,
        transforms: Callable,
        label_encoder: Optional[SampleEncoder] = None,
    ):
        self._trainer = trainer
        self.transforms = transforms
        self.label_encoder = label_encoder or DoNothingEncoder()

    def fit(
        self, samples: list[Sample], val_samples: Optional[list[Sample]] = None
    ) -> "Detector":
        encoded = self.label_encoder.fit_transform(samples)
        dataset = SampleDataset(encoded, self.transforms)

        val_dataset = None
        if val_samples is not None:
            val_encoded = self.label_encoder.transform(val_samples)
            val_dataset = SampleDataset(val_encoded, self.transforms)

        self._trainer.fit(dataset, val_dataset)
        return self

    def transform(self, samples: list[Sample]) -> list[Sample]:
        encoded = self.label_encoder.transform(samples)
        dataset = SampleDataset(encoded, self.transforms)
        preds = self._trainer.predict(dataset)
        return self.label_encoder.inverse_transform(preds)

    def transform_single(self, frame: np.ndarray) -> list[Sample]:
        blob = self.transforms(frame).unsqueeze(0)
        preds = self._trainer.predict_single(blob)
        return self.label_encoder.inverse_transform(preds)


@dataclass(frozen=True)
class Architecture:
    """What defines a model family: how to build the network, its anchors,
    and the loss builder (the loss owns the encode/decode codec). Presets
    live next to their model definitions in modelinhos/models/."""

    build_model: Callable  # (weights, resolution, n_classes) -> nn.Module
    anchors: Callable  # (resolution) -> priors tensor
    loss: Callable  # (priors, score_thresh) -> DetectionLoss
    normalize: Callable = normalize
    weights: Any = None


def build_detector(
    arch: Architecture,
    lencoder: SampleEncoder,
    resolution: tuple[int, int],
    train: TrainConfig = TrainConfig(),
    th: float = 0.4,
) -> Detector:
    """Assemble a Detector from an architecture, a label encoder and the
    training knobs. The classification head is sized from lencoder.l2i,
    so the encoder must be fit before building. Model and priors are built
    independently (anchors only matter to the loss), and the DetectionLoss
    instance is created exactly once -- it serves both backprop and
    decoding."""
    if not lencoder.l2i:
        raise ValueError(
            "the label encoder has no classes -- fit it before building "
            "the detector (the classification head is sized from l2i)"
        )
    # max index + 1, not len(): duplicate labels (COCO's "N/A" slots)
    # collapse in the dict, but the head must still cover every channel
    # the checkpoint was trained with -- same convention as
    # MetricCollector in evaluation.py.
    n_classes = max(lencoder.l2i.values()) + 1
    model = arch.build_model(
        weights=arch.weights,
        resolution=resolution,
        n_classes=n_classes,
    )
    priors = arch.anchors(resolution)
    loss_fn: DetectionLoss = arch.loss(priors=priors, score_thresh=th)
    trainer = SimpleTrainer(
        model=DetectionModel(model),
        decode=loss_fn.decode,
        collate=Collate(),
        label_encoder=lencoder,
        loss_fn=loss_fn,
        config=train,
    )
    return Detector(
        trainer=trainer,
        transforms=build_transform(arch.weights, arch.normalize),
        label_encoder=lencoder,
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
                    pred["boxes"].cpu().numpy(),
                    pred["scores"].cpu().numpy(),
                    pred["labels"].cpu().numpy(),
                )
                if s > score_thresh
            ],
        )
        for pred in predictions
    ]


class TorchvisionDetector(Detector):
    """Reference detector wrapping a torchvision-native model verbatim, for
    comparison against our own reimplementations. Inference only -- there is
    no DetectionLoss here, since torchvision computes its own loss
    internally and we never train through this path."""

    def __init__(
        self,
        build_model: Callable,
        resolution: tuple[int, int],
        weights: Any,
        lencoder: Optional[SampleEncoder] = None,
        anchors=None,
        th: float = 0.4,
    ):
        trainer = SimpleTrainer(
            model=build_model(weights=weights),
            decode=partial(
                torchvision_to_samples,
                priors=anchors,
                resolution=resolution,
                score_thresh=th,
            ),
            collate=Collate(
                nms=lambda x, pad_value: x,
                to_samples=lambda x: x,
            ),
            label_encoder=lencoder or DoNothingEncoder(),
        )
        super().__init__(
            trainer=trainer,
            transforms=build_transform(weights, lambda x: x),
            label_encoder=lencoder,
        )
