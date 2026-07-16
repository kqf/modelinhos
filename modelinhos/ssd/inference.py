from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import torch

from modelinhos.detection import (
    Collate,
    PerBatchEncoded,
    SampleDataset,
    decode,
    to_preds,
    torchvision_to_samples,
)
from modelinhos.preprocess.image import build_transform, normalize
from modelinhos.preprocess.lables import DoNothingEncoder, SampleEncoder
from modelinhos.sample import Sample
from modelinhos.trainer.simple import (
    DLBuilder,
    SimpleTrainer,
    default_dataloader_builder,
    default_optimizer_builder,
)


class DetectionModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> PerBatchEncoded:
        return to_preds(self.model(x))


class Detector:
    """Thin wrapper around an already-built trainer: converts between
    Sample objects and whatever the trainer's dataset/decode expect.
    Built exclusively by DetectionConfig.build()/TorchvisionDetector — do
    not construct directly elsewhere."""

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


@dataclass
class DetectionConfig:
    """Everything needed to build a trainable Detector for one of our own
    (non-torchvision-native) architectures. Anchors only matter for the
    loss (matching/encode/decode) -- build_model builds the model alone.
    loss is built exactly once and used for both decode and backprop --
    see .build()."""

    build_model: Callable  # (weights, resolution) -> model
    anchors: Callable  # (resolution) -> anchors tensor
    resolution: tuple[int, int]
    weights: Any
    lencoder: SampleEncoder
    loss: Callable  # (priors, score_thresh) -> DetectionLoss
    normalize: Callable = normalize
    th: float = 0.4
    epochs: int = 1
    lr: float = 1e-3
    optimizer_builder: Callable = default_optimizer_builder
    metrics: Optional[Callable] = None
    train_dataloader_builder: DLBuilder = default_dataloader_builder
    valid_dataloader_builder: DLBuilder = default_dataloader_builder
    device: Optional[str] = None

    def build(self) -> Detector:
        model = self.build_model(
            weights=self.weights, resolution=self.resolution
        )
        priors = self.anchors(self.resolution)
        loss_fn = self.loss(priors=priors, score_thresh=self.th)
        trainer = SimpleTrainer(
            model=DetectionModel(model),
            decode=partial(decode, loss=loss_fn),
            collate=Collate(),
            label_encoder=self.lencoder,
            loss_fn=loss_fn,
            metrics=self.metrics,
            optimizer_builder=self.optimizer_builder,
            lr=self.lr,
            train_dataloader_builder=self.train_dataloader_builder,
            valid_dataloader_builder=self.valid_dataloader_builder,
            epochs=self.epochs,
            device=self.device,
        )
        return Detector(
            trainer=trainer,
            transforms=build_transform(self.weights, self.normalize),
            label_encoder=self.lencoder,
        )


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
