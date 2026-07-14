from functools import partial
from typing import Callable, Optional

import numpy as np
import torch

from modelinhos.postprocess import (
    Collate,
    PerBatchEncoded,
    SampleDataset,
    to_preds,
    torchvision_to_samples,
)
from modelinhos.preprocess.image import build_transform, normalize
from modelinhos.preprocess.lables import DoNothingEncoder, SampleEncoder
from modelinhos.sample import Sample
from modelinhos.trainer.simple import build_trainer


class Detector:
    def __init__(
        self,
        build_model: Callable,
        build_trainer: Callable = build_trainer,
        lencoder: SampleEncoder = None,
    ):
        self.model, self.transforms, self.collate = build_model()
        self.label_encoder = lencoder or DoNothingEncoder()
        self._trainer = build_trainer(
            self.model,
            self.collate,
            self.label_encoder,
        )

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


def custom_model(
    build_model,
    resolution,
    weights,
    normalize=normalize,
    th=0.4,
):
    model, anchors = build_model(weights=weights, resolution=resolution)

    class DetectionModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()

            self.model = model

        def forward(self, x: torch.nn.Module) -> PerBatchEncoded:
            return to_preds(self.model(x))

    return (
        DetectionModel(model),
        build_transform(weights, normalize),
        Collate(),
    )


def torchvision_model(
    build_model,
    resolution,
    weights,
    anchors=None,
    th=0.4,
):
    return (
        build_model(weights=weights),
        build_transform(weights, lambda x: x),
        partial(
            torchvision_to_samples,
            priors=anchors,
            resolution=resolution,
            score_thresh=th,
        ),
        Collate(
            nms=lambda x, pad_value: x,
            to_samples=lambda x: x,
        ),
    )
