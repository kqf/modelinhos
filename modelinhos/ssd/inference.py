from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch
import tqdm

from modelinhos.postprocess import (
    PerBatch,
    SampleDataset,
    postprocess,
    sample_collate_fn,
    to_preds,
    torchvision_to_samples,
)
from modelinhos.preprocess.image import build_transform, normalize
from modelinhos.preprocess.lables import DoNothingEncoder
from modelinhos.sample import Sample

DataloaderBuilder = Callable[
    [torch.utils.data.Dataset],
    torch.utils.data.DataLoader,
]


def default_dataloader_builder(
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=sample_collate_fn,
    )


@runtime_checkable
class SampleEncoder(Protocol):
    l2i: dict[str, int]
    i2l: dict[int, str]

    def fit_transform(self, samples: list[Sample]) -> list[Sample]: ...

    def transform(self, samples: list[Sample]) -> list[Sample]: ...

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]: ...


@runtime_checkable
class Trainer(Protocol):
    model: torch.nn.Module

    def fit(
        self,
        samples: list[Sample],
    ) -> torch.nn.Module: ...


@dataclass
class DoNothingTrainer:
    model: torch.nn.Module
    anchors: torch.Tensor

    def fit(self, samples: list[Sample]) -> torch.nn.Module:
        return self.model


TrainerFactory = Callable[[torch.nn.Module, torch.Tensor], Trainer]


class Detector:
    def __init__(
        self,
        build_model,
        lencoder: SampleEncoder = None,
        build_trainer: TrainerFactory = DoNothingTrainer,
        train_dataloader: DataloaderBuilder = default_dataloader_builder,
        valid_dataloader: DataloaderBuilder = default_dataloader_builder,
    ):
        self.model, self.transforms, self.postprocess, self.w = build_model()
        self.label_encoder = lencoder or DoNothingEncoder()
        self.trainer = build_trainer(self.model, None)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def fit(self, samples: list[Sample]) -> "Detector":
        self.trainer.fit(self.label_encoder.fit_transform(samples))
        return self

    def transform(self, samples: list[Sample]) -> list[Sample]:
        encoded = self.label_encoder.transform(samples)
        dataset = SampleDataset(
            encoded,
            self.transforms,
        )
        loader = self.valid_dataloader(dataset)
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, _ in tqdm.tqdm(loader):
                results.extend(
                    self.postprocess(
                        predictions=self.model(images),
                    )
                )
        return self.label_encoder.inverse_transform(results)

    def transform_single(self, frame: np.ndarray) -> list[Sample]:
        self.model.eval()
        blob = self.transforms(frame).unsqueeze(0)
        with torch.no_grad():
            return self.postprocess(
                predictions=self.model(blob),
            )


def custom_model(
    build_model,
    resolution,
    weights,
    postprocess=postprocess,
    normalize=normalize,
    th=0.4,
):
    model, anchors = build_model(weights=weights, resolution=resolution)

    class DetectionModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()

            self.model = model

        def forward(self, x: torch.nn.Module) -> PerBatch:
            return to_preds(self.model(x))

    return (
        DetectionModel(model),
        build_transform(weights, normalize),
        partial(
            postprocess,
            resolution=resolution,
            priors=anchors,
            score_thresh=th,
        ),
        weights,
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
        weights,
    )
