from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol, runtime_checkable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tqdm

from modelinhos.postprocess import (
    SampleDataset,
    sample_collate_fn,
    torchvision_to_samples,
)
from modelinhos.processing import DoNothingEncoder
from modelinhos.sample import Sample


def normalize(
    image: torch.Tensor,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
):
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(image_std, dtype=dtype, device=device)
    return (image - mean[:, None, None]) / std[:, None, None]


def build_transform(weights, normalize):
    return T.Compose(
        [
            T.Lambda(
                lambda frame: cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )
            ),
            T.Lambda(
                lambda frame: (
                    torch.from_numpy(
                        frame,
                    )
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
            ),
            weights.transforms(),
            T.Lambda(normalize),
        ]
    )


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
        self.model, self.transform, self.postprocess = build_model()
        # ~weights = None
        # ~self.transforms = build_transform(weights, self.normalize)
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
            self.transform,
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
        blob = self.transform(frame).unsqueeze(0)
        with torch.no_grad():
            return self.postprocess(
                predictions=self.model(blob),
            )


def torchvision_model(
    model: torch.nn.Module,
    resolution,
    anchors=None,
    th=0.4,
):
    return (
        model,
        lambda x: x,
        partial(
            torchvision_to_samples,
            priors=anchors,
            resolution=resolution,
            score_thresh=th,
        ),
    )
