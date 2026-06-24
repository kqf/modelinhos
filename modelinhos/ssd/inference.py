from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import tqdm

from modelinhos.postprocess import (
    ImageTensors,
    postprocess,
    sample_collate_fn,
    sample_to_image_tensors,
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
                    torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                )
            ),
            weights.transforms(),
            T.Lambda(normalize),
        ]
    )


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

    # ---> MODIFIED: Now returns image, gt_tensors, and file_name <---
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ImageTensors, Path]:
        sample = self.samples[idx]
        image = cv2.imread(str(sample.file_name))

        image_tensor = self.transform(image)
        gt_tensors = sample_to_image_tensors(sample)

        return image_tensor, gt_tensors, sample.file_name


DataloaderBuilder = Callable[
    [torch.utils.data.Dataset],
    torch.utils.data.DataLoader,
]


# ---> MODIFIED: Attached the new sample_collate_fn <---
def default_dataloader_builder(
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, collate_fn=sample_collate_fn
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
        resolution: tuple[int, int],
        build_model,
        weights,
        th=0.4,
        postprocess=postprocess,
        normalize=normalize,
        lencoder: SampleEncoder = None,
        build_trainer: TrainerFactory = DoNothingTrainer,
        train_dataloader: DataloaderBuilder = default_dataloader_builder,
        valid_dataloader: DataloaderBuilder = default_dataloader_builder,
    ):
        self.transforms = build_transform(weights, normalize)
        self.model, self.priors = self._build(build_model, resolution, weights)
        self.weights = weights
        self.postprocess = postprocess
        self.normalize = normalize
        self.th = th
        self.resolution = resolution
        self.label_encoder = lencoder or DoNothingEncoder()
        self.trainer = build_trainer(self.model, self.priors)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def _build(self, build_model, resolution, weights):
        return build_model(resolution=resolution, weights=weights)

    def fit(self, samples: list[Sample]) -> "Detector":
        self.trainer.fit(self.label_encoder.fit_transform(samples))
        return self

    # ---> MODIFIED: Unpacks the dataloader tuple and passes file_names <---
    def transform(self, samples: list[Sample]) -> list[Sample]:
        dataset = SampleDataset(samples, self.transforms)
        loader = self.valid_dataloader(dataset)
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, _, file_names in tqdm.tqdm(loader):
                results.extend(
                    self.postprocess(
                        predictions=self.model(images),
                        priors=self.priors,
                        resolution=self.resolution,
                        file_names=file_names,
                        score_thresh=self.th,
                    )
                )
        return self.label_encoder.inverse_transform(results)

    def transform_single(self, frame: np.ndarray) -> list[Sample]:
        self.model.eval()
        blob = self.transforms(frame).unsqueeze(0)
        with torch.no_grad():
            return self.postprocess(
                predictions=self.model(blob),
                priors=self.priors,
                resolution=self.resolution,
                file_names=[Path("fake-file.png")],
                score_thresh=self.th,
            )


class TorchvisionDetector(Detector):
    def __init__(
        self,
        resolution: tuple[int, int],
        build_model,
        weights,
        th=0.4,
        postprocess=torchvision_to_samples,
        normalize=lambda x: x,
        lencoder: SampleEncoder = None,
        build_trainer: TrainerFactory = DoNothingTrainer,
        train_dataloader: DataloaderBuilder = default_dataloader_builder,
        valid_dataloader: DataloaderBuilder = default_dataloader_builder,
    ):
        super().__init__(
            resolution,
            build_model,
            weights,
            th,
            postprocess,
            normalize,
            lencoder,
            build_trainer,
            train_dataloader,
            valid_dataloader,
        )

    def _build(self, build_model, resolution, weights):
        return build_model(resolution=resolution, weights=weights), None

    def _run_model(self, batch: torch.Tensor):
        return self.model(list(self.normalize(batch)))
