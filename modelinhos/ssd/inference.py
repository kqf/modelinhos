from functools import partial
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import torch
import tqdm

from modelinhos.postprocess import (
    Collate,
    PerBatchEncoded,
    SampleDataset,
    postprocess,
    to_preds,
    torchvision_to_samples,
)
from modelinhos.preprocess.image import build_transform, normalize
from modelinhos.preprocess.lables import DoNothingEncoder
from modelinhos.sample import Sample

DataloaderBuilder = Callable[
    [torch.utils.data.Dataset, Callable],
    torch.utils.data.DataLoader,
]


def default_dataloader_builder(
    dataset,
    collate_fn,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
    )


@runtime_checkable
class SampleEncoder(Protocol):
    l2i: dict[str, int]
    i2l: dict[int, str]

    def fit_transform(self, samples: list[Sample]) -> list[Sample]: ...

    def transform(self, samples: list[Sample]) -> list[Sample]: ...

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]: ...


class Detector:
    def __init__(
        self,
        build_model=lambda: print,
        build_metrics=None,
        lencoder: SampleEncoder = None,
        train_dataloader: DataloaderBuilder = default_dataloader_builder,
        valid_dataloader: DataloaderBuilder = default_dataloader_builder,
    ):
        self.model, self.transforms, self.decode, self.w, collate = (
            build_model()
        )
        self.label_encoder = lencoder or DoNothingEncoder()
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss = print
        self.metrics = build_metrics
        self.collate = collate

    def fit(self, samples: list[Sample]) -> "Detector":
        encoded = self.label_encoder.transform(samples)
        dataset = SampleDataset(encoded, self.transforms)
        loader = self.train_dataloader(dataset, self.collate.collate)
        self.model.train()
        for images, batch in tqdm.tqdm(loader):
            preds = self.model(images)
            losses = self.loss(batch, preds)
            print(losses)
            true = self.label_encoder.inverse_transform(
                self.collate.un_batch(batch),
            )
            pred = self.label_encoder.inverse_transform(
                self.collate.un_batch_nms(self.decode(preds))
            )
            self.metrics(true, pred)
        return self

    def transform(self, samples: list[Sample]) -> list[Sample]:
        encoded = self.label_encoder.transform(samples)
        dataset = SampleDataset(encoded, self.transforms)
        loader = self.valid_dataloader(dataset, self.collate.collate)
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, _ in tqdm.tqdm(loader):
                results.extend(
                    self.collate.un_batch_nms(self.decode(self.model(images)))
                )
        return self.label_encoder.inverse_transform(results)

    def transform_single(self, frame: np.ndarray) -> list[Sample]:
        self.model.eval()
        blob = self.transforms(frame).unsqueeze(0)
        with torch.no_grad():
            preds = self.collate.un_batch_nms(self.decode(self.model(blob)))
        return self.label_encoder.inverse_transform(preds)


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

        def forward(self, x: torch.nn.Module) -> PerBatchEncoded:
            return to_preds(self.model(x))

    return (
        DetectionModel(model),
        build_transform(weights, normalize),
        postprocess(
            resolution=resolution,
            priors=anchors,
            score_thresh=th,
        ),
        weights,
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
        weights,
        Collate(
            nms=lambda x, pad_value: x,
            to_samples=lambda x: x,
        ),
    )
