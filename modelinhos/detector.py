from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from modelinhos.augment import Augmentation, identity, no_augment
from modelinhos.containers import Collate, to_preds
from modelinhos.data import SampleDataset
from modelinhos.engine import Engine
from modelinhos.loss.loss import DetectionLoss
from modelinhos.preprocess.image import rgb_normalized_image_encoder
from modelinhos.preprocess.lables import DoNothingEncoder, SampleEncoder
from modelinhos.sample import Annotation, Sample
from modelinhos.tasks.standard import PerBatchEncoded


class DetectionModel(torch.nn.Module):
    """Adapts a (boxes, classes) two-headed model to the pipeline's
    PerBatchEncoded container."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> PerBatchEncoded:
        return to_preds(self.model(x))


class Detector:
    """The stable public API: samples in, samples out, regardless of
    which model family or training engine sits behind it. Converts
    between Sample objects (pixel space, string labels) and whatever
    the engine's dataset/decode expect. Compose via build_detector()."""

    def __init__(
        self,
        engine: Engine,
        image_encoder: Callable,
        label_encoder: Optional[SampleEncoder] = None,
        augment: Augmentation = identity,
    ):
        self._engine = engine
        self.image_encoder = image_encoder
        self.label_encoder = label_encoder or DoNothingEncoder()
        self.augment = augment

    def fit(
        self, samples: list[Sample], val_samples: Optional[list[Sample]] = None
    ) -> "Detector":
        encoded = self.label_encoder.fit_transform(samples)
        # Augmentation is a train-only concern: the val dataset below and
        # transform() build their SampleDatasets without it.
        dataset = SampleDataset(
            encoded,
            self.image_encoder,
            augment=self.augment,
        )

        val_dataset = None
        if val_samples is not None:
            val_encoded = self.label_encoder.transform(val_samples)
            val_dataset = SampleDataset(val_encoded, self.image_encoder)

        self._engine.fit(dataset, val_dataset)
        return self

    def transform(self, samples: list[Sample]) -> list[Sample]:
        encoded = self.label_encoder.transform(samples)
        dataset = SampleDataset(encoded, self.image_encoder)
        preds = self._engine.predict(dataset)
        return self.label_encoder.inverse_transform(preds)

    def transform_single(self, frame: np.ndarray) -> list[Sample]:
        blob = self.image_encoder(frame).unsqueeze(0)
        preds = self._engine.predict_single(blob)
        return self.label_encoder.inverse_transform(preds)


@dataclass(frozen=True)
class Baked:
    """What a recipe produces once the label space and geometry are
    known: the framework-agnostic, tensor-level pieces every training
    engine consumes. The loss owns the codec -- decoding raw model
    output is loss.decode."""

    model: torch.nn.Module
    loss: DetectionLoss
    collate: Collate
    iencoder: Callable
    augment: Augmentation


@dataclass(frozen=True)
class DetectionRecipe:
    """The internally consistent definition of how detection works for one
    model family: the anchors must be the ones the loss encodes against,
    which must match the head layout build_model produces, which must see
    pixels the way iencoder prepares them. Everything here is fixed before
    any data is seen; the label space (n_classes) and geometry
    (resolution) arrive at bake() time. Presets live next to their model
    definitions in modelinhos/models/."""

    build_model: Callable  # (weights, resolution, n_classes) -> nn.Module
    anchors: Callable  # (resolution) -> priors tensor
    # (priors, score_thresh) -> DetectionLoss: the returned loss carries
    # the bound matcher, which analysis (modelinhos.infos) reuses -- the
    # recipe is the single owner of the matching configuration.
    loss: Callable[..., DetectionLoss]
    # (resolution) -> Augmentation, applied to the training dataset only
    # (never validation or inference). Defaults to none.
    augment: Callable = no_augment
    # BGR uint8 HWC ndarray -> normalized float CHW tensor. Must be a
    # built encoder, not the rgb_normalized_image_encoder factory itself.
    iencoder: Callable = field(default_factory=rgb_normalized_image_encoder)
    # (weights, frames, th) -> list[Sample]: the torchvision-native
    # upstream this recipe mirrors (see torchvision_reference), used as
    # the ground truth in parity tests. None when there is no upstream.
    reference: Optional[Callable] = None

    def bake(
        self,
        n_classes: int,
        resolution: tuple[int, int],
        weights: Any = None,
        th: float = 0.4,
    ) -> Baked:
        """Tie the pieces together for one label space and geometry.
        Model and priors are built independently (anchors only matter to
        the loss), and the DetectionLoss instance is created exactly
        once -- it serves both backprop and decoding."""
        model = self.build_model(
            weights=weights,
            resolution=resolution,
            n_classes=n_classes,
        )
        priors = self.anchors(resolution)
        loss: DetectionLoss = self.loss(priors=priors, score_thresh=th)
        return Baked(
            model=DetectionModel(model),
            loss=loss,
            collate=Collate(),
            iencoder=self.iencoder,
            augment=self.augment(resolution),
        )


EngineBuilder = Callable[[Baked], Engine]


def build_detector(
    arch: DetectionRecipe,
    lencoder: SampleEncoder,
    resolution: tuple[int, int],
    engine: EngineBuilder,
    th: float = 0.4,
    weights: Any = None,
) -> Detector:
    """Compose the three independent choices -- architecture (recipe),
    label space (lencoder) and training backend (engine) -- into a
    Detector. The classification head is sized from lencoder.n_classes,
    so the encoder must be fit before building."""
    baked = arch.bake(
        n_classes=lencoder.n_classes,
        resolution=resolution,
        weights=weights,
        th=th,
    )
    return Detector(
        engine=engine(baked),
        image_encoder=baked.iencoder,
        label_encoder=lencoder,
        augment=baked.augment,
    )


def torchvision_reference(build_model: Callable) -> Callable:
    """Wrap a torchvision-native detection model constructor into a plain
    predict function -- (weights, frames, th) -> list[Sample]. The
    reference path needs none of the Detector machinery: torchvision
    models resize and normalize internally (GeneralizedRCNNTransform)
    and name their own classes via weights.meta. Their pixel-space boxes
    are normalized by each frame's size on the way out, so reference
    predictions are relative like everything else. Used as
    DetectionRecipe.reference for parity tests."""
    encode = rgb_normalized_image_encoder(lambda x: x)

    def predict(
        weights,
        frames: list[np.ndarray],
        th: float = 0.4,
        batch_size: int = 8,
    ) -> list[Sample[Annotation]]:
        model = build_model(weights=weights)
        # eval() is load-bearing: torchvision models in train mode demand
        # targets inside forward() and would raise without them.
        model.eval()
        categories = weights.meta["categories"]
        results: list[Sample[Annotation]] = []
        with torch.no_grad():
            for start in range(0, len(frames), batch_size):
                batch = frames[start : start + batch_size]
                predictions = model([encode(frame) for frame in batch])
                results.extend(
                    Sample(
                        file_name=Path("fake-file.png"),
                        annotations=[
                            Annotation(
                                bbox=tuple(
                                    (
                                        box
                                        / np.array(
                                            [
                                                frame.shape[1],
                                                frame.shape[0],
                                            ]
                                            * 2
                                        )
                                    ).tolist()
                                ),  # type: ignore
                                label=categories[int(label)],
                                score=float(score),
                            )
                            for box, score, label in zip(
                                pred["boxes"].cpu().numpy(),
                                pred["scores"].cpu().numpy(),
                                pred["labels"].cpu().numpy(),
                            )
                            if score > th
                        ],
                    )
                    for frame, pred in zip(batch, predictions)
                )
        return results

    return predict
