"""Training engines: interchangeable owners of the optimization loop.

An engine speaks datasets and tensors -- it never sees Samples in pixel
space; that boundary belongs to Detector. One adapter module per
framework (simple / skorch / lit) so the optional dependencies stay
optional: importing modelinhos.engine.skorch requires skorch, importing
this package does not.
"""

from typing import Protocol, runtime_checkable

import torch

from modelinhos.sample import Sample, TrainAnnotation


@runtime_checkable
class Engine(Protocol):
    """What Detector needs from a training backend. Predictions come
    back decoded, NMS'd and un-batched, still in the model's normalized
    space -- Detector's label encoder does the inverse transform."""

    def fit(self, dataset, val_dataset=None) -> "Engine": ...

    def predict(self, dataset) -> list[Sample[TrainAnnotation]]: ...

    def predict_single(
        self, blob: torch.Tensor
    ) -> list[Sample[TrainAnnotation]]: ...
