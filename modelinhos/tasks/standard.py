from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Generic, TypeVar

import torch
from typing_extensions import Self

T = TypeVar("T")
U = TypeVar("U")
D = TypeVar("D", bound="StandardDetection")


@dataclass(frozen=True)
class StandardDetection(Generic[T]):
    """One record of the standard detection task: something that has
    boxes, scores and labels. T is the per-field payload -- torch.Tensor
    for the pipeline states below, WeightedLoss for Sublosses, and so on.

    The pipeline states (PerImage/PerBatch/PerBatchEncoded) are kept as
    distinct empty subclasses rather than aliases on purpose: they are
    different stages of the data flow and must not be interchangeable for
    a type checker, even though all the field-reflection machinery is
    written once against this base.
    """

    bboxes: T
    scores: T
    labels: T

    def to(self, device) -> Self:
        # Only meaningful when T supports .to() (i.e. tensors)
        return replace(
            self,
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)},
        )


@dataclass(frozen=True)
class PerImage(StandardDetection[torch.Tensor]):
    """(K, .) detections of a single image."""


@dataclass(frozen=True)
class PerBatch(StandardDetection[torch.Tensor]):
    """(B, K, .) decoded, collated batch, padded with pad_value."""


@dataclass(frozen=True)
class PerBatchEncoded(StandardDetection[torch.Tensor]):
    """(B, A, .) raw per-anchor model outputs, before decoding."""


def map_fields(
    fn: Callable[..., U],
    *objs: StandardDetection,
    into: type[D],
) -> D:
    """Apply fn field-wise across parallel StandardDetection objects and
    collect the results into the target state. All the stringly-typed
    reflection lives here; call sites stay concretely typed through their
    signatures."""
    return into(
        **{
            f.name: fn(*(getattr(o, f.name) for o in objs))
            for f in fields(into)
        }
    )
