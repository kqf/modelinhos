import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from modelinhos.sample import Annotation, Sample, TrainAnnotation

# module logger
logger = logging.getLogger(__name__)


@runtime_checkable
class SampleEncoder(Protocol):
    l2i: dict[str, int]
    i2l: dict[int, str]

    @property
    def n_classes(self) -> int: ...

    def fit_transform(self, samples: list[Sample]) -> list[Sample]: ...

    def transform(self, samples: list[Sample]) -> list[Sample]: ...

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]: ...


@dataclass
class LabelEncoder:
    # Purely label <-> index: bboxes pass through untouched, since they
    # are relative ([0, 1]) everywhere in the library and stay so.
    # leave both empty to learn the classes from data via fit(); a
    # provided mapping must keep index 0 for BACKGROUND (__post_init__).
    l2i: dict[str, int] = field(default_factory=dict)
    i2l: dict[int, str] = field(default_factory=dict)
    l2i_background: dict[str, int] = field(
        default_factory=lambda: {"__background__": 0},
    )

    def __post_init__(self):
        if self.l2i and not self.i2l:
            self.i2l = {v: k for k, v in self.l2i.items()}
        elif self.i2l and not self.l2i:
            self.l2i = {v: k for k, v in self.i2l.items()}
        if not self.l2i:
            return  # classes will be learned by fit()
        background = next(iter(self.l2i_background))
        zero = self.i2l.get(0)
        if zero is not None and zero != background:
            raise ValueError(
                f"index 0 is reserved for background, got {zero!r}: "
                "every loss/decode treats channel 0 as background, so "
                f"{zero!r} could never be predicted"
            )
        if zero is None:
            logger.warning(
                "the provided l2i has no zero class -- added %r for you, "
                "since every loss/decode reserves index 0 for background",
                self.l2i_background,
            )
            self.l2i.update(self.l2i_background)
            self.i2l[0] = background

    @property
    def n_classes(self) -> int:
        """Classification head size: max index + 1, not len() --
        duplicate labels (COCO's "N/A" slots) collapse in the dict, but
        the head must still cover every channel the checkpoint was
        trained with. Same convention as MetricCollector in
        evaluation.py."""
        if not self.l2i:
            raise ValueError(
                "the label encoder has no classes -- fit it (or provide "
                "l2i) before asking for n_classes; the classification "
                "head is sized from it"
            )
        return max(self.l2i.values()) + 1

    def fit(self, samples: list[Sample[Annotation]]) -> "LabelEncoder":
        if self.l2i:
            return self
        ul = sorted(
            {ann.label for s in samples for ann in s.annotations}
            - set(self.l2i_background)
        )
        self.l2i = {
            **self.l2i_background,
            **{label: idx for idx, label in enumerate(ul, start=1)},
        }
        self.i2l = {idx: label for label, idx in self.l2i.items()}
        return self

    def transform(
        self, samples: list[Sample[Annotation]]
    ) -> list[Sample[TrainAnnotation]]:
        return [
            Sample(
                file_name=sample.file_name,
                annotations=[
                    TrainAnnotation(
                        bboxes=ann.bbox,
                        scores=(ann.score,),
                        labels=(self.l2i[ann.label],),
                    )
                    for ann in sample.annotations
                ],
            )
            for sample in samples
        ]

    def fit_transform(
        self, samples: list[Sample[Annotation]]
    ) -> list[Sample[TrainAnnotation]]:
        return self.fit(samples).transform(samples)

    def inverse_transform(
        self, samples: list[Sample[TrainAnnotation]]
    ) -> list[Sample[Annotation]]:
        return [
            Sample(
                file_name=sample.file_name,
                annotations=[
                    Annotation(
                        bbox=ann.bboxes,
                        score=ann.scores[0],
                        label=self.i2l[int(ann.labels[0])],
                    )
                    for ann in sample.annotations
                ],
            )
            for sample in samples
        ]


@dataclass
class DoNothingEncoder:
    l2i: dict[str, int] = field(default_factory=dict)
    i2l: dict[int, str] = field(default_factory=dict)
    n_classes: int = 0

    def fit_transform(self, samples: list[Sample]) -> list[Sample]:
        return samples

    def transform(self, samples: list[Sample]) -> list[Sample]:
        return samples

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]:
        return samples
