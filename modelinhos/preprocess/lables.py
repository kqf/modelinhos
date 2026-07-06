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

    def fit_transform(self, samples: list[Sample]) -> list[Sample]: ...

    def transform(self, samples: list[Sample]) -> list[Sample]: ...

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]: ...


@dataclass
class LabelEncoder:
    l2i: dict[str, int] = field(default_factory=dict)
    i2l: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.l2i and not self.i2l:
            self.i2l = {v: k for k, v in self.l2i.items()}
        elif self.i2l and not self.l2i:
            self.l2i = {v: k for k, v in self.i2l.items()}

    def fit(self, samples: list[Sample[Annotation]]) -> "LabelEncoder":
        if self.l2i and self.i2l:
            return self
        ul = sorted({ann.label for s in samples for ann in s.annotations})
        self.l2i = {label: idx for idx, label in enumerate(ul)}
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


class DoNothingEncoder:
    l2i: dict[str, int] = {}
    i2l: dict[int, str] = {}

    def fit_transform(self, samples: list[Sample]) -> list[Sample]:
        return samples

    def transform(self, samples: list[Sample]) -> list[Sample]:
        return samples

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]:
        return samples
