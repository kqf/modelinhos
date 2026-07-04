import logging
from copy import deepcopy
from dataclasses import dataclass, field

from modelinhos.sample import Sample

# module logger
logger = logging.getLogger(__name__)


@dataclass
class LabelEncoder:
    l2i: dict[str, int] = field(default_factory=dict)
    i2l: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.l2i and not self.i2l:
            self.i2l = {v: k for k, v in self.l2i.items()}
        elif self.i2l and not self.l2i:
            self.l2i = {v: k for k, v in self.i2l.items()}

    def fit(self, samples: list[Sample]) -> "LabelEncoder":
        if self.l2i and self.i2l:
            logger.info("Already fitted, skipping for now.")
            return self

        ul = sorted({ann.labels for s in samples for ann in s.annotations})
        self.l2i = {label: idx for idx, label in enumerate(ul)}
        self.i2l = {idx: label for label, idx in self.l2i.items()}
        return self

    def transform(self, samples: list[Sample]) -> list[Sample]:
        samples = [deepcopy(s) for s in samples]
        for sample in samples:
            for ann in sample.annotations:
                # TODO: Fix me later, ignore for now
                ann.labels = self.l2i[ann.labels]  # type: ignore
        return samples

    def fit_transform(self, samples: list[Sample]) -> list[Sample]:
        return self.fit(samples).transform(samples)

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]:
        samples = [deepcopy(s) for s in samples]
        for sample in samples:
            for ann in sample.annotations:
                ann.labels = self.i2l[int(ann.labels)]
        return samples


class DoNothingEncoder:
    l2i: dict[str, int] = {}
    i2l: dict[int, str] = {}

    def fit_transform(self, samples: list[Sample]) -> list[Sample]:
        return samples

    def transform(self, samples: list[Sample]) -> list[Sample]:
        return samples

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]:
        return samples
