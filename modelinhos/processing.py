from dataclasses import dataclass, field

from modelinhos.sample import Sample


@dataclass
class LabelEncoder:
    l2i: dict[str, int] = field(default_factory=dict)
    i2l: dict[int, str] = field(default_factory=dict)

    def fit(self, samples: list[Sample]) -> "LabelEncoder":
        ul = sorted({ann.label for s in samples for ann in s.annotations})
        self.l2i = {label: idx for idx, label in enumerate(ul)}
        self.i2l = {idx: label for label, idx in self.l2i.items()}
        return self

    def transform(self, samples: list[Sample]) -> list[Sample]:
        for sample in samples:
            for ann in sample.annotations:
                ann.label = str(self.l2i[ann.label])
        return samples

    def fit_transform(self, samples: list[Sample]) -> list[Sample]:
        return self.fit(samples).transform(samples)

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]:
        for sample in samples:
            for ann in sample.annotations:
                ann.label = self.i2l[int(ann.label)]
        return samples
