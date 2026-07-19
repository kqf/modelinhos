from collections import Counter
from dataclasses import dataclass
from typing import Callable

import cv2

from modelinhos.preprocess.labels import SampleEncoder
from modelinhos.sample import Annotation, Sample


@dataclass
class Linted:
    good: list[Sample]
    problematic: list[Sample]
    # Unreadable files, apart from problematic: plotting problematic
    # samples requires images that actually open.
    corrupt: list[Sample]
    # label -> instance count over the good samples: print it to decide
    # which classes are worth modelling and to build the task l2i.
    classes: dict[str, int]


def check_visible(
    annotation: Annotation,
    resolution: tuple[int, int],
    min_size: float = 12.0,
) -> bool:
    """A box is visible when both sides reach min_size pixels at the
    image's own resolution. Below the matcher's floor (see match_boxes)
    a box is never assigned an anchor, and the object it marks trains
    as background."""
    h, w = resolution
    xmin, ymin, xmax, ymax = annotation.bbox
    return (xmax - xmin) * w >= min_size and (ymax - ymin) * h >= min_size


def contains_annotations(
    sample: Sample,
    resolution: tuple[int, int],
    valid_annotation: Callable = check_visible,
) -> bool:
    """Keep samples with at least one valid annotation: drops empty
    images along with images whose objects are all below the floor."""
    return any(valid_annotation(a, resolution) for a in sample.annotations)


def contains_no_invalid_annotations(
    sample: Sample,
    resolution: tuple[int, int],
    valid_annotation: Callable = check_visible,
) -> bool:
    """Keep samples whose annotations are all valid. Unlike
    contains_annotations this keeps empty images (legitimate negative
    examples) and rejects any image carrying even one sub-floor box --
    the safe choice under hard-negative mining, where an invisible box
    is unlabeled foreground."""
    return all(valid_annotation(a, resolution) for a in sample.annotations)


def lint(
    dataset: list[Sample],
    valid_sample: Callable = contains_annotations,
) -> Linted:
    good: list[Sample] = []
    problematic: list[Sample] = []
    corrupt: list[Sample] = []
    for sample in dataset:
        image = cv2.imread(str(sample.file_name))
        if image is None:
            corrupt.append(sample)
            continue
        resolution = image.shape[0], image.shape[1]
        target = good if valid_sample(sample, resolution) else problematic
        target.append(sample)

    classes = Counter(
        annotation.label
        for sample in good
        for annotation in sample.annotations
    )
    return Linted(
        good=good,
        problematic=problematic,
        corrupt=corrupt,
        classes=dict(classes),
    )


def rename(
    dataset: list[Sample[Annotation]],
    lencoder: SampleEncoder,
    new: dict[str, str],
) -> list[Sample[Annotation]]:
    """Rename labels via new (old -> new); labels outside the mapping
    pass through. Every resulting label must be one the encoder knows
    -- loud failure here (before anything is touched), since an unknown
    label would otherwise surface as a KeyError deep inside
    lencoder.transform at training time."""
    unknown = {
        new.get(annotation.label, annotation.label)
        for sample in dataset
        for annotation in sample.annotations
    } - set(lencoder.l2i)
    if unknown:
        raise ValueError(
            f"labels {sorted(unknown)} are outside the encoder's space "
            f"{sorted(lencoder.l2i)} -- extend new to cover them"
        )
    for sample in dataset:
        for annotation in sample.annotations:
            annotation.label = new.get(annotation.label, annotation.label)
    return dataset


def normalize(data: list[Sample]) -> list[Sample]:
    def process(sample: Sample) -> Sample:
        image = cv2.imread(str(sample.file_name))
        h, w, _ = image.shape
        for ann in sample.annotations:
            x1, y1, x2, y2 = ann.bbox
            ann.bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
        return sample

    return list(map(process, data))
