import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from dacite import Config, from_dict
from dataclasses_json import config, dataclass_json

# Boxes live in relative coordinates (fractions of image size, [0, 1])
# everywhere inside the library. Pixels exist only at the edges: ingest
# converts to relative once, plotting and evaluation scale back by the
# image / evaluation resolution they have in hand.
RelativeXYXY = tuple[float, float, float, float]


@dataclass_json
@dataclass
class Annotation:
    bbox: RelativeXYXY
    label: str
    score: float = float("nan")


@dataclass(frozen=True)
class TrainAnnotation:
    bboxes: RelativeXYXY
    labels: tuple[int, ...]
    scores: tuple[float, ...]


AnnotationT = TypeVar("AnnotationT")


@dataclass_json
@dataclass
class Sample(Generic[AnnotationT]):
    file_name: Path = field(metadata=config(encoder=str))
    annotations: list[AnnotationT]


def to_sample(entry: dict[str, Any]) -> Sample[Annotation]:
    try:
        return from_dict(
            data_class=Sample[Annotation],
            data=entry,
            config=Config(cast=[tuple, Path]),
        )
    except Exception:
        print(f"Failed to parse entry: {entry}")
        raise


def read_samples(
    path: Path,
    relative: bool = True,
) -> list[Sample[Annotation]]:
    path = Path(path)
    with open(path) as f:
        df = json.load(f)
    samples = [to_sample(x) for x in df if x]
    # Loud failure for pixel-space annotations: everything downstream
    # assumes relative coordinates, and pixel boxes would be silently
    # wrong (tiny) rather than broken. Inside the tolerance the
    # coordinates are clamped rather than rejected: annotation files
    # routinely carry a hair of float noise past the edge, and strict
    # consumers (albumentations validates its bbox range) would raise
    # on it mid-training. Clamping here is what lets everything
    # downstream assume [0, 1] exactly.
    for sample in samples:
        for ann in sample.annotations:
            if not all(-0.01 <= c <= 1.01 for c in ann.bbox):
                raise ValueError(
                    f"{sample.file_name}: bbox {ann.bbox} is not in "
                    "relative coordinates -- annotations must be "
                    "normalized to [0, 1] fractions of the image size"
                )
            ann.bbox = tuple(min(max(c, 0.0), 1.0) for c in ann.bbox)  # type: ignore
    if relative:
        for sample in samples:
            sample.file_name = path.parent / sample.file_name
    return samples


def save_samples(samples: list[Sample[Annotation]], path: Path) -> Path:
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        json.dump([sample.to_dict() for sample in samples], f)  # type: ignore
    return path
