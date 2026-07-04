import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dacite import Config, from_dict
from dataclasses_json import dataclass_json

AbsoluteXYXY = tuple[float, float, float, float]


@dataclass_json
@dataclass
class Annotation:
    bboxes: AbsoluteXYXY
    labels: str
    scores: float = float("nan")


@dataclass_json
@dataclass
class Sample:
    file_name: Path
    annotations: list[Annotation]


def to_sample(entry: dict[str, Any]) -> Sample:
    try:
        return from_dict(
            data_class=Sample,
            data=entry,
            config=Config(cast=[tuple, Path]),
        )
    except Exception as e:
        print(f"Failed to parse entry: {entry}")
        raise e


def read_samples(path: Path, relative=True) -> list[Sample]:
    path = Path(path)
    with open(path) as f:
        df = json.load(f)
    samples = [to_sample(x) for x in df if x]
    if relative:
        for sample in samples:
            sample.file_name = path.parent / sample.file_name
    return list(samples)


def save_samples(samples: list[Sample], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([sample.to_dict() for sample in samples], f)  # type: ignore
