import pathlib

import pytest

from modelinhos.analysis.distributions import boxes, divergence, labels
from modelinhos.sample import Annotation, Sample


@pytest.fixture
def dataset(
    wide=(0.1, 0.1, 0.5, 0.3),  # w=0.4 h=0.2 area=0.08 aspect=2.0
    square=(0.2, 0.2, 0.4, 0.4),  # w=0.2 h=0.2 area=0.04 aspect=1.0
    tall=(0.5, 0.1, 0.6, 0.5),  # w=0.1 h=0.4 area=0.04 aspect=0.25
) -> list[Sample]:
    return [
        Sample(
            file_name=pathlib.Path("a.png"),
            annotations=[
                Annotation(bbox=wide, label="person"),
                Annotation(bbox=square, label="person"),
            ],
        ),
        Sample(
            file_name=pathlib.Path("b.png"),
            annotations=[Annotation(bbox=tall, label="dog")],
        ),
        Sample(file_name=pathlib.Path("c.png"), annotations=[]),
    ]


def test_boxes(dataset: list[Sample]):
    geometry = boxes(dataset)

    # One row per box: the empty image contributes nothing
    assert geometry.file.tolist() == ["a.png", "a.png", "b.png"]
    assert geometry.label.tolist() == ["person", "person", "dog"]
    assert geometry.w.tolist() == pytest.approx([0.4, 0.2, 0.1])
    assert geometry.h.tolist() == pytest.approx([0.2, 0.2, 0.4])
    assert geometry.area.tolist() == pytest.approx([0.08, 0.04, 0.04])
    assert geometry.aspect.tolist() == pytest.approx([2.0, 1.0, 0.25])


def test_labels(dataset: list[Sample]):
    counts = labels(dataset)

    # Observed labels only: whether they cover the task is a verdict
    assert counts.label.tolist() == ["person", "dog"]
    assert counts["count"].tolist() == [2, 1]
    assert counts.share.tolist() == pytest.approx([2 / 3, 1 / 3])


def test_divergence(dataset: list[Sample]):
    geometry = boxes(dataset)
    drift = divergence(geometry, geometry.assign(w=geometry.w + 1))

    report = drift.set_index("column")
    # Only shared numeric columns are compared: file/label stay out
    assert report.index.tolist() == ["w", "h", "area", "aspect"]
    # Disjoint w distributions max out the CDF gap, the rest are equal
    assert report.ks.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert report.drifted.tolist() == [True, False, False, False]
    assert report.reference_mean["w"] == pytest.approx(0.7 / 3)
    assert report.other_mean["w"] == pytest.approx(3.7 / 3)
