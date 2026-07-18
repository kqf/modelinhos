from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from modelinhos.evaluation import (
    per_sample_metrics,
    per_size_metrics,
    visualize_map_size,
)
from modelinhos.sample import Annotation, Sample


@pytest.fixture
def y_true() -> list[Sample]:
    return [
        Sample(
            file_name=Path("frame.jpg"),
            annotations=[
                Annotation(bbox=(0.1, 0.1, 0.9, 0.9), label="person"),
                Annotation(bbox=(0.0, 0.0, 0.1, 0.1), label="person"),
            ],
        )
    ]


@pytest.fixture
def y_pred() -> list[Sample]:
    return [
        Sample(
            file_name=Path("frame.jpg"),
            annotations=[
                Annotation(
                    bbox=(0.1, 0.1, 0.9, 0.9),
                    label="person",
                    score=0.9,
                ),
            ],
        )
    ]


def test_per_sample_metrics_counts_at_threshold():
    y_true = [
        Sample(
            file_name=Path("frame.jpg"),
            annotations=[
                Annotation(bbox=(0.1, 0.1, 0.5, 0.5), label="person"),
            ],
        )
    ]
    # A confident miss plus a sub-threshold hit: at threshold 0.5 the
    # hit does not count, so the image has one FP and one FN. Counting
    # the last point of the PR curve instead would report a clean image.
    y_pred = [
        Sample(
            file_name=Path("frame.jpg"),
            annotations=[
                Annotation(
                    bbox=(0.6, 0.6, 0.9, 0.9),
                    label="person",
                    score=0.9,
                ),
                Annotation(
                    bbox=(0.1, 0.1, 0.5, 0.5),
                    label="person",
                    score=0.3,
                ),
            ],
        )
    ]

    df = per_sample_metrics(
        y_true,
        y_pred,
        l2i={"person": 0},
        resolution=(100, 100),
        threshold=0.5,
    )

    row = df.iloc[0]
    assert row["tp"] == 0
    assert row["fp"] == 1
    assert row["fn"] == 1


def test_per_size_metrics(y_true, y_pred):
    df = per_size_metrics(
        y_true,
        y_pred,
        l2i={"person": 0},
        resolution=(100, 100),
        bins=[0, 32, 100],
    )

    # The 10 px box is missed entirely: a pure miss must still show up.
    small = df[df["size_lo"] == 0].iloc[0]
    assert small["mAP"] == pytest.approx(0.0)
    assert small["tp"] == 0
    assert small["fp"] == 0
    assert small["fn"] == 1

    # The 80 px box is matched perfectly.
    large = df[df["size_lo"] == 32].iloc[0]
    assert large["mAP"] == pytest.approx(1.0)
    assert large["tp"] == 1
    assert large["fp"] == 0
    assert large["fn"] == 0

    plt.switch_backend("agg")
    visualize_map_size(df)
