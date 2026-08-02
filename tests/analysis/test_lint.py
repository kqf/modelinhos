import pathlib

import cv2
import numpy as np
import pytest

from modelinhos.analysis.lint import (
    contains_annotations,
    contains_no_invalid_annotations,
    lint,
    rename,
)
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import Annotation, Sample


@pytest.fixture
def resolution() -> tuple[int, int]:
    return 120, 160


@pytest.fixture
def dataset(
    resolution: tuple[int, int],
    tmp_path: pathlib.Path,
) -> list[Sample]:
    h, w = resolution
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    file_names = []
    for i in range(4):
        file_name = tmp_path / f"blank_{i}.png"
        cv2.imwrite(str(file_name), image)
        file_names.append(file_name)
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(b"not really a png")

    big = (0.1, 0.1, 0.1 + 32 / w, 0.1 + 32 / h)
    tiny = (0.5, 0.5, 0.5 + 4 / w, 0.5 + 4 / h)
    return [
        Sample(
            file_name=file_names[0],
            annotations=[Annotation(bbox=big, label="person")],
        ),
        Sample(
            file_name=file_names[1],
            annotations=[
                Annotation(bbox=big, label="person"),
                Annotation(bbox=big, label="car"),
            ],
        ),
        Sample(
            file_name=file_names[2],
            annotations=[Annotation(bbox=tiny, label="person")],
        ),
        Sample(file_name=file_names[3], annotations=[]),
        Sample(
            file_name=tmp_path / "missing.png",
            annotations=[Annotation(bbox=big, label="person")],
        ),
        Sample(
            file_name=truncated,
            annotations=[Annotation(bbox=big, label="person")],
        ),
    ]


@pytest.mark.parametrize(
    "valid_sample, good, problematic",
    [
        # The default drops the empty image with the sub-floor one ...
        pytest.param(
            contains_annotations,
            [0, 1],
            [2, 3],
            id="contains_annotations",
        ),
        # ... the all-valid criterion keeps it as a legitimate negative
        pytest.param(
            contains_no_invalid_annotations,
            [0, 1, 3],
            [2],
            id="contains_no_invalid_annotations",
        ),
    ],
)
def test_lints(
    dataset: list[Sample],
    valid_sample,
    good: list[int],
    problematic: list[int],
):
    linted = lint(dataset, valid_sample=valid_sample)

    # The clean part -- proposed to be saved
    assert linted.good == [dataset[i] for i in good]

    # The problematic part -- proposed to be plotted
    assert linted.problematic == [dataset[i] for i in problematic]

    # Unreadable either way -- nothing to plot here
    assert linted.corrupt == dataset[4:]

    # Printed to decide which labels are worth modelling
    assert linted.classes == {"person": 2, "car": 1}


def test_renames(dataset: list[Sample]):
    lencoder = LabelEncoder(
        l2i={"__background__": 0, "person": 1, "other": 2},
    )

    # A label outside the encoder's space fails loudly ...
    with pytest.raises(ValueError, match="car"):
        rename(dataset, lencoder, new={})

    # ... and before anything is touched: no annotation was renamed
    assert [
        annotation.label
        for sample in dataset
        for annotation in sample.annotations
    ] == ["person", "person", "car", "person", "person", "person"]

    renamed = rename(dataset, lencoder, new={"car": "other"})

    # In-place, same list: mapped labels change, the rest pass through
    assert renamed is dataset
    assert [
        annotation.label
        for sample in renamed
        for annotation in sample.annotations
    ] == ["person", "person", "other", "person", "person", "person"]
