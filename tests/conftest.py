import pathlib

import cv2
import numpy as np
import pytest

from modelinhos.sample import Annotation, Sample, save_samples


@pytest.fixture
def resolution() -> tuple[int, int]:
    return 120, 160


@pytest.fixture
def data(
    resolution: tuple[int, int],
    tmp_path: pathlib.Path,
    size: int = 32,
) -> list[Sample]:
    # A single image with a black dot in the center, repeated across
    # several samples so there's more than one batch per epoch.
    h, w = resolution
    image = np.full((h, w, 3), 255, dtype=np.uint8)

    cx, cy = w // 2, h // 2
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = cx + size // 2, cy + size // 2
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    file_name = tmp_path / f"dot_{size}.png"
    cv2.imwrite(str(file_name), image)

    annotation = Annotation(
        bbox=(float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h),
        label="dot",
        score=1.0,
    )
    return [Sample(file_name=file_name, annotations=[annotation])] * 32


@pytest.fixture
def dataset(data, tmp_path: pathlib.Path) -> pathlib.Path:
    return save_samples(data, tmp_path / "data" / "annotations.json")
