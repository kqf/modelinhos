import pathlib
from functools import partial
from typing import Callable

import cv2
import numpy as np
import pytest
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
)

from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import Annotation, Sample, read_samples, save_samples
from modelinhos.zoo import build_trainable_retina, build_trainable_ssd


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
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        label="dot",
        score=1.0,
    )
    return [Sample(file_name=file_name, annotations=[annotation])] * 32


@pytest.fixture
def dataset(data, tmp_path: pathlib.Path) -> pathlib.Path:
    # TODO: Save data to that file
    return save_samples(data, tmp_path / "data" / "annotations.json")


@pytest.mark.parametrize(
    "build_model",
    [
        partial(
            build_trainable_retina,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            epochs=1,
        ),
        partial(
            build_trainable_ssd,
            weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            epochs=1,
        ),
    ],
)
def test_pipeline(
    build_model: Callable,
    resolution: tuple[int, int],
    dataset: pathlib.Path,
):
    data = read_samples(dataset)

    # Learn l2i from the data itself (the full from-scratch path). The
    # encoder must be fit BEFORE building: the classification head is
    # sized from len(l2i) at construction time.
    lencoder = LabelEncoder(
        resolution=resolution,
        l2i={"__background__": 0, "dot": 1},
    ).fit(data)
    model = build_model(resolution=resolution, lencoder=lencoder)
    model.fit(data)
    y_pred = model.transform(data)
    m_ap = mean_average_precision(data, y_pred, model.label_encoder.l2i)
    assert m_ap["mAP"].iloc[0] == pytest.approx(1.0)

    aps = per_sample_metrics(data, y_pred, l2i=model.label_encoder.l2i)
    assert len(aps) == len(data)

    assert aps.iloc[0]["mAP"] == pytest.approx(1.0)
    visualize_fp_fn(aps, i2l=model.label_encoder.i2l)
