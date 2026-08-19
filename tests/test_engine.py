import pathlib
from collections.abc import Callable

import cv2
import matplotlib
import numpy as np
import pytest

from modelinhos.engine.lit import lightning_engine
from modelinhos.engine.simple import simple_engine
from modelinhos.engine.skorch import skorch_engine
from modelinhos.evaluation import mean_average_precision
from modelinhos.models.blazenet import RETINANET_F
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import Annotation, Sample, read_samples, save_samples
from modelinhos.zoo import build_blaze

# TODO: Unify the fixtures


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


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(simple_engine(max_epochs=1), id="simple"),
        pytest.param(skorch_engine(max_epochs=1), id="skorch"),
        pytest.param(lightning_engine(max_epochs=1), id="lightning"),
    ],
)
def test_engine(
    engine: Callable,
    resolution: tuple[int, int],
    dataset: pathlib.Path,
):

    matplotlib.use("Agg")
    data = read_samples(dataset)

    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
    ).fit(data)

    # It's the smallest one, check it overfits
    model = build_blaze(
        arch=RETINANET_F,
        resolution=resolution,
        lencoder=lencoder,
        engine=engine,
    )
    model.fit(data)
    y_pred = model.transform(data)
    m_ap = mean_average_precision(
        data,
        y_pred,
        model.label_encoder.l2i,
        resolution=resolution,
    )
    assert m_ap["mAP"].iloc[0] == pytest.approx(1.0)
