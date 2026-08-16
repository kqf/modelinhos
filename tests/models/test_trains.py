import matplotlib

matplotlib.use("Agg")

import pathlib
from collections.abc import Callable
from functools import partial

import cv2
import numpy as np
import pytest

from modelinhos.engine.lit import lightning_engine
from modelinhos.engine.simple import simple_engine
from modelinhos.engine.skorch import skorch_engine
from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.models.blazenet import RETINANET_F
from modelinhos.models.fcos import FCOS_DELTA
from modelinhos.models.retinanet import RETINANET
from modelinhos.models.ssdlite import SSDLITE
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import Annotation, Sample, read_samples, save_samples
from modelinhos.zoo import build_blaze, build_fcos, build_retina, build_ssd


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
    "build_model",
    [
        # pytest.param(
        #     partial(build_retina, arch=RETINANET),
        #     id="retinanet",
        # ),
        # pytest.param(
        #     partial(build_ssd, arch=SSDLITE),
        #     id="ssdlite",
        # ),
        pytest.param(
            build_retina,
            id="torchvision_retinanet",
            marks=pytest.mark.skip(reason="Tuned, works"),
        ),
        pytest.param(
            build_ssd,
            id="torchvision_ssdlite",
            marks=pytest.mark.skip(reason="Tuned, works"),
        ),
        # The vanilla BLAZEFACE recipes keep MediaPipe's full-image
        # anchors, which never match a small box -- only the retina-
        # anchored trainable flavor can learn this dataset.
        # pytest.param(
        #     partial(build_blaze, arch=RETINANET_F),
        #     id="blazenet",
        # ),
        # FCOS on RetinaNet's anchors and codec -- the same training
        # problem as the retinanet flavor with only the head swapped
        # (see models/fcos.py for the A/B rationale). Unlike retinanet,
        # its warm start is convention-mismatched (the FCOS checkpoint's
        # relu'd ltrb regression and 91 COCO class channels must be
        # unlearned before the delta codec works), so one shared epoch
        # is not enough -- test_fcos_delta_converges below trains it
        # with its own budget.
        # pytest.param(
        #     partial(build_fcos, arch=FCOS_DELTA),
        #     id="fcos",
        #     marks=pytest.mark.skip(reason="Needs 5 epochs"),
        # ),
        pytest.param(
            build_fcos,
            id="torchvision_fcos",
        ),
    ],
)
@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(simple_engine(max_epochs=1), id="simple"),
        # pytest.param(skorch_engine(max_epochs=1), id="skorch"),
        # pytest.param(lightning_engine(max_epochs=1), id="lightning"),
    ],
)
def test_pipeline(
    build_model: Callable,
    engine: Callable,
    resolution: tuple[int, int],
    dataset: pathlib.Path,
):
    data = read_samples(dataset)

    # Learn l2i from the data itself (the full from-scratch path). The
    # encoder must be fit BEFORE building: the classification head is
    # sized from lencoder.n_classes at construction time.
    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
    ).fit(data)

    model = build_model(
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

    aps = per_sample_metrics(
        data,
        y_pred,
        l2i=model.label_encoder.l2i,
        resolution=resolution,
    )
    assert len(aps) == len(data)

    assert aps.iloc[0]["mAP"] == pytest.approx(1.0)
    visualize_fp_fn(aps, i2l=model.label_encoder.i2l)
