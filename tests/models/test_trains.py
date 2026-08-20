import pathlib
from collections.abc import Callable
from functools import partial

import matplotlib
import pytest

from modelinhos.engine.simple import simple_engine
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
from modelinhos.sample import read_samples
from modelinhos.zoo import build_blaze, build_fcos, build_retina, build_ssd


@pytest.mark.parametrize(
    "build_model, max_epochs",
    [
        pytest.param(
            partial(build_retina, arch=RETINANET),
            1,
            id="retinanet",
        ),
        pytest.param(
            partial(build_ssd, arch=SSDLITE),
            1,
            id="ssdlite",
        ),
        pytest.param(
            build_retina,
            1,
            id="torchvision_retinanet",
        ),
        pytest.param(
            build_ssd,
            1,
            id="torchvision_ssdlite",
        ),
        # The vanilla BLAZEFACE recipes keep MediaPipe's full-image
        # anchors, which never match a small box -- only the retina-
        # anchored trainable flavor can learn this dataset.
        pytest.param(
            partial(build_blaze, arch=RETINANET_F),
            1,
            id="blazenet",
        ),
        # FCOS on RetinaNet's anchors and codec -- the same training
        # problem as the retinanet flavor with only the head swapped
        # (see models/fcos.py for the A/B rationale). Unlike retinanet,
        # its warm start is convention-mismatched (the FCOS checkpoint's
        # relu'd ltrb regression and 91 COCO class channels must be
        # unlearned before the delta codec works), so it needs a larger
        # epoch budget than the rest.
        pytest.param(
            partial(build_fcos, arch=FCOS_DELTA),
            5,
            id="fcos",
        ),
        pytest.param(
            build_fcos,
            1,
            id="torchvision_fcos",
        ),
    ],
)
def test_pipeline(
    build_model: Callable,
    max_epochs: int,
    resolution: tuple[int, int],
    dataset: pathlib.Path,
):
    matplotlib.use("Agg")
    data = read_samples(dataset)

    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
    ).fit(data)

    model = build_model(
        resolution=resolution,
        lencoder=lencoder,
        engine=simple_engine(max_epochs=max_epochs),
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
