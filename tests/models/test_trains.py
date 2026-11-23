import pathlib

import matplotlib
import pytest

from modelinhos.detector import DetectionRecipe, build_detector
from modelinhos.engine.simple import simple_engine
from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.models.blazenet import RETINANET_F
from modelinhos.models.fcos import FCOS_DELTA, TORCHVISION_FCOS
from modelinhos.models.load import warm_start
from modelinhos.models.retinanet import RETINANET, TORCHVISION_RETINANET
from modelinhos.models.ssdlite import SSDLITE, TORCHVISION_SSDLITE
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import read_samples


@pytest.mark.parametrize(
    "arch, max_epochs",
    [
        pytest.param(
            RETINANET,
            1,
            id="retinanet",
        ),
        pytest.param(
            SSDLITE,
            1,
            id="ssdlite",
        ),
        pytest.param(
            TORCHVISION_RETINANET,
            1,
            id="torchvision_retinanet",
        ),
        pytest.param(
            TORCHVISION_SSDLITE,
            1,
            id="torchvision_ssdlite",
        ),
        # The vanilla BLAZEFACE recipes keep MediaPipe's full-image
        # anchors, which never match a small box -- only the retina-
        # anchored trainable flavor can learn this dataset.
        pytest.param(
            RETINANET_F,
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
            FCOS_DELTA,
            5,
            id="fcos",
        ),
        pytest.param(
            TORCHVISION_FCOS,
            1,
            id="torchvision_fcos",
        ),
    ],
)
def test_pipeline(
    arch: DetectionRecipe,
    max_epochs: int,
    resolution: tuple[int, int],
    dataset: pathlib.Path,
):
    matplotlib.use("Agg")
    data = read_samples(dataset)

    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
    ).fit(data)

    model = build_detector(
        resolution=resolution,
        lencoder=lencoder,
        arch=arch,
        weights=warm_start(
            arch.pretraining,
            fresh=arch.diff,
            progress=False,
        ),
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
