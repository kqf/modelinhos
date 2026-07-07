import pathlib

import cv2
import numpy as np
import pytest
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)

from modelinhos.detector import Detector
from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
)
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import Annotation, Sample
from modelinhos.zoo import build_trainable_retina, build_trainable_ssd

# One entry per architecture under test. anchor_size is picked to match
# that architecture's own smallest anchor (see ssd/ssdlite.py /
# ssd/retinanet.py), so the synthetic dot lands on a well-matched anchor.
ARCHITECTURES = {
    "ssd": {
        "build_trainable": build_trainable_ssd,
        "weights": SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        "anchor_size": 32,
        "epochs": 5,
    },
    "retina": {
        "build_trainable": build_trainable_retina,
        "weights": RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        "anchor_size": 16,
        # heads train from scratch even with a pretrained warm start (see
        # build_trainable_retina), so give it a few more epochs than SSD.
        "epochs": 8,
    },
}


@pytest.fixture
def resolution() -> tuple[int, int]:
    # 480p (height, width) -- overrides the 800x1088 shared ssd fixture,
    # this test doesn't need it and it'd make training slow.
    return 480, 640


@pytest.fixture(params=["ssd", "retina"])
def architecture(request) -> dict:
    return ARCHITECTURES[request.param]


def _dot_samples(
    resolution: tuple[int, int],
    tmp_path: pathlib.Path,
    size: int,
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
    return [Sample(file_name=file_name, annotations=[annotation])] * 8


@pytest.fixture
def model(resolution: tuple[int, int], architecture: dict):
    # resolution is what makes the encoder normalise GT boxes to [0, 1] --
    # without it they stay in pixel space, never overlap the normalised
    # anchors, and the loss silently trains on zero positives.
    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
        resolution=resolution,
    )
    return architecture["build_trainable"](
        resolution,
        lencoder=lencoder,
        weights=architecture["weights"],
        epochs=architecture["epochs"],
    )


@pytest.fixture
def train(
    resolution: tuple[int, int],
    tmp_path: pathlib.Path,
    architecture: dict,
) -> list[Sample]:
    return _dot_samples(resolution, tmp_path, size=architecture["anchor_size"])


def test_pipeline(model: Detector, train: list[Sample]):
    model.fit(train)
    y_pred = model.transform(train)
    m_ap = mean_average_precision(train, y_pred, model.label_encoder.l2i)
    # mAP averages only over classes present in the GT, so the reserved
    # background slot no longer caps a perfect single-class run at 0.5
    assert m_ap["mAP"].iloc[0] == pytest.approx(1.0)

    aps = per_sample_metrics(train, y_pred, l2i=model.label_encoder.l2i)
    assert len(aps) == len(train)
