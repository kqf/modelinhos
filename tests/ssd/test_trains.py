import pathlib

import cv2
import numpy as np
import pytest
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)

from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
)
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import Annotation, Sample
from modelinhos.ssd.inference import Detector
from modelinhos.zoo import build_trainable_retina, build_trainable_ssd


@pytest.fixture
def resolution() -> tuple[int, int]:
    # 480p (height, width) -- overrides the 800x1088 shared ssd fixture,
    # this test doesn't need it and it'd make training slow.
    return 480, 640


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
def model(resolution: tuple[int, int]):
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
        resolution=resolution,
    )
    return build_trainable_ssd(
        resolution,
        lencoder=lencoder,
        weights=weights,
        epochs=5,
    )


@pytest.fixture
def train(resolution: tuple[int, int], tmp_path: pathlib.Path) -> list[Sample]:
    # 32px matches build_ssdlite's smallest anchor (see ssd/ssdlite.py)
    return _dot_samples(resolution, tmp_path, size=32)


def test_pipeline(model: Detector, train: list[Sample]):
    model.fit(train)
    y_pred = model.transform(train)
    m_ap = mean_average_precision(train, y_pred, model.label_encoder.l2i)
    assert m_ap["mAP"].iloc[0] == pytest.approx(1.0)

    aps = per_sample_metrics(train, y_pred, l2i=model.label_encoder.l2i)
    assert len(aps) == len(train)


@pytest.fixture
def model_retina(resolution: tuple[int, int]):
    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
        resolution=resolution,
    )
    # No pretrained detection weights here (see build_trainable_retina) --
    # the backbone is still ImageNet-pretrained (RetinaNetPure always loads
    # that), only the heads train from scratch, so this needs a few more
    # epochs than the SSD case to converge.
    return build_trainable_retina(
        resolution,
        lencoder=lencoder,
        epochs=8,
    )


@pytest.fixture
def train_retina(
    resolution: tuple[int, int], tmp_path: pathlib.Path
) -> list[Sample]:
    # 16px matches bulid_retinanet's smallest anchor (see ssd/retinanet.py)
    return _dot_samples(resolution, tmp_path, size=16)


def test_pipeline_retina(model_retina: Detector, train_retina: list[Sample]):
    model_retina.fit(train_retina)
    y_pred = model_retina.transform(train_retina)
    m_ap = mean_average_precision(
        train_retina, y_pred, model_retina.label_encoder.l2i
    )
    assert m_ap["mAP"].iloc[0] == pytest.approx(1.0)

    aps = per_sample_metrics(
        train_retina, y_pred, l2i=model_retina.label_encoder.l2i
    )
    assert len(aps) == len(train_retina)
