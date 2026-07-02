import pathlib

import pytest
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.processing import LabelEncoder
from modelinhos.sample import read_samples
from modelinhos.ssd.inference import TorchvisionDetector, torchvision_model


@pytest.fixture
def model(resolution: tuple[int, int]):
    return TorchvisionDetector(
        build_model=torchvision_model(
            ssdlite320_mobilenet_v3_large, resolution
        ),
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        lencoder=LabelEncoder(l2i={"person": 1, "tie": 34}),
    )


def train_test_split(data):
    print("Not implemented")
    return data, data


@pytest.fixture
def dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path("tests/assets/annotations.json")


def test_pipeline(model: TorchvisionDetector, dataset: pathlib.Path):
    samples = read_samples(dataset)
    train, valid = train_test_split(samples)
    # We don't fit in this repo ~
    # model.fit(X_train, y_train) ~
    y_pred = model.transform(valid)
    m_ap = mean_average_precision(valid, y_pred, l2i={"person": 0, "tie": 1})
    assert m_ap["mAP"].iloc[0] == pytest.approx(0.5)

    aps = per_sample_metrics(valid, y_pred, l2i=model.label_encoder.l2i)
    assert len(aps) == len(valid)
    assert aps.iloc[0]["mAP"] == pytest.approx(0.028571429)
    visualize_fp_fn(aps, i2l=model.label_encoder.i2l)
