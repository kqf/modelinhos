from functools import partial

import cv2
import numpy as np
import pytest
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from modelinhos.ssd.inference import build_inference_model, to_blob
from modelinhos.ssd.lite import (
    build_lite_torchvision,
    build_pure_ssd_lite,
    ssd_normalize,
    ssd_postprocess,
)
from modelinhos.ssd.retianent_tv import (
    build_retinanet_torchvision,
)
from modelinhos.ssd.retinanet import build_vanilla_ssd


@pytest.fixture
def batch(resolution: tuple[int, int]) -> torch.Tensor:
    return torch.rand(1, 3, *resolution)


@pytest.mark.parametrize(
    "build_model",
    [
        (
            partial(
                build_vanilla_ssd,
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            )
        ),
        (
            partial(
                build_retinanet_torchvision,
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            )
        ),
        (
            partial(
                build_lite_torchvision,
                weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            )
        ),
    ],
)
def test_ssd(
    build_model,
    resolution,
    batch,
    n_classes=91,
):
    model, priors = build_model(n_classes=n_classes, resolution=resolution)
    boxes, classes = model(batch)
    assert boxes.shape == (1, *priors.shape)
    assert classes.shape == (1, priors.shape[0], n_classes)


def pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]

    square = target_h == target_w
    target_h = max(target_h, h)
    target_w = max(target_w, w)
    if square:
        target_h = max(target_h, target_w)
        target_w = max(target_w, target_w)

    t = (target_h - h) // 2
    b = target_h - h - t
    l = (target_w - w) // 2  # noqa
    r = target_w - w - l

    return cv2.copyMakeBorder(
        image,
        t,
        b,
        l,
        r,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


@pytest.fixture
def frame(resolution, path: str = "tests/assets/person.jpg") -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        pytest.skip(f"Asset not found: {path}")
    return cv2.resize(pad(image, *resolution), resolution[::-1])


def plot_predictions(image_bgr, predictions, score_threshold=0.5):
    pred = predictions[0]
    boxes = pred["boxes"].numpy()
    scores = pred["scores"].numpy()
    print()
    for box, score in zip(boxes, scores):
        if score < score_threshold:
            continue
        print(score)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Predictions", image_bgr)
    cv2.waitKey(1)
    cv2.destroyAllWindows()


def build_inference_model_torchvision(model_builder, weights):
    model = model_builder(weights=weights)
    model.eval()

    def build(resolution):
        def infer(frame: np.ndarray):
            blob = to_blob(frame, weights)
            return model(blob)

        return infer

    return build


@pytest.mark.parametrize(
    "build_model",
    [
        build_inference_model_torchvision(
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        ),
        build_inference_model(
            build_lite_torchvision,
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            postprocess=ssd_postprocess,
            normalize=ssd_normalize,
        ),
        build_inference_model(
            partial(build_pure_ssd_lite, n_classes=91),
            th=0.01,
            weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        ),
        build_inference_model_torchvision(
            retinanet_resnet50_fpn_v2,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
        build_inference_model(
            build_retinanet_torchvision,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
        build_inference_model(
            partial(build_vanilla_ssd, n_classes=91),
            th=0.05,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
        build_inference_model(
            partial(build_vanilla_ssd, n_classes=2),
            th=0.01,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
    ],
)
def test_weights_match(frame, build_model):
    model = build_model(frame.shape[:2])
    with torch.no_grad():
        predictions = model(frame)
    plot_predictions(frame, predictions, score_threshold=0.4)
