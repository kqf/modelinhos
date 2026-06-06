from functools import partial

import cv2
import numpy as np
import pytest
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from modelinhos.plot import plot
from modelinhos.processing import LabelEncoder
from modelinhos.ssd.inference import Detector, TorchvisionDetector
from modelinhos.ssd.lite import (
    build_ssdlite,
    build_torchvision_ssdlite,
    ssd_normalize,
    ssd_postprocess,
)
from modelinhos.ssd.retinanet import (
    build_torchvision_retinanet,
    bulid_retinanet,
)


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


@pytest.mark.parametrize(
    "build_model",
    [
        partial(
            TorchvisionDetector,
            build_model=ssdlite320_mobilenet_v3_large,
            weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        ),
        partial(
            Detector,
            build_model=build_torchvision_ssdlite,
            weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            postprocess=ssd_postprocess,
            normalize=ssd_normalize,
        ),
        partial(
            Detector,
            build_model=partial(build_ssdlite, n_classes=91),
            th=0.01,
            weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
        ),
        partial(
            TorchvisionDetector,
            build_model=retinanet_resnet50_fpn_v2,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
        partial(
            Detector,
            build_model=build_torchvision_retinanet,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
        partial(
            Detector,
            build_model=partial(bulid_retinanet, n_classes=91),
            th=0.05,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
        partial(
            Detector,
            build_model=partial(bulid_retinanet, n_classes=2),
            th=0.01,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
        partial(
            Detector,
            build_model=partial(bulid_retinanet, n_classes=92 * 2),
            th=0.01,
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        ),
    ],
)
def test_weights_match(frame, build_model, headless):
    model = build_model(frame.shape[:2])
    predictions = model.transform([frame])[0]

    # Convert predictions to meaningful labels
    labels = model.weights.meta["categories"]
    le = LabelEncoder(l2i={label: i for i, label in enumerate(labels)})
    predictions = le.inverse_transform(predictions)

    frame = plot(frame, predictions)

    # sourcery skip: no-conditionals-in-tests
    if not headless:
        cv2.imshow("Predictions", frame)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
