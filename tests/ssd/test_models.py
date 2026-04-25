from functools import partial

import cv2
import numpy as np
import pytest
import torch
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from modelinhos.ssd.retianent_tv import (
    build_inference_model,
    build_retinanet_torchvision,
)
from modelinhos.ssd.retinanet import build_vanilla_ssd


@pytest.fixture
def batch(resolution: tuple[int, int]) -> torch.Tensor:
    return torch.rand(1, 3, *resolution)


@pytest.mark.skip
@pytest.mark.parametrize(
    "build_model",
    [
        (
            partial(
                build_vanilla_ssd,
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            ),
        ),
        (
            partial(
                build_retinanet_torchvision,
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
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
    return pad(image, *resolution)


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
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def to_blob(frame: np.ndarray, weights) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    preprocess = weights.transforms()
    return preprocess(tensor).unsqueeze(0)


@pytest.mark.parametrize(
    "build_model",
    [
        lambda weights, resolution: retinanet_resnet50_fpn_v2(weights=weights),
        build_inference_model(build_retinanet_torchvision),
    ],
)
def test_weights_match(frame, build_model):
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = build_model(weights=weights, resolution=frame.shape[:2])
    model.eval()
    blob = to_blob(frame, weights)
    with torch.no_grad():
        predictions = model(blob.clone())
    plot_predictions(frame, predictions)
