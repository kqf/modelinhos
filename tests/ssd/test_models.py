from functools import partial

import cv2
import numpy as np
import pytest
import torch
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from modelinhos.ssd.anchors import anchors
from modelinhos.ssd.load import (
    load_with_mismatch_from_weights,
)
from modelinhos.ssd.retianent_tv import (
    build_retinanet_torchvision,
    normalize,
    postprocess,
)
from modelinhos.ssd.retinanet import RetinaNetPure


@pytest.fixture
def batch(resolution: tuple[int, int]) -> torch.Tensor:
    return torch.rand(1, 3, *resolution)


@pytest.mark.parametrize(
    "build_model, build_anchors, load_weights",
    [
        (
            RetinaNetPure,
            partial(
                anchors,
                min_sizes=[[16, 32], [64, 128], [256, 512]],
                steps=[8, 16, 32],
                clip=False,
            ),
            partial(
                load_with_mismatch_from_weights,
                weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
                progress=False,
            ),
        )
    ],
)
@pytest.mark.skip()
@pytest.mark.parametrize(
    "resolution",
    [
        (640, 480),
    ],
)
def test_ssd(
    build_model,
    build_anchors,
    load_weights,
    resolution,
    batch,
    n_classes=2,
):
    priors = build_anchors(image_size=resolution[::-1])
    print(priors.shape)
    model = build_model(n_classes=n_classes)
    model = load_weights(model)
    boxes, classes = model(batch)
    assert boxes.shape == (1, *priors.shape)
    assert classes.shape == (1, priors.shape[0], n_classes)


def pad(image: np.ndarray, target_h=800, target_w=1066) -> np.ndarray:
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
def frame(path: str = "tests/assets/person.jpg") -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        pytest.skip(f"Asset not found: {path}")
    return pad(image)


def plot_predictions(image_rgb, predictions, score_threshold=0.5):
    pred = predictions[0]
    boxes = pred["boxes"].numpy()
    scores = pred["scores"].numpy()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    print("here")
    for box, score in zip(boxes, scores):
        if score < score_threshold:
            continue
        print(score)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Predictions", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_weights_match(frame):
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    pure, priors = build_retinanet_torchvision(frame.shape[:2])
    print(priors.shape)
    pure.load_state_dict(weights.get_state_dict())
    preprocess = weights.transforms()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    input_tensor = preprocess(tensor).unsqueeze(0)

    model = retinanet_resnet50_fpn_v2(weights=weights)
    model.eval()
    pure.eval()

    with torch.no_grad():
        predictions = model(input_tensor.clone())
        predictions_tv = postprocess(
            pure(normalize(input_tensor)),
            priors,
            image_size=frame.shape[:2],
        )

    plot_predictions(frame_rgb, predictions)
    plot_predictions(frame_rgb, predictions_tv)
