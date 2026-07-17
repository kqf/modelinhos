import cv2
import numpy as np
import pytest
import torch

from modelinhos.blaze.infer import EXPECTED
from modelinhos.blaze.postprocessing import predict_on_image
from modelinhos.models.blazenet import (
    BlazeNet_Weights,
    blaze_anchors,
    blaze_back_anchors,
    build_blazenet,
    download_blaze_asset,
    load_repo_anchors,
)


@pytest.mark.parametrize(
    "weights, build_anchors",
    [
        (BlazeNet_Weights.FRONT_V1, blaze_anchors),
        (BlazeNet_Weights.BACK_V1, blaze_back_anchors),
    ],
)
def test_anchors_match_repo(weights, build_anchors):
    expected = load_repo_anchors(weights)
    actual = build_anchors(weights.meta["resolution"])
    np.testing.assert_almost_equal(
        actual.cpu().numpy(),
        expected.cpu().numpy(),
        decimal=6,
    )


@pytest.mark.parametrize(
    "back_model, build_anchors, resolution",
    [
        (False, blaze_anchors, (128, 128)),
        (False, blaze_anchors, (480, 640)),
        (True, blaze_back_anchors, (256, 256)),
    ],
)
def test_one_prediction_per_anchor(back_model, build_anchors, resolution):
    model = build_blazenet(back_model=back_model)
    priors = build_anchors(resolution)
    boxes, classes = model(torch.rand(1, 3, *resolution))
    assert boxes.shape == (1, priors.shape[0], 16)
    assert classes.shape == (1, priors.shape[0], 1)


@pytest.fixture
def face() -> np.ndarray:
    image = cv2.imread(str(download_blaze_asset("1face.png")))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def test_vanilla_blazenet_matches_repo(face):
    weights = BlazeNet_Weights.FRONT_V1
    model = build_blazenet(weights=weights)
    model.anchors = blaze_anchors(weights.meta["resolution"])
    predictions = predict_on_image(
        model,
        face,
        back_model=False,
        min_suppression_threshold=model.min_suppression_threshold,
        min_score_thresh=model.min_score_thresh,
    )
    np.testing.assert_almost_equal(
        predictions.cpu().numpy(),
        EXPECTED,
        decimal=4,
    )
