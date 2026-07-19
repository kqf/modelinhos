import cv2
import numpy as np
import pytest
import torch

from modelinhos.models.blazenet import (
    BLAZEFACE,
    RETINANET,
    BlazeNet_Weights,
    blaze_anchors,
    blaze_back_anchors,
    build_blazenet,
    build_retina_blazenet,
    download_blaze_asset,
    load_repo_anchors,
    predict_on_image,
    retina_anchors,
)
from modelinhos.models.load import restore
from modelinhos.zoo import blaze_label_encoder, build_blaze

# Reference output of the original BlazeFace-PyTorch implementation
# (hollance's blazeface.py, weights + anchors.npy from that repo) on its
# 1face.png sample: ymin, xmin, ymax, xmax, 6 keypoints (x, y), score.
EXPECTED = np.array(
    [
        [
            0.27143508195877075,
            0.31713399291038513,
            0.44155359268188477,
            0.48725229501724243,
            0.3863072991371155,
            0.3126678764820099,
            0.46129563450813293,
            0.3186052143573761,
            0.43995076417922974,
            0.355654776096344,
            0.4327559769153595,
            0.3914948105812073,
            0.3153791129589081,
            0.32960063219070435,
            0.4778282344341278,
            0.3367067277431488,
            0.9959920644760132,
        ]
    ],
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


@pytest.fixture
def frame() -> np.ndarray:
    # BGR, like every frame entering the Detector flow.
    return cv2.imread(str(download_blaze_asset("1face.png")))


def test_blazeface_recipe_matches_reference(frame):
    weights = BlazeNet_Weights.FRONT_V1
    assert BLAZEFACE.reference is not None
    reference = BLAZEFACE.reference(weights, [frame], th=0.75)[0]

    detector = build_blaze(
        resolution=weights.meta["resolution"],
        lencoder=blaze_label_encoder(weights),
        weights=restore(weights),
        th=0.75,
    )
    preds = detector.transform_single(frame)[0]

    # NB: we accept the difference between the pipeline and the
    # reference -- hard NMS picks the best-scoring box, the reference
    # blends the overlapping ones (weighted NMS).
    assert len(preds.annotations) == len(reference.annotations) == 1
    ref, pred = reference.annotations[0], preds.annotations[0]
    assert pred.label == ref.label == "face"
    assert pred.score == pytest.approx(ref.score, abs=0.01)
    for ours, theirs in zip(pred.bbox, ref.bbox):
        assert ours == pytest.approx(theirs, abs=0.01)


@pytest.mark.parametrize("resolution", [(128, 128), (128, 160)])
def test_blaze_pure_one_prediction_per_anchor(resolution):
    model = build_retina_blazenet(resolution, n_classes=3)
    priors = retina_anchors(resolution)
    boxes, classes = model(torch.rand(1, 3, *resolution))
    assert boxes.shape == (1, priors.shape[0], 4)
    assert classes.shape == (1, priors.shape[0], 3)


def test_blaze_retinanet_recipe_decodes():
    resolution = (128, 160)
    baked = RETINANET.bake(n_classes=3, resolution=resolution)
    preds = baked.model(torch.rand(2, 3, *resolution))
    decoded = baked.loss.decode(preds)
    n_anchors = retina_anchors(resolution).shape[0]
    assert decoded.bboxes.shape == (2, n_anchors, 4)
    assert decoded.scores.shape == (2, n_anchors, 1)
    assert decoded.labels.shape == (2, n_anchors, 1)
