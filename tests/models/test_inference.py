from pathlib import Path

import cv2
import numpy as np
import pytest
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)

from modelinhos.models.retinanet import TORCHVISION_RETINANET
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.plot import plot
from modelinhos.sample import Annotation, Sample
from modelinhos.zoo import build_retina, build_ssd, coco_label_encoder


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


def _show(frame, predictions: Sample, headless: bool) -> Sample:
    annotated = plot(frame, predictions)
    # sourcery skip: no-conditionals-in-tests
    if not headless:
        cv2.imshow("Predictions", annotated)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    return predictions


def assert_same_sample(preds, expect):
    assert preds.file_name == expect.file_name
    assert len(preds.annotations) == len(expect.annotations)

    for tv, md in zip(preds.annotations, expect.annotations):
        assert tv.label == md.label
        assert tv.score == pytest.approx(md.score, 1e-4)
        for x1, x2 in zip(tv.bbox, md.bbox):
            assert x1 == pytest.approx(x2, abs=0.01)


@pytest.mark.parametrize(
    "resolution, arch, build_custom, weights, tv_expected, md_expected",
    [
        pytest.param(
            (320, 320),
            TORCHVISION_SSDLITE,
            build_ssd,
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            Sample(
                file_name=Path("fake-file.png"),
                annotations=[
                    Annotation(
                        bbox=(
                            125.99763488769531,
                            56.866600036621094,
                            196.1458282470703,
                            264.6971130371094,
                        ),
                        label="person",
                        score=0.9418545961380005,
                    )
                ],
            ),
            Sample(
                file_name=Path("fake-file.png"),
                annotations=[
                    Annotation(
                        bbox=(
                            125.92370986938477,
                            54.23077583312988,
                            195.98981857299805,
                            263.4918975830078,
                        ),
                        label="person",
                        score=0.8907293081283569,
                    )
                ],
            ),
            id="ssdlite320_mobilenet_v3_large",
        ),
        pytest.param(
            # This is magic resolution to avoid additional geometric conversion
            (
                800,
                1088,
            ),
            TORCHVISION_RETINANET,
            build_retina,
            RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            Sample(
                file_name=Path("fake-file.png"),
                annotations=[
                    Annotation(
                        bbox=(
                            488.3135681152344,
                            227.34669494628906,
                            597.8914184570312,
                            575.18212890625,
                        ),
                        label="person",
                        score=0.9937841892242432,
                    ),
                    Annotation(
                        bbox=(
                            531.1905517578125,
                            289.6827392578125,
                            542.9974365234375,
                            329.750732421875,
                        ),
                        label="tie",
                        score=0.6264503002166748,
                    ),
                ],
            ),
            Sample(
                file_name=Path("fake-file.png"),
                annotations=[
                    Annotation(
                        bbox=(
                            491.05938720703125,
                            230.8584747314453,
                            593.6303100585938,
                            572.8419189453125,
                        ),
                        label="person",
                        score=0.9877095222473145,
                    )
                ],
            ),
            id="retinanet_resnet50_fpn_v2",
        ),
    ],
)
def test_weights_match(
    resolution,
    frame,
    arch,
    build_custom,
    weights,
    tv_expected,
    md_expected,
    headless,
):
    shape = frame.shape[:2]
    assert arch.reference is not None
    tv_preds = _show(frame, arch.reference(weights, [frame])[0], headless)
    # NB: We accept the difference between Custom and TV
    assert_same_sample(tv_preds, tv_expected)

    detector = build_custom(
        resolution=shape,
        lencoder=coco_label_encoder(weights, resolution=shape),
        arch=arch,
        weights=weights,
    )
    md_preds = _show(frame, detector.transform_single(frame)[0], headless)
    assert_same_sample(md_preds, md_expected)
