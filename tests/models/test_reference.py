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
                            0.39374260902404784,
                            0.17770812511444092,
                            0.6129557132720947,
                            0.8271784782409668,
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
                            0.3935115933418274,
                            0.16947117447853088,
                            0.6124681830406189,
                            0.8234121799468994,
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
                            0.4488176177529728,
                            0.2841833686828613,
                            0.549532553728889,
                            0.7189776611328125,
                        ),
                        label="person",
                        score=0.9937841892242432,
                    ),
                    Annotation(
                        bbox=(
                            0.48822661007151885,
                            0.36210342407226564,
                            0.49907852621639476,
                            0.41218841552734375,
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
                            0.4513413608074188,
                            0.28857308626174927,
                            0.545616090297699,
                            0.7160523533821106,
                        ),
                        label="person",
                        score=0.9877095222473145,
                    )
                ],
            ),
        ),
    ],
)
def test_references_match(
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
        lencoder=coco_label_encoder(weights),
        arch=arch,
        weights=weights,
    )
    md_preds = _show(frame, detector.transform_single(frame)[0], headless)
    assert_same_sample(md_preds, md_expected)
