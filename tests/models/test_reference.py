from pathlib import Path

import cv2
import numpy as np
import pytest
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.fcos import FCOS_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)

from modelinhos.models.fcos import TORCHVISION_FCOS
from modelinhos.models.load import warm_start
from modelinhos.models.retinanet import TORCHVISION_RETINANET
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.plot import plot
from modelinhos.sample import Annotation, Sample
from modelinhos.zoo import (
    build_fcos,
    build_retina,
    build_ssd,
    coco_label_encoder,
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
    return cv2.resize(pad(image, *resolution), resolution[::-1])


def _show(frame, predictions: Sample, headless: bool) -> Sample:
    # plot() draws in place; the copy keeps the fixture frame pristine
    # for the custom-flavor half of the test.
    annotated = plot(frame.copy(), predictions)
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
                            0.39374256134033203,
                            0.17770814895629883,
                            0.6129556894302368,
                            0.8271784782409668,
                        ),
                        label="person",
                        score=0.9418545961380005,
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
                            0.44881758093833923,
                            0.28418341279029846,
                            0.5495325326919556,
                            0.718977689743042,
                        ),
                        label="person",
                        score=0.9937841892242432,
                    ),
                    Annotation(
                        bbox=(
                            0.4882265329360962,
                            0.3621034026145935,
                            0.49907851219177246,
                            0.4121883511543274,
                        ),
                        label="tie",
                        score=0.6264503002166748,
                    ),
                ],
            ),
        ),
        pytest.param(
            # This is magic resolution to avoid additional geometric conversion
            (
                800,
                1088,
            ),
            TORCHVISION_FCOS,
            build_fcos,
            FCOS_ResNet50_FPN_Weights.COCO_V1,
            Sample(
                file_name=Path("fake-file.png"),
                annotations=[
                    Annotation(
                        bbox=(
                            0.4511823093189913,
                            0.27916501998901366,
                            0.5481687433579389,
                            0.7202278900146485,
                        ),
                        label="person",
                        score=0.9291312098503113,
                    ),
                    Annotation(
                        bbox=(
                            0.4872165567734662,
                            0.3600777053833008,
                            0.4979667102589327,
                            0.42221889495849607,
                        ),
                        label="tie",
                        score=0.7631757259368896,
                    ),
                    Annotation(
                        bbox=(
                            0.476831828846651,
                            0.35910423278808595,
                            0.5004750420065487,
                            0.5312664031982421,
                        ),
                        label="tie",
                        score=0.4490560293197632,
                    ),
                ],
            ),
            Sample(
                file_name=Path("fake-file.png"),
                annotations=[
                    Annotation(
                        bbox=(
                            0.4511823058128357,
                            0.27916499972343445,
                            0.5481687784194946,
                            0.720227837562561,
                        ),
                        label="person",
                        score=0.9291312098503113,
                    ),
                    Annotation(
                        bbox=(
                            0.4872165322303772,
                            0.3600776791572571,
                            0.4979666769504547,
                            0.42221885919570923,
                        ),
                        label="tie",
                        score=0.7631757259368896,
                    ),
                    Annotation(
                        bbox=(
                            0.4768318235874176,
                            0.3591042459011078,
                            0.5004750490188599,
                            0.5312663912773132,
                        ),
                        label="tie",
                        score=0.4490560293197632,
                    ),
                ],
            ),
            id="fcos_resnet50_fpn",
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
        weights=warm_start(weights),
    )
    md_preds = _show(frame, detector.transform_single(frame)[0], headless)
    assert_same_sample(md_preds, md_expected)


@pytest.fixture
def resolution() -> tuple[int, int]:
    # This is magic resolution to avoid additional geometric conversion
    return 800, 1088
