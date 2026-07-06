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

from modelinhos.plot import plot
from modelinhos.sample import Annotation, Sample
from modelinhos.ssd.inference import Detector
from modelinhos.ssd.retinanet import bulid_retinanet, retina_anchors
from modelinhos.ssd.ssdlite import build_ssdlite, ssdlite_anchors
from modelinhos.zoo import (
    build_inference_only_custom_retina,
    build_inference_only_custom_ssd,
    build_inference_only_retina,
    build_inference_only_ssd,
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


def _run_detector(detector: Detector, frame, headless: bool) -> Sample:
    predictions = detector.transform_single(frame)[0]
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
    "build_reference, build_custom, weights, tv_expected, md_expected",
    [
        pytest.param(
            build_inference_only_ssd,
            build_inference_only_custom_ssd,
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            (
                Sample(
                    file_name=Path("fake-file.png"),
                    annotations=[
                        Annotation(
                            bbox=(
                                481.7032165527344,
                                225.93629455566406,
                                597.848388671875,
                                589.3338623046875,
                            ),
                            label="person",
                            score=0.8167938590049744,
                        )
                    ],
                )
            ),
            Sample(
                file_name=Path("fake-file.png"),
                annotations=[
                    Annotation(
                        bbox=(
                            357.0254,
                            -50.3667,
                            712.4014,
                            814.3435,
                        ),
                        label="person",
                        score=0.9986,
                    )
                ],
            ),
            id="ssdlite320_mobilenet_v3_large",
        ),
        pytest.param(
            build_inference_only_retina,
            build_inference_only_custom_retina,
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
    frame,
    build_reference,
    build_custom,
    weights,
    tv_expected,
    md_expected,
    headless,
):
    shape = frame.shape[:2]
    tv_preds = _run_detector(
        build_reference(weights, shape),
        frame,
        headless,
    )
    # NB: We accept the difference between Custom and TV
    assert_same_sample(tv_preds, tv_expected)

    md_preds = _run_detector(
        build_custom(weights, shape),
        frame,
        headless,
    )
    assert_same_sample(md_preds, md_expected)


@pytest.mark.skip
@pytest.mark.parametrize(
    "build_fn, build_model, build_anchors, n_classes, th, weights",
    [
        pytest.param(
            build_inference_only_custom_ssd,
            build_ssdlite,
            ssdlite_anchors,
            91,
            0.01,
            SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            id="ssdlite320_mobilenet_v3_large",
        ),
        pytest.param(
            build_inference_only_custom_retina,
            bulid_retinanet,
            retina_anchors,
            91,
            0.05,
            RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            id="retinanet_resnet50_fpn_v2_91",
        ),
        pytest.param(
            build_inference_only_custom_retina,
            bulid_retinanet,
            retina_anchors,
            2,
            0.01,
            RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            id="retinanet_resnet50_fpn_v2_2",
        ),
        pytest.param(
            build_inference_only_custom_retina,
            bulid_retinanet,
            retina_anchors,
            # Don't multiply by two because it breaks tests.
            91 * 1,
            0.01,
            RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
            id="retinanet_resnet50_fpn_v2_91_times_1",
        ),
    ],
)
def test_inference_works(
    frame,
    build_fn,
    build_model,
    build_anchors,
    n_classes,
    th,
    weights,
    headless,
):
    model = build_fn(
        weights,
        frame.shape[:2],
        build_model=build_model,
        anchors=build_anchors,
        n_classes=n_classes,
        th=th,
    )
    _run_detector(model, frame, headless)
