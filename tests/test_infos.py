import pathlib

import pytest

from modelinhos.infos import anchor_advice, matchability, summarize
from modelinhos.models.retinanet import RETINANET, TORCHVISION_RETINANET
from modelinhos.models.ssdlite import SSDLITE, TORCHVISION_SSDLITE
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import Annotation, Sample


@pytest.mark.parametrize(
    "recipe",
    [
        SSDLITE,
        TORCHVISION_SSDLITE,
        RETINANET,
        TORCHVISION_RETINANET,
    ],
)
def test_summarize(recipe, resolution=(640, 480), n_classes=4):
    print()
    print(
        summarize(
            recipe,
            (1, 3, *resolution),
            n_classes=n_classes,
            warmup=1,
            repeats=3,
        ).to_string()
    )


@pytest.mark.parametrize(
    "recipe",
    [
        SSDLITE,
        TORCHVISION_SSDLITE,
        RETINANET,
        TORCHVISION_RETINANET,
    ],
)
def test_matchability(
    recipe,
    resolution=(640, 480),
    # ~111px scale at 640x480: inside every recipe's anchor bracket
    person=(0.4, 0.4, 0.6, 0.6),
    # ~2px scale: far below any anchor floor, matchable by nothing
    tie=(0.5, 0.5, 0.503, 0.503),
):
    samples = [
        Sample(
            file_name=pathlib.Path("a.png"),
            annotations=[
                Annotation(bbox=person, label="person"),
                Annotation(bbox=tie, label="tie"),
            ],
        ),
        Sample(file_name=pathlib.Path("b.png"), annotations=[]),
    ]

    matched = matchability(
        samples,
        recipe,
        resolution,
        lencoder=LabelEncoder(
            l2i={"__background__": 0, "person": 1, "tie": 2}
        ),
    )
    print()
    print(matched.to_string())
    print()
    print(anchor_advice(matched, recipe, resolution).to_string())

    counts = matched.set_index("label").matched
    assert counts["person"] > 0
    assert counts["tie"] == 0
    # Some background got collected for the image with a positive --
    # exact counts are the matcher's business (mining vs all-unmatched)
    assert (matched.negatives > 0).all()
