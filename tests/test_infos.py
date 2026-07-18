import pytest

from modelinhos.infos import summarize
from modelinhos.models.retinanet import RETINANET, TORCHVISION_RETINANET
from modelinhos.models.ssdlite import SSDLITE, TORCHVISION_SSDLITE


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
        )
    )
