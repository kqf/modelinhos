"""Export every detection recipe to static-shape ONNX for the C++
benchmark in inference/infer.cpp.

Each recipe gets its meaningful resolution ladder up to 1080p: the
native size first, then VGA, 720p and 1080p rounded to the stride-32
grid (720 and 1080 themselves are not divisible by 32 -- the stride-8
feature map goes odd and the stride-2 blocks of every backbone here
break, hence 736x1280 and 1088x1920). Models are built fresh, weights
untouched: the benchmark measures shapes and speed, not accuracy.

Files land in inference/models/<recipe>/<height>-<width>.onnx; run from
the repo root:  python inference/export.py
"""

from pathlib import Path

import torch

from modelinhos.models.blazenet import (
    BLAZEFACE_F,
    RETINANET_F as BLAZE_RETINANET,
)
from modelinhos.models.retinanet import RETINANET, TORCHVISION_RETINANET
from modelinhos.models.ssdlite import SSDLARGE, SSDLITE, TORCHVISION_SSDLITE

VGA, HD, FHD = (480, 640), (736, 1280), (1088, 1920)

# recipe -> (preset, n_classes, resolutions); blazeface is the fixed
# single-face-channel head (n_classes=2), everything else gets the COCO
# label space (91 with background).
CONFIGS = {
    "blazeface": (
        BLAZEFACE_F,
        2,
        [
            (128, 128),
            (256, 256),
            VGA,
            HD,
            FHD,
        ],
    ),
    "blaze-retina": (
        BLAZE_RETINANET,
        91,
        [
            (256, 256),
            (320, 320),
            VGA,
            HD,
            FHD,
        ],
    ),
    "ssdlite": (
        SSDLITE,
        91,
        [
            (320, 320),
            VGA,
            HD,
            FHD,
        ],
    ),
    "ssdlarge": (
        SSDLARGE,
        91,
        [
            (320, 320),
            VGA,
            HD,
            FHD,
        ],
    ),
    "retinanet": (
        RETINANET,
        91,
        [
            (320, 320),
            VGA,
            HD,
            FHD,
        ],
    ),
    "torchvision-ssdlite": (
        TORCHVISION_SSDLITE,
        91,
        [(320, 320), VGA, HD, FHD],
    ),
    "torchvision-retinanet": (
        TORCHVISION_RETINANET,
        91,
        [
            VGA,
            (800, 1088),
            HD,
            FHD,
        ],
    ),
}


def main(models: Path = Path(__file__).parent / "models"):
    for name, (recipe, n_classes, resolutions) in CONFIGS.items():
        for height, width in resolutions:
            model = recipe.build_model(
                resolution=(height, width),
                n_classes=n_classes,
            )
            model.eval()
            path = models / name / f"{height}-{width}.onnx"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(
                model,
                torch.randn(1, 3, height, width),
                str(path),
                input_names=["image"],
                output_names=["boxes", "classes"],
                opset_version=13,
                do_constant_folding=True,
            )
            print(f"exported {path}")


if __name__ == "__main__":
    main()
