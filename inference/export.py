"""Export every detection recipe to static-shape ONNX for the C++
benchmark in inference/infer.cpp.

Each recipe gets its meaningful resolution ladder up to 1080p: the
native size first, then VGA, 720p and 1080p rounded to the stride-32
grid (720 and 1080 themselves are not divisible by 32 -- the stride-8
feature map goes odd and the stride-2 blocks of every backbone here
break, hence 736x1280 and 1088x1920). Models are built fresh with
randomized weights: the benchmark measures shapes and speed, not
accuracy.

Files land in inference/models/<recipe>/<height>-<width>.onnx; run from
the repo root:  python inference/export.py
"""

import itertools
from pathlib import Path

import onnx
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
            # Torchvision inits leave many byte-identical tensors (zero
            # biases, unit norm scales); torch.onnx deduplicates those
            # into Identity-of-initializer nodes that OpenCV 4.6 cannot
            # parse, and their degenerate values (Mul by 1, Add 0) would
            # let optimizers fold real per-layer work out of the graph.
            # Random positive values keep every tensor unique and every
            # op live, so the benchmark measures the trained-model cost.
            with torch.no_grad():
                for tensor in itertools.chain(
                    model.parameters(),
                    model.buffers(),
                ):
                    if tensor.is_floating_point():
                        tensor.uniform_(0.01, 0.1)
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
            # Heads shared across pyramid levels reuse the same
            # Parameter object; the exporter deduplicates it into one
            # initializer plus Identity aliases, which OpenCV 4.6
            # cannot parse. Materialize each alias as its own
            # initializer and drop the Identity; chains resolve over
            # iterations. Pure aliasing: no compute is removed.
            exported = onnx.load(str(path))
            graph = exported.graph
            while True:
                initializers = {i.name: i for i in graph.initializer}
                aliases = [
                    node
                    for node in graph.node
                    if node.op_type == "Identity"
                    and node.input[0] in initializers
                ]
                if not aliases:
                    break
                for node in aliases:
                    clone = onnx.TensorProto()
                    clone.CopyFrom(initializers[node.input[0]])
                    clone.name = node.output[0]
                    graph.initializer.append(clone)
                    graph.node.remove(node)
            onnx.save(exported, str(path))
            print(f"exported {path}")


if __name__ == "__main__":
    main()
