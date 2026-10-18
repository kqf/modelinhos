import itertools
import warnings
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper

from modelinhos.models.blazenet import (
    BLAZEFACE_B,
    BLAZEFACE_F,
    RETINANET_B as BLAZE_RETINANET_B,
    RETINANET_F as BLAZE_RETINANET,
)
from modelinhos.models.fcos import FCOS_DELTA, TORCHVISION_FCOS
from modelinhos.models.retinanet import RETINANET, TORCHVISION_RETINANET
from modelinhos.models.ssdlite import SSDLARGE, SSDLITE, TORCHVISION_SSDLITE

# Every model is exported at the same resolutions, so the benchmarks
# compare them like for like.
RESOLUTIONS = [
    (128, 128),
    (256, 256),
    (320, 320),
    (480, 640),
    (512, 640),
    (800, 1088),
    (736, 1280),
    (1088, 1920),
]

# recipe -> (preset, n_classes); blazeface is the fixed single-face-channel
# head (n_classes=2), everything else gets the COCO label space (91 with
# background).
CONFIGS = {
    "blazeface": (BLAZEFACE_F, 2),
    "blazeface-back": (BLAZEFACE_B, 2),
    "blaze-retina": (BLAZE_RETINANET, 91),
    "blaze-retina-back": (BLAZE_RETINANET_B, 91),
    "ssdlite": (SSDLITE, 91),
    "ssdlarge": (SSDLARGE, 91),
    "retinanet": (RETINANET, 91),
    "fcos": (FCOS_DELTA, 91),
    "torchvision-ssdlite": (TORCHVISION_SSDLITE, 91),
    "torchvision-retinanet": (TORCHVISION_RETINANET, 91),
    "torchvision-fcos": (TORCHVISION_FCOS, 91),
}


def pool_globally(node, initializers):
    # The exporter only emits opset 18 (a downgrade to 13 fails),
    # where ReduceMean takes its axes as an input. OpenCV 4.6 only
    # reads the attribute form and reduces the channels away as well.
    # The squeeze-excite pools are a spatial mean with keepdims, which
    # is exactly GlobalAveragePool: same compute, parseable everywhere.
    if node.op_type != "ReduceMean" or len(node.input) != 2:
        return
    axes = initializers.get(node.input[1])
    keepdims = all(a.i for a in node.attribute if a.name == "keepdims")
    if axes is None or not keepdims:
        return
    if set(numpy_helper.to_array(axes).tolist()) not in ({2, 3}, {-2, -1}):
        return
    node.op_type = "GlobalAveragePool"
    del node.input[1:]
    del node.attribute[:]


def clip_by_attributes(node, initializers):
    # Same opset story for the ReLU6s: since opset 11 Clip takes its
    # bounds as inputs, while OpenCV 4.6 insists on a single input and
    # reads min/max from attributes. Constant bounds move back into
    # attributes; the clamp itself is unchanged.
    if node.op_type != "Clip" or len(node.input) == 1:
        return
    bounds = dict(zip(("min", "max"), node.input[1:]))
    if any(name and name not in initializers for name in bounds.values()):
        return
    for key, name in bounds.items():
        if name:
            value = float(numpy_helper.to_array(initializers[name]).item())
            node.attribute.append(onnx.helper.make_attribute(key, value))
    del node.input[1:]


def drop_unused_initializers(graph):
    # The rewrites above orphan the tensors they no longer read.
    used = {name for node in graph.node for name in node.input}
    for initializer in graph.initializer:
        if initializer.name not in used:
            graph.initializer.remove(initializer)


def materialize_aliases(model):
    # Heads shared across pyramid levels reuse the same Parameter
    # object;
    # the exporter deduplicates it into one initializer plus
    # Identity aliases, which OpenCV 4.6 cannot parse. Materialize
    # each alias as its own initializer and drop the Identity;
    # chains resolve over iterations.
    # Pure aliasing: no compute is removed.
    graph = model.graph
    while True:
        initializers = {i.name: i for i in graph.initializer}
        aliases = [
            node
            for node in graph.node
            if node.op_type == "Identity" and node.input[0] in initializers
        ]

        if not aliases:
            return

        for node in aliases:
            clone = onnx.TensorProto()
            clone.CopyFrom(initializers[node.input[0]])
            clone.name = node.output[0]
            graph.initializer.append(clone)
            graph.node.remove(node)


def initialize(model):
    # Torchvision inits leave many byte-identical tensors (zero
    # biases, unit norm scales); torch.onnx deduplicates those
    # into Identity-of-initializer nodes that OpenCV 4.6 cannot
    # parse, and their degenerate values (Mul by 1, Add 0) would
    # let optimizers fold real per-layer work out of the graph.
    # Random positive values keep every tensor unique and every
    # op live, so the benchmark measures the trained-model cost.
    tensors = [
        tensor
        for tensor in itertools.chain(model.parameters(), model.buffers())
        if tensor.is_floating_point()
    ]
    # A tensor filled with a single value is exactly the degenerate
    # case above; if there is none, the weights are already usable.
    if not any(torch.all(t == t.flatten()[0]) for t in tensors if t.numel()):
        return

    warnings.warn(
        f"{type(model).__name__} has constant tensors, "
        "initializing the weights with random values",
        stacklevel=2,
    )
    with torch.no_grad():
        for tensor in tensors:
            tensor.uniform_(0.01, 0.1)


def main(models: Path = Path(__file__).parent / "models"):
    for name, (recipe, n_classes) in CONFIGS.items():
        for height, width in RESOLUTIONS:
            model = recipe.build_model(
                resolution=(height, width),
                n_classes=n_classes,
            )
            model.eval()
            initialize(model)
            path = models / name / f"{height}-{width}.onnx"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(
                model,
                (torch.randn(1, 3, height, width),),
                str(path),
                input_names=["image"],
                output_names=["boxes", "classes"],
                opset_version=18,
                do_constant_folding=True,
                external_data=False,
            )

            exported = onnx.load(str(path))
            materialize_aliases(exported)
            graph = exported.graph
            initializers = {i.name: i for i in graph.initializer}
            for node in graph.node:
                pool_globally(node, initializers)
                clip_by_attributes(node, initializers)
            drop_unused_initializers(graph)
            onnx.save(exported, str(path))
            print(f"exported {path}")
            break


if __name__ == "__main__":
    main()
