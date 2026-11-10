"""Named detector presets: build_detector with each family's default
(arch, weights) pair pinned. Every builder takes
(resolution, lencoder, arch=..., weights=..., engine=..., th=...); any
default can be overridden by keyword -- e.g. arch=SSDLITE for the trimmed
custom flavor, weights=restore(path) / from_scratch, engine=skorch_engine(...).

lencoder must be fit: it sizes the classification head, index 0 being
background. The default weights are warm_start (mismatch-tolerant), so
any head size warm-starts; paired with coco_label_encoder /
blaze_label_encoder the defaults reproduce the pretrained detector
exactly. Torchvision-native reference predictions (the parity baseline)
come from recipe.reference -- a plain predict function, not a Detector."""

from functools import partial

from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.fcos import FCOS_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)

from modelinhos.detector import build_detector
from modelinhos.engine.simple import simple_engine
from modelinhos.models.blazenet import BLAZEFACE_F, BlazeNet_Weights
from modelinhos.models.fcos import TORCHVISION_FCOS
from modelinhos.models.load import warm_start
from modelinhos.models.retinanet import TORCHVISION_RETINANET
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.preprocess.labels import LabelEncoder


def coco_label_encoder(weights) -> LabelEncoder:
    """Label encoder over the checkpoint's own COCO categories -- pairs
    with build_ssd/build_retina/build_fcos to reproduce the pretrained
    detector exactly (91 channels, so the mismatch-tolerant load is a
    passthrough)."""
    labels = weights.meta["categories"]
    return LabelEncoder(
        l2i={label: i for i, label in enumerate(labels)},
    )


def blaze_label_encoder(weights) -> LabelEncoder:
    """Label encoder over a blaze checkpoint's categories. Unlike the
    COCO checkpoints, blaze meta carries no background slot (the vanilla
    net emits a single sigmoid face channel), so indices start at 1 and
    0 keeps the background convention -- n_classes comes out as 2."""
    labels = weights.meta["categories"]
    return LabelEncoder(
        l2i={
            "__background__": 0,
            **{label: i for i, label in enumerate(labels, start=1)},
        },
    )


_engine = simple_engine(max_epochs=10)

build_ssd = partial(
    build_detector,
    arch=TORCHVISION_SSDLITE,
    weights=warm_start(SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
    engine=_engine,
)

# The RetinaNet and FCOS losses mirror the ones their checkpoints were
# trained under (sigmoid focal; focal + GIoU + centerness), so the warm
# start is convention-exact. FCOS_DELTA puts the FCOS head on RetinaNet's
# anchors/codec -- see models/fcos.py for that A/B.
build_retina = partial(
    build_detector,
    arch=TORCHVISION_RETINANET,
    weights=warm_start(
        RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, progress=False
    ),
    engine=_engine,
)

build_fcos = partial(
    build_detector,
    arch=TORCHVISION_FCOS,
    weights=warm_start(FCOS_ResNet50_FPN_Weights.COCO_V1, progress=False),
    engine=_engine,
)

# Front and back cameras are separate recipes (BLAZEFACE_F/BLAZEFACE_B,
# likewise RETINANET_F/RETINANET_B for the retina-anchored trainable
# extensions); pass the matching arch and weights together.
build_blaze = partial(
    build_detector,
    arch=BLAZEFACE_F,
    weights=warm_start(BlazeNet_Weights.FRONT_V1),
    engine=_engine,
)
