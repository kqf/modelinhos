"""Named detector presets. Every script/test should build its Detector
through one of these functions -- the raw assembly lives in
modelinhos.detector, the per-family DetectionRecipe presets next to their
model definitions in modelinhos.models.

Every detector built here is trainable; the only axis of variation is the
recipe. Torchvision-native reference predictions (the parity baseline)
come from recipe.reference -- a plain predict function, not a Detector."""

from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)

from modelinhos.detector import (
    DetectionRecipe,
    Detector,
    EngineBuilder,
    build_detector,
)
from modelinhos.engine.simple import simple_engine
from modelinhos.models.retinanet import TORCHVISION_RETINANET
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.preprocess.lables import LabelEncoder


def coco_label_encoder(
    weights,
    resolution: tuple[int, int],
) -> LabelEncoder:
    """Label encoder over the checkpoint's own COCO categories -- pairs
    with build_ssd/build_retina to reproduce the pretrained detector
    exactly (91 channels, so the mismatch-tolerant load is a passthrough).
    Pass the image resolution the detector is built for."""
    labels = weights.meta["categories"]
    return LabelEncoder(
        resolution=resolution,
        l2i={label: i for i, label in enumerate(labels)},
    )


def build_ssd(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    arch: DetectionRecipe = TORCHVISION_SSDLITE,
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
    engine: EngineBuilder = simple_engine(epochs=10),
    th: float = 0.4,
) -> Detector:
    """Our SSD reimplementation. lencoder must be fit: the classification
    head is sized from lencoder.n_classes at construction time, with
    index 0 reserved for background (LabelEncoder enforces both). Weights
    load mismatch-tolerantly, so any head size warm-starts from the
    checkpoint; with coco_label_encoder the defaults reproduce the
    torchvision pretrained detector exactly. Pass arch=SSDLITE for the
    trimmed custom flavor, weights=None to start from scratch, and any
    engine (skorch_engine, lightning_engine, ...) to swap the training
    backend."""
    return build_detector(
        arch=arch,
        weights=weights,
        lencoder=lencoder,
        resolution=resolution,
        th=th,
        engine=engine,
    )


def build_retina(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    arch: DetectionRecipe = TORCHVISION_RETINANET,
    weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
    engine: EngineBuilder = simple_engine(epochs=10),
    th: float = 0.4,
) -> Detector:
    """Our RetinaNet reimplementation. Same contract as build_ssd:
    lencoder must be fit (it sizes the head), weights load
    mismatch-tolerantly, and with coco_label_encoder the defaults
    reproduce the torchvision pretrained detector exactly. The loss
    (build_ret_loss) is the sigmoid focal loss the pretrained checkpoint
    was itself trained under, so the warm start is convention-exact: the
    head starts quiet (prior-probability biases) instead of firing
    everywhere. Pass arch=RETINANET for the trimmed custom flavor,
    weights=None to start from scratch."""
    return build_detector(
        arch=arch,
        weights=weights,
        lencoder=lencoder,
        resolution=resolution,
        th=th,
        engine=engine,
    )
