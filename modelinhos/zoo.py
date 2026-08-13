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
from torchvision.models.detection.fcos import FCOS_ResNet50_FPN_Weights
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
from modelinhos.models.blazenet import BLAZEFACE_F, BlazeNet_Weights
from modelinhos.models.fcos import TORCHVISION_FCOS
from modelinhos.models.load import Weights, warm_start
from modelinhos.models.retinanet import TORCHVISION_RETINANET
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.preprocess.labels import LabelEncoder


def coco_label_encoder(weights) -> LabelEncoder:
    """Label encoder over the checkpoint's own COCO categories -- pairs
    with build_ssd/build_retina to reproduce the pretrained detector
    exactly (91 channels, so the mismatch-tolerant load is a
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


def build_ssd(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    arch: DetectionRecipe = TORCHVISION_SSDLITE,
    weights: Weights = warm_start(
        SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    ),
    engine: EngineBuilder = simple_engine(max_epochs=10),
    th: float = 0.4,
) -> Detector:
    """Our SSD reimplementation. lencoder must be fit: the classification
    head is sized from lencoder.n_classes at construction time, with
    index 0 reserved for background (LabelEncoder enforces both). weights
    is a loader from modelinhos.models.load: warm_start(...) (the
    default) is mismatch-tolerant, so any head size warm-starts from the
    checkpoint -- with coco_label_encoder the defaults reproduce the
    torchvision pretrained detector exactly; restore(path) loads a
    trained checkpoint strictly for evaluation/export; from_scratch
    skips loading. Pass arch=SSDLITE for the trimmed custom flavor and any
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
    weights: Weights = warm_start(
        RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, progress=False
    ),
    engine: EngineBuilder = simple_engine(max_epochs=10),
    th: float = 0.4,
) -> Detector:
    """Our RetinaNet reimplementation. Same contract as build_ssd:
    lencoder must be fit (it sizes the head), weights is a
    warm_start/restore loader (from_scratch to skip loading), and with
    coco_label_encoder the defaults
    reproduce the torchvision pretrained detector exactly. The loss
    (build_ret_loss) is the sigmoid focal loss the pretrained checkpoint
    was itself trained under, so the warm start is convention-exact: the
    head starts quiet (prior-probability biases) instead of firing
    everywhere. Pass arch=RETINANET for the trimmed custom flavor,
    weights=from_scratch to skip loading."""
    return build_detector(
        arch=arch,
        weights=weights,
        lencoder=lencoder,
        resolution=resolution,
        th=th,
        engine=engine,
    )


def build_fcos(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    arch: DetectionRecipe = TORCHVISION_FCOS,
    weights: Weights = warm_start(
        FCOS_ResNet50_FPN_Weights.COCO_V1, progress=False
    ),
    engine: EngineBuilder = simple_engine(max_epochs=10),
    th: float = 0.4,
) -> Detector:
    """Our FCOS reimplementation. Same contract as build_ssd: lencoder
    must be fit (it sizes the head), weights is a warm_start/restore
    loader (from_scratch to skip loading), and with coco_label_encoder
    the defaults reproduce the torchvision pretrained detector exactly.
    The loss mirrors torchvision's own (focal + GIoU + centerness BCE
    with centre-sampling matching), so the warm start is
    convention-exact. Pass arch=FCOS_DELTA for the retina-anchored
    trainable flavor on the standard delta codec -- see models/fcos.py
    for why that pair is a clean RetinaNet A/B."""
    return build_detector(
        arch=arch,
        weights=weights,
        lencoder=lencoder,
        resolution=resolution,
        th=th,
        engine=engine,
    )


def build_blaze(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    arch: DetectionRecipe = BLAZEFACE_F,
    weights: Weights = warm_start(BlazeNet_Weights.FRONT_V1),
    engine: EngineBuilder = simple_engine(max_epochs=10),
    th: float = 0.4,
) -> Detector:
    """Our BlazeFace wiring. Same contract as build_ssd: lencoder must
    be fit (it sizes the head; with blaze_label_encoder the defaults
    reproduce the pretrained face detector at its native 128x128), and
    weights is a warm_start/restore loader. Front and back cameras are
    separate recipes (BLAZEFACE_F/BLAZEFACE_B, likewise RETINANET_F/
    RETINANET_B for the retina-anchored trainable extensions); pass the
    matching arch and weights together, weights=from_scratch to skip
    loading."""
    return build_detector(
        arch=arch,
        weights=weights,
        lencoder=lencoder,
        resolution=resolution,
        th=th,
        engine=engine,
    )
