"""Named detector presets. Every script/test should build its Detector
through one of these functions -- the raw assembly lives in
modelinhos.detector, the per-family Architecture presets next to their
model definitions in modelinhos.models.

Every detector built here is trainable; the only axis of variation is the
Architecture. The build_reference_* baselines are the exception: they wrap
torchvision-native models verbatim and are inference-only by construction
(torchvision computes its losses internally)."""

from dataclasses import replace

from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from modelinhos.detector import (
    DetectionRecipe,
    Detector,
    TorchvisionDetector,
    build_detector,
)
from modelinhos.models.retinanet import TORCHVISION_RETINANET
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.trainer.simple import TrainConfig


def coco_label_encoder(
    weights,
    resolution: tuple[int, int],
) -> LabelEncoder:
    """Label encoder over the checkpoint's own COCO categories. Pass the
    image resolution for build_ssd/build_retina (they work in normalized
    box space); the reference builders construct their own pixel-space
    encoder internally."""
    labels = weights.meta["categories"]
    return LabelEncoder(
        resolution=resolution,
        l2i={label: i for i, label in enumerate(labels)},
    )


def build_reference_ssd(
    weights,
    resolution: tuple[int, int],
) -> Detector:
    """Reference torchvision-native SSDLite, used as a comparison baseline.
    resolution=(1, 1) keeps bboxes in pixel space -- torchvision-native
    models work in pixels end to end."""
    return TorchvisionDetector(
        build_model=ssdlite320_mobilenet_v3_large,
        resolution=resolution,
        weights=weights,
        lencoder=coco_label_encoder(weights, resolution=(1, 1)),
    )


def build_reference_retina(
    weights,
    resolution: tuple[int, int],
) -> Detector:
    """Reference torchvision-native RetinaNet, used as a comparison
    baseline. See build_reference_ssd for the (1, 1) resolution."""
    return TorchvisionDetector(
        build_model=retinanet_resnet50_fpn_v2,
        resolution=resolution,
        weights=weights,
        lencoder=coco_label_encoder(weights, resolution=(1, 1)),
    )


def build_ssd(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    arch: DetectionRecipe = TORCHVISION_SSDLITE,
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
    epochs: int = 10,
    th: float = 0.4,
) -> Detector:
    """Our SSD reimplementation. lencoder must be fit: the classification
    head is sized from len(l2i) at construction time, with index 0
    reserved for background (LabelEncoder enforces both). Weights load
    mismatch-tolerantly, so any head size warm-starts from the checkpoint;
    with coco_label_encoder the defaults reproduce the torchvision
    pretrained detector exactly. Pass arch=SSDLITE for the trimmed custom
    flavor, weights=None to start from scratch."""
    return build_detector(
        replace(arch, weights=weights),
        lencoder=lencoder,
        resolution=resolution,
        th=th,
        train=TrainConfig(epochs=epochs),
    )


def build_retina(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    arch: DetectionRecipe = TORCHVISION_RETINANET,
    weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
    epochs: int = 10,
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
        replace(arch, weights=weights),
        lencoder=lencoder,
        resolution=resolution,
        th=th,
        train=TrainConfig(epochs=epochs),
    )
