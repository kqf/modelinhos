"""Named detector presets. Every script/test should build its Detector
through one of these functions -- the raw assembly lives in
modelinhos.detector, the per-family Architecture presets next to their
model definitions in modelinhos.models."""

from dataclasses import replace
from typing import Optional

from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from modelinhos.detector import (
    Architecture,
    Detector,
    TorchvisionDetector,
    build_detector,
)
from modelinhos.models.retinanet import RETINANET, TORCHVISION_RETINANET
from modelinhos.models.ssdlite import SSDLITE, TORCHVISION_SSDLITE
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.trainer.simple import TrainConfig


def _torchvision_label_encoder(
    weights,
    resolution: Optional[tuple[int, int]] = None,
) -> LabelEncoder:
    labels = weights.meta["categories"]
    return LabelEncoder(
        l2i={label: i for i, label in enumerate(labels)},
        resolution=resolution,
    )


def build_inference_only_ssd(
    weights,
    resolution: tuple[int, int],
    lencoder: Optional[LabelEncoder] = None,
) -> Detector:
    """Reference torchvision-native SSDLite, used as a comparison baseline."""
    return TorchvisionDetector(
        build_model=ssdlite320_mobilenet_v3_large,
        resolution=resolution,
        weights=weights,
        lencoder=lencoder or _torchvision_label_encoder(weights),
    )


def build_inference_only_retina(
    weights,
    resolution: tuple[int, int],
    lencoder: Optional[LabelEncoder] = None,
) -> Detector:
    """Reference torchvision-native RetinaNet, used as a comparison
    baseline."""
    return TorchvisionDetector(
        build_model=retinanet_resnet50_fpn_v2,
        resolution=resolution,
        weights=weights,
        lencoder=lencoder or _torchvision_label_encoder(weights),
    )


def build_inference_only_custom_ssd(
    weights,
    resolution: tuple[int, int],
    arch: Architecture = TORCHVISION_SSDLITE,
    n_classes: int = 91,
    th: float = 0.4,
    lencoder: Optional[LabelEncoder] = None,
) -> Detector:
    """Our SSD reimplementation, loaded with (possibly mismatched)
    pretrained weights, inference only."""
    return build_detector(
        replace(arch, weights=weights),
        lencoder=lencoder
        or _torchvision_label_encoder(weights, resolution=resolution),
        resolution=resolution,
        th=th,
        n_classes=n_classes,
    )


def build_inference_only_custom_retina(
    weights,
    resolution: tuple[int, int],
    arch: Architecture = TORCHVISION_RETINANET,
    n_classes: int = 91,
    th: float = 0.4,
    lencoder: Optional[LabelEncoder] = None,
) -> Detector:
    """Our RetinaNet reimplementation, loaded with (possibly mismatched)
    pretrained weights, inference only (decode-only loss, matching how the
    real pretrained weights were trained)."""
    return build_detector(
        replace(arch, weights=weights),
        lencoder=lencoder
        or _torchvision_label_encoder(weights, resolution=resolution),
        resolution=resolution,
        th=th,
        n_classes=n_classes,
    )


def build_trainable_ssd(
    resolution: tuple[int, int],
    lencoder: Optional[LabelEncoder] = None,
    epochs: int = 10,
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
) -> Detector:
    """SSD detector configured for training: lencoder.l2i must already be
    fit (it decides the classification head size and, conventionally,
    reserves index 0 for background via l2i_forced)."""
    return build_detector(
        replace(SSDLITE, weights=weights),
        lencoder=lencoder
        or LabelEncoder(
            l2i={"__background__": 0, "dot": 1},
            resolution=resolution,
        ),
        resolution=resolution,
        train=TrainConfig(epochs=epochs),
    )


def build_trainable_retina(
    resolution: tuple[int, int],
    lencoder: Optional[LabelEncoder] = None,
    epochs: int = 10,
    weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
) -> Detector:
    """RetinaNet detector configured for training: lencoder.l2i must
    already be fit (it decides the classification head size and,
    conventionally, reserves index 0 for background via l2i_forced). The
    loss (build_ret_loss) is the sigmoid focal loss the pretrained
    checkpoint was itself trained under, so the warm start is
    convention-exact: the head starts quiet (prior-probability biases)
    instead of firing everywhere. Pass weights=None to train from
    scratch."""
    return build_detector(
        replace(RETINANET, weights=weights),
        lencoder=lencoder
        or LabelEncoder(
            l2i={"__background__": 0, "dot": 1},
            resolution=resolution,
        ),
        resolution=resolution,
        train=TrainConfig(epochs=epochs),
    )
