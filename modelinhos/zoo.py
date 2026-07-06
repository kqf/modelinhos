"""Detector builders. `Detector`/`TorchvisionDetector` should only ever be
instantiated here — every script/test should build one through one of
these named functions."""

from typing import Optional

from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2

from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.ssd.inference import (
    DetectionConfig,
    Detector,
    TorchvisionDetector,
)
from modelinhos.ssd.retinanet import (
    build_ret_loss,
    build_torchvision_retinanet,
    build_trainable_retina_loss,
    bulid_retinanet,
    retina_anchors,
    torchvision_retina_anchors,
)
from modelinhos.ssd.ssdlite import (
    build_ssd_loss,
    build_ssdlite,
    build_torchvision_ssdlite,
    ssd_normalize,
    ssdlite_anchors,
    torchvision_ssdlite_anchors,
)


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
    build_model=build_torchvision_ssdlite,
    anchors=torchvision_ssdlite_anchors,
    n_classes: int = 91,
    th: float = 0.4,
    lencoder: Optional[LabelEncoder] = None,
) -> Detector:
    """Our SSD reimplementation, loaded with (possibly mismatched) pretrained
    weights, inference only. build_model/anchors must be a matching pair
    (see ssd/ssdlite.py)."""
    return DetectionConfig(
        build_model=lambda weights, resolution: build_model(
            weights=weights, resolution=resolution, n_classes=n_classes
        ),
        anchors=anchors,
        resolution=resolution,
        weights=weights,
        lencoder=lencoder
        or _torchvision_label_encoder(weights, resolution=resolution),
        loss=build_ssd_loss,
        normalize=ssd_normalize,
        th=th,
    ).build()


def build_inference_only_custom_retina(
    weights,
    resolution: tuple[int, int],
    build_model=build_torchvision_retinanet,
    anchors=torchvision_retina_anchors,
    n_classes: int = 91,
    th: float = 0.4,
    lencoder: Optional[LabelEncoder] = None,
) -> Detector:
    """Our RetinaNet reimplementation, loaded with (possibly mismatched)
    pretrained weights, inference only (decode-only loss, matching how the
    real pretrained weights were trained). build_model/anchors must be a
    matching pair (see ssd/retinanet.py)."""
    return DetectionConfig(
        build_model=lambda weights, resolution: build_model(
            weights=weights, resolution=resolution, n_classes=n_classes
        ),
        anchors=anchors,
        resolution=resolution,
        weights=weights,
        lencoder=lencoder
        or _torchvision_label_encoder(weights, resolution=resolution),
        loss=build_ret_loss,
        th=th,
    ).build()


def build_trainable_ssd(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    epochs: int = 10,
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
) -> Detector:
    """SSD detector configured for training: lencoder.l2i must already be
    fit (it decides the classification head size and, conventionally,
    reserves index 0 for background via l2i_forced)."""
    n_classes = len(lencoder.l2i)
    return DetectionConfig(
        build_model=lambda weights, resolution: build_ssdlite(
            weights=weights,
            resolution=resolution,
            n_classes=n_classes,
        ),
        anchors=ssdlite_anchors,
        resolution=resolution,
        weights=weights,
        lencoder=lencoder,
        loss=build_ssd_loss,
        normalize=ssd_normalize,
        th=0.4,
        epochs=epochs,
    ).build()


def build_trainable_retina(
    resolution: tuple[int, int],
    lencoder: LabelEncoder,
    epochs: int = 10,
    weights=None,
) -> Detector:
    """RetinaNet detector configured for training: lencoder.l2i must
    already be fit (it decides the classification head size and,
    conventionally, reserves index 0 for background via l2i_forced). Uses
    build_trainable_retina_loss, not build_ret_loss -- the latter is
    decode-only. Defaults to no pretrained weights, since the pretrained
    checkpoint was trained under a different (sigmoid) convention than
    build_trainable_retina_loss's softmax one."""
    n_classes = len(lencoder.l2i)
    return DetectionConfig(
        build_model=lambda weights, resolution: bulid_retinanet(
            weights=weights,
            resolution=resolution,
            n_classes=n_classes,
        ),
        anchors=retina_anchors,
        resolution=resolution,
        weights=weights,
        lencoder=lencoder,
        loss=build_trainable_retina_loss,
        th=0.4,
        epochs=epochs,
    ).build()
