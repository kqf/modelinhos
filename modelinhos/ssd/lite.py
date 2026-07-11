from functools import partial

import torch
import torch.nn.functional as F
from torchvision.models.detection import (
    _utils as det_utils,
)
from torchvision.models.detection.ssdlite import (
    SSDLiteClassificationHead,
    SSDLiteRegressionHead,
    _mobilenet_extractor,
    mobilenet_v3_large,
)

from modelinhos.loss.matching import match
from modelinhos.loss.subloss import Sublosses, WeightedLoss, sum_normalized
from modelinhos.postprocess import DetectionLoss
from modelinhos.preprocess.boxes import encode_boxes
from modelinhos.preprocess.image import normalize
from modelinhos.ssd.anchors import anchors
from modelinhos.ssd.load import load_with_mismatch


class SSDPureHead(torch.nn.Module):
    def __init__(self, out_channels, num_anchors, norm_layer, n_classes):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            num_classes=n_classes,
        )
        self.regression_head = SSDLiteRegressionHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
        )

    def forward(self, features):
        classes = self.classification_head(features)
        boxes = self.regression_head(features)
        return boxes, classes


class SSDPure(torch.nn.Module):
    def __init__(self, resolution, n_classes, num_anchors=2, extra=-3):
        super().__init__()
        self.n_classes = n_classes
        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        backbone = mobilenet_v3_large(
            weights=None,
            progress=True,
            norm_layer=norm_layer,
            reduced_tail=True,
        )
        self.backbone = _mobilenet_extractor(
            backbone,
            6,
            norm_layer,
        )
        out_channels = det_utils.retrieve_out_channels(
            self.backbone,
            resolution,
        )
        num_anchors = [num_anchors for _ in out_channels]
        self.head = SSDPureHead(
            out_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            n_classes=n_classes,
        )
        self.extra = extra

    def forward(self, images):
        features = self.backbone(images.float())
        features = list(features.values())[: self.extra]
        return self.head(features)


def ssd_anchors(resolution: tuple[int, int], backbone) -> torch.Tensor:
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    from torchvision.models.detection.image_list import ImageList

    H, W = resolution

    # Get real feature map sizes from the backbone
    with torch.no_grad():
        dummy = torch.zeros(1, 3, H, W)
        features = list(backbone(dummy).values())

    generator = DefaultBoxGenerator(
        aspect_ratios=[[2, 3]] * 6,
        min_ratio=0.2,
        max_ratio=0.95,
        clip=False,
    )

    dummy_images = ImageList(torch.zeros(1, 3, H, W), [(H, W)])
    dboxes = generator(dummy_images, features)[0]

    scale = torch.tensor([W, H, W, H], dtype=dboxes.dtype)
    boxes = dboxes / scale
    return torch.cat(
        [
            (boxes[:, :2] + boxes[:, 2:]) / 2,
            boxes[:, 2:] - boxes[:, :2],
        ],
        dim=1,
    )


ssd_normalize = partial(
    normalize,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
)


# This configures ssd as it was trained
def build_torchvision_ssdlite(
    n_classes=91,
    resolution: tuple[int, int] = (320, 320),
    weights=None,
):
    model = SSDPure(
        resolution=resolution,
        n_classes=n_classes,
        num_anchors=6,
        extra=None,
    )
    priors = ssd_anchors(resolution, model.backbone)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict())
    model.eval()
    return model, priors


# This configures retina-net like network
def build_ssdlite(
    resolution: tuple[int, int],
    n_classes: int = 92,
    weights=None,
):
    model = SSDPure(resolution, n_classes=n_classes)
    priors = anchors(
        resolution,
        sizes=[[32, 64], [64, 128], [128, 256]],
        steps=[16, 32, 64],
        clip=False,
    )

    if weights is not None:
        model = load_with_mismatch(model, weights.get_state_dict())
    return model, priors


# weights for encoding/decoding box regression targets, same convention
# as torchvision's SSD (see postprocess.ssd_postprocess)
ssd_box_weights = (10.0, 10.0, 5.0, 5.0)


def build_ssd_loss(
    priors: torch.Tensor,
    negpos_ratio: int = 7,
    overlap: float = 0.35,
) -> DetectionLoss:
    sublosses = Sublosses(
        bboxes=WeightedLoss(
            loss=sum_normalized(partial(F.smooth_l1_loss, reduction="sum")),
            enc_true=partial(encode_boxes, weights=ssd_box_weights),
        ),
        # scores are derived from the same logits as labels at decode
        # time, there is nothing to train here
        scores=WeightedLoss(loss=None),
        labels=WeightedLoss(
            loss=sum_normalized(partial(F.cross_entropy, reduction="sum")),
            needs_negatives=True,
        ),
    )
    return DetectionLoss(
        priors=priors,
        sublosses=sublosses,
        match=partial(match, negpos_ratio=negpos_ratio, overalp=overlap),
    )
