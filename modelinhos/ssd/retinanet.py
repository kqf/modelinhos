from functools import partial

import torch
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import (
    LastLevelP6P7,
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
)
from torchvision.models.resnet import ResNet50_Weights, resnet50

from modelinhos.loss.loss import DetectionLoss
from modelinhos.loss.matching import match
from modelinhos.loss.subloss import Sublosses, WeightedLoss, sum_normalized
from modelinhos.preprocess.boxes import decode_boxes, encode_boxes
from modelinhos.ssd.anchors import anchors, tvison_anchors
from modelinhos.ssd.load import load_with_mismatch_from_weights


class RetinaNetPureHead(torch.nn.Module):
    def __init__(self, out_channels, num_anchors, norm_layer, n_classes):
        super().__init__()

        self.classification_head = RetinaNetClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            num_classes=n_classes,
        )
        self.regression_head = RetinaNetRegressionHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
        )

    def forward(self, features):
        classes = self.classification_head(features)
        boxes = self.regression_head(features)
        return boxes, classes


class RetinaNetPure(torch.nn.Module):
    def __init__(self, n_classes, extra_blocks=None, num_anchors=2):
        super().__init__()
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1,
            progress=True,
            norm_layer=torch.nn.BatchNorm2d,
        )
        # skip P2 because it generates too many anchors
        self.backbone = _resnet_fpn_extractor(
            backbone,
            5,
            returned_layers=[2, 3, 4],
            extra_blocks=extra_blocks,
        )

        self.head = RetinaNetPureHead(
            self.backbone.out_channels,
            num_anchors=num_anchors,
            n_classes=n_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )
        self.last_feature = -1 if extra_blocks is None else None

    def forward(self, images):
        features = self.backbone(images.float())
        features = list(features.values())[: self.last_feature]
        return self.head(features)


def retina_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors matching bulid_retinanet -- a pure function of resolution."""
    return anchors(
        resolution,
        sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
    )


def torchvision_retina_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors matching build_torchvision_retinanet -- a pure function of
    resolution (tvison_anchors doesn't need the backbone)."""
    return tvison_anchors(
        resolution=resolution,
        steps=[8, 16, 32, 64, 128],
        aspect_ratios=[0.5, 1.0, 2.0],
        scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
        base_sizes=[32, 64, 128, 256, 512],
    )


def bulid_retinanet(n_classes, resolution: tuple[int, int], weights):
    model = RetinaNetPure(n_classes)
    if weights is not None:
        load_with_mismatch_from_weights(model, weights=weights, progress=False)
    return model


def build_torchvision_retinanet(
    n_classes=91,
    resolution: tuple[int, int] = (800, 1088),
    weights=None,
):
    model = RetinaNetPure(
        n_classes,
        extra_blocks=LastLevelP6P7(2048, 256),
        num_anchors=9,
    )
    if weights is not None:
        model.load_state_dict(weights.get_state_dict())
    return model


# Decode-only: matches how the real pretrained RetinaNet weights were
# actually trained (independent per-class sigmoid, no shared background
# class) -- used for comparing our reimplementation's inference output
# against torchvision's. Not usable for training (every loss is None).
def build_ret_loss(
    priors: torch.Tensor,
    score_thresh: float,
) -> DetectionLoss:
    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        scores, labels = torch.sigmoid(raw_logits).max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels.unsqueeze(-1)

    return DetectionLoss(
        priors=priors,
        sublosses=Sublosses(
            bboxes=WeightedLoss(
                loss=None,
                dec_pred=partial(
                    decode_boxes,
                    priors=priors,
                ),
            ),
            scores=WeightedLoss(
                loss=None,
                dec_pred=lambda x: torch.sigmoid(x)
                .max(dim=-1)[0]
                .unsqueeze(-1),
            ),
            labels=WeightedLoss(
                loss=None,
                dec_pred=decode_labels,
            ),
        ),
    )


# weights for encoding/decoding box regression targets -- same convention
# as SSD_BOX_WEIGHTS in ssd/ssdlite.py, shared by build_trainable_retina_loss
# below between encode (training) and decode (dec_pred).
RETINA_BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)


# Trainable: unlike build_ret_loss, this defines real losses so our own
# RetinaNet reimplementation can actually be trained/overfit. It reuses
# SSD's softmax + reserved-background-class convention rather than true
# per-class sigmoid/focal loss, since the matching/select machinery this
# codebase shares across architectures is built around a single class id
# per anchor, not independent multi-label targets.
def build_trainable_retina_loss(
    priors: torch.Tensor,
    score_thresh: float,
    negpos_ratio: int = 7,
    overlap: float = 0.35,
    box_weights: tuple[float, float, float, float] = RETINA_BOX_WEIGHTS,
) -> DetectionLoss:
    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        probs = torch.softmax(raw_logits, dim=-1)
        probs[..., 0] = 0.0  # exclude background class before taking max
        scores, labels = probs.max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels.unsqueeze(-1)

    def decode_scores(raw_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(raw_logits, dim=-1)
        probs[..., 0] = 0.0
        return probs.max(dim=-1)[0].unsqueeze(-1)

    sublosses = Sublosses(
        bboxes=WeightedLoss(
            loss=sum_normalized(partial(F.smooth_l1_loss, reduction="sum")),
            enc_true=partial(encode_boxes, weights=box_weights),
            dec_pred=partial(decode_boxes, priors=priors, weights=box_weights),
        ),
        scores=WeightedLoss(loss=None, dec_pred=decode_scores),
        labels=WeightedLoss(
            loss=sum_normalized(partial(F.cross_entropy, reduction="sum")),
            needs_negatives=True,
            dec_pred=decode_labels,
        ),
    )
    return DetectionLoss(
        priors=priors,
        sublosses=sublosses,
        match=partial(match, negpos_ratio=negpos_ratio, overalp=overlap),
    )
