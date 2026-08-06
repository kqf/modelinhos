from functools import partial

import torch
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import (
    LastLevelP6P7,
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.ops import sigmoid_focal_loss

from modelinhos.detector import DetectionRecipe, torchvision_reference
from modelinhos.loss.loss import DetectionLoss
from modelinhos.loss.matching import match_all_negatives
from modelinhos.loss.subloss import (
    Sublosses,
    WeightedLoss,
    positive_normalized,
    sum_normalized,
)
from modelinhos.models.anchors import anchors, tvison_anchors
from modelinhos.preprocess.boxes import decode_boxes, encode_boxes


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


def bulid_retinanet(n_classes, resolution: tuple[int, int]):
    return RetinaNetPure(n_classes)


def build_torchvision_retinanet(
    n_classes=91,
    resolution: tuple[int, int] = (800, 1088),
):
    return RetinaNetPure(
        n_classes,
        extra_blocks=LastLevelP6P7(2048, 256),
        num_anchors=9,
    )


# The single RetinaNet loss, trainable and decode-faithful at once: it
# uses the independent per-class sigmoid + focal loss convention the real
# pretrained weights were trained under (arxiv.org/abs/1708.02002), so the
# same builder serves fine-tuning our reimplementation AND comparing its
# inference against torchvision's. The box codec keeps the (1, 1, 1, 1)
# torchvision weights for the same reason: a warm-started regression head
# emits deltas at the right scale from step 0. There is no hard-negative
# mining -- focal loss runs over every anchor (see match_all_negatives)
# and is normalized by the positive count, like the box loss. Channel 0
# stays reserved for background by the shared l2i convention; its one-hot
# target is always off, so focal loss trains it to silence and decode
# zeroes it before taking the max.
def build_ret_loss(
    priors: torch.Tensor,
    score_thresh: float,
    overlap: float = 0.35,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> DetectionLoss:
    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        probs = torch.sigmoid(raw_logits)
        probs[..., 0] = 0.0  # exclude background class before taking max
        scores, labels = probs.max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels.unsqueeze(-1)

    def decode_scores(raw_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(raw_logits)
        probs[..., 0] = 0.0
        return probs.max(dim=-1)[0].unsqueeze(-1)

    def focal_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # select() hands over one integer class id per anchor (0 for the
        # mined-in negatives); the sigmoid convention wants per-class
        # binary targets, with background rows all-off.
        target = F.one_hot(y_true, num_classes=y_pred.shape[-1])
        target = target.to(y_pred.dtype)
        target[:, 0] = 0.0
        return sigmoid_focal_loss(
            y_pred,
            target,
            alpha=alpha,
            gamma=gamma,
            reduction="sum",
        )

    sublosses = Sublosses(
        bboxes=WeightedLoss(
            loss=sum_normalized(partial(F.smooth_l1_loss, reduction="sum")),
            enc_true=encode_boxes,
            dec_pred=partial(decode_boxes, priors=priors),
        ),
        scores=WeightedLoss(loss=None, dec_pred=decode_scores),
        labels=WeightedLoss(
            loss=positive_normalized(focal_loss),
            needs_negatives=True,
            dec_pred=decode_labels,
        ),
    )
    return DetectionLoss(
        priors=priors,
        sublosses=sublosses,
        match=partial(match_all_negatives, overalp=overlap),
    )


# Trainable configuration (sigmoid focal loss, matching the checkpoint).
RETINANET = DetectionRecipe(
    build_model=bulid_retinanet,
    anchors=retina_anchors,
    loss=build_ret_loss,
)

# Faithful-to-torchvision configuration -- same loss, torchvision's own
# anchors/head shape, for comparing our inference against the reference.
# Trainable like any other flavor (warm_start loading, so the head can
# be sized for any label set).
TORCHVISION_RETINANET = DetectionRecipe(
    build_model=build_torchvision_retinanet,
    anchors=torchvision_retina_anchors,
    loss=build_ret_loss,
    reference=torchvision_reference(retinanet_resnet50_fpn_v2),
)
