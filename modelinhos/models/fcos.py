from functools import partial

import torch
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.fcos import (
    FCOSClassificationHead,
    FCOSRegressionHead,
    fcos_resnet50_fpn,
)
from torchvision.models.detection.retinanet import LastLevelP6P7
from torchvision.models.resnet import ResNet50_Weights, resnet50
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from modelinhos.detector import DetectionRecipe, torchvision_reference
from modelinhos.loss.loss import DetectionLoss
from modelinhos.loss.matching import match_all_negatives
from modelinhos.loss.subloss import (
    Sublosses,
    WeightedLoss,
    positive_normalized,
    sum_normalized,
)
from modelinhos.models.anchors import tvison_anchors
from modelinhos.models.retinanet import retina_anchors
from modelinhos.preprocess.boxes import decode_boxes, encode_boxes


class SignedFCOSRegressionHead(FCOSRegressionHead):
    """The parent ReLUs its box output because the FCOS linear codec
    predicts non-negative ltrb distances. The standard delta codec
    (encode_boxes) is signed -- centre offsets and log-scales must be
    able to go negative -- so this variant emits the raw convolution."""

    def forward(self, x):
        all_bbox_regression = []
        all_bbox_ctrness = []

        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_feature)
            bbox_ctrness = self.bbox_ctrness(bbox_feature)

            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)
            all_bbox_regression.append(bbox_regression)

            bbox_ctrness = bbox_ctrness.view(N, -1, 1, H, W)
            bbox_ctrness = bbox_ctrness.permute(0, 3, 4, 1, 2)
            bbox_ctrness = bbox_ctrness.reshape(N, -1, 1)
            all_bbox_ctrness.append(bbox_ctrness)

        return (
            torch.cat(all_bbox_regression, dim=1),
            torch.cat(all_bbox_ctrness, dim=1),
        )


class FCOSPureHead(torch.nn.Module):
    """Submodules named like torchvision's FCOSHead (classification_head
    / regression_head) so the pretrained state dict addresses ours
    directly. The centerness logit rides as one extra channel on the
    class logits: to_preds duplicates that concat into both the scores
    and labels slots of PerBatchEncoded, so each field's decode has
    everything the sqrt(cls * ctrness) score needs (see
    build_fcos_loss)."""

    def __init__(self, out_channels, num_anchors, n_classes, signed_boxes):
        super().__init__()
        self.classification_head = FCOSClassificationHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            num_classes=n_classes,
        )
        regression_head = (
            SignedFCOSRegressionHead if signed_boxes else FCOSRegressionHead
        )
        self.regression_head = regression_head(
            in_channels=out_channels,
            num_anchors=num_anchors,
        )

    def forward(self, features):
        classes = self.classification_head(features)
        boxes, ctrness = self.regression_head(features)
        return boxes, torch.cat([classes, ctrness], dim=-1)


class FCOSPure(torch.nn.Module):
    def __init__(
        self,
        n_classes,
        extra_blocks=None,
        num_anchors=2,
        signed_boxes=True,
    ):
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

        self.head = FCOSPureHead(
            self.backbone.out_channels,
            num_anchors=num_anchors,
            n_classes=n_classes,
            signed_boxes=signed_boxes,
        )
        self.last_feature = -1 if extra_blocks is None else None

    def forward(self, images):
        features = self.backbone(images.float())
        features = list(features.values())[: self.last_feature]
        return self.head(features)


def fcos_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors matching build_torchvision_fcos -- one square anchor per
    location whose side equals the stride, torchvision FCOS's own
    degenerate grid. The anchor-free points keep a nominal box so the
    prior-based machinery can read the stride off pw/ph: the codec
    normalization and the matcher's scale ranges all measure in units
    of the prior size."""
    return tvison_anchors(
        resolution=resolution,
        base_sizes=[8, 16, 32, 64, 128],
        steps=[8, 16, 32, 64, 128],
        aspect_ratios=[1.0],
        scales=[1.0],
    )


def encode_ltrb(
    boxes: torch.Tensor,  # (..., 4) normalised xyxy ground-truth boxes
    priors: torch.Tensor,  # (..., 4) normalised cxcywh anchors
) -> torch.Tensor:  # (..., 4) (l, t, r, b) in units of the prior size
    """torchvision's BoxLinearCoder(normalize_by_size=True) in prior
    space: distances from the prior centre to the box edges. In
    normalised coordinates pw = stride/W and ph = stride/H, so dividing
    l, r by pw and t, b by ph reproduces the pixel-space division by
    the (square) anchor side exactly."""
    pcx, pcy, pw, ph = priors.to(boxes).unbind(-1)
    gx1, gy1, gx2, gy2 = boxes.unbind(-1)
    return torch.stack(
        (
            (pcx - gx1) / pw,
            (pcy - gy1) / ph,
            (gx2 - pcx) / pw,
            (gy2 - pcy) / ph,
        ),
        dim=-1,
    )


def decode_ltrb(
    rel_codes: torch.Tensor,  # (B, N, 4) or (N, 4)
    priors: torch.Tensor,  # (N, 4) normalised cxcywh
) -> torch.Tensor:  # normalised xyxy, same leading shape as rel_codes
    """Inverse of encode_ltrb; broadcasts batched codes against flat
    priors like decode_boxes does."""
    pcx, pcy, pw, ph = priors.to(rel_codes).unbind(-1)
    left, top, right, bottom = rel_codes.unbind(-1)
    return torch.stack(
        (
            pcx - left * pw,
            pcy - top * ph,
            pcx + right * pw,
            pcy + bottom * ph,
        ),
        dim=-1,
    )


def centerness(boxes: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    """FCOS centerness target of a matched (GT box, prior) pair:
    sqrt(min/max(l, r) * min/max(t, b)) -- 1 at the box centre, falling
    to 0 at its edge. The ratios are per-axis, so the value is identical
    in normalised and pixel coordinates. Clamped at 0 for locations
    outside their box, which IoU matching (the FCOS_DELTA flavor) can
    produce -- the native matcher only ever assigns inside locations."""
    codes = encode_ltrb(boxes, priors)
    left_right = codes[..., [0, 2]]
    top_bottom = codes[..., [1, 3]]
    ratios = (
        left_right.min(dim=-1).values / left_right.max(dim=-1).values
    ) * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values)
    return torch.sqrt(ratios.clamp(min=0))


def fcos_boxes(
    boxes: torch.Tensor,  # [n_obj, 4] normalised xyxy
    priors: torch.Tensor,  # [n_anchors, 4] normalised cxcywh
    radius: float = 1.5,
    lower: float = 4.0,
    upper: float = 8.0,
) -> torch.Tensor:  # [n_anchors, n_obj] ~
    """FCOS assignment (torchvision FCOS.compute_loss) in prior space: a
    location is positive for a GT when its centre lies within radius
    prior-sizes of the GT centre, sits inside the GT box, and the
    largest ltrb distance falls in its level's scale range ((lower,
    upper) prior-sizes; open below the first level and above the last).
    A location claimed by several GTs goes to the smallest one. Levels
    are recovered from the priors themselves -- consecutive runs of
    equal pw -- and padding boxes ([-1] * 4 after collate) are
    degenerate, so the inside test discards them."""
    n_anchors = priors.shape[0]
    n_obj = boxes.shape[0]

    # Same guard as match_boxes: nothing to match on annotation-less
    # (possibly mis-shaped (0, 1)) box tensors.
    if n_obj == 0:
        return torch.zeros(
            (n_anchors, 0), dtype=torch.bool, device=boxes.device
        )

    pcx, pcy, pw, ph = priors.to(boxes).unbind(-1)
    gx1, gy1, gx2, gy2 = boxes.unbind(-1)

    # [n_anchors, n_obj] ltrb distances in units of the prior size
    dist = torch.stack(
        (
            (pcx[:, None] - gx1[None]) / pw[:, None],
            (pcy[:, None] - gy1[None]) / ph[:, None],
            (gx2[None] - pcx[:, None]) / pw[:, None],
            (gy2[None] - pcy[:, None]) / ph[:, None],
        ),
        dim=-1,
    )

    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2
    positive = (
        torch.maximum(
            (pcx[:, None] - gcx[None]).abs() / pw[:, None],
            (pcy[:, None] - gcy[None]).abs() / ph[:, None],
        )
        < radius
    )
    positive &= dist.min(dim=-1).values > 0

    _, counts = torch.unique_consecutive(pw, return_counts=True)
    lower_bound = torch.full_like(pw, lower)
    lower_bound[: int(counts[0])] = 0.0
    upper_bound = torch.full_like(pw, upper)
    upper_bound[-int(counts[-1]) :] = float("inf")
    largest = dist.max(dim=-1).values
    positive &= (largest > lower_bound[:, None]) & (
        largest < upper_bound[:, None]
    )

    # Contested locations go to the smallest GT (torchvision's
    # 1e8 - area trick keeps the argmax vectorised)
    areas = (gx2 - gx1) * (gy2 - gy1)
    scored = positive.to(boxes.dtype) * (1e8 - areas[None])
    best, best_obj = scored.max(dim=1)

    matching_table = torch.zeros(
        (n_anchors, n_obj), dtype=torch.bool, device=boxes.device
    )
    matched = best >= 1e-5
    matching_table[matched, best_obj[matched]] = True
    return matching_table


def fcos_match(
    y_pred,
    y_true,
    anchors: torch.Tensor,
    radius: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matching for build_fcos_loss: fcos_boxes per image and -- like
    match_all_negatives -- every unmatched location is a negative, the
    focal-loss convention."""
    positives = torch.stack(
        [fcos_boxes(b, anchors, radius) for b in y_true.bboxes]
    )
    return positives, ~positives.any(dim=2)


def slice_classes(concat: torch.Tensor, anchors) -> torch.Tensor:
    """enc_pred stripping the centerness channel off the concat (see
    FCOSPureHead) before the classification loss."""
    return concat[..., :-1]


def slice_ctrness(concat: torch.Tensor, anchors) -> torch.Tensor:
    """enc_pred keeping only the centerness logit of the concat."""
    return concat[..., -1]


def fcos_decoders(score_thresh: float) -> tuple:
    """decode_scores/decode_labels over the [cls_logits | ctrness]
    concat: the detection score is torchvision's
    sqrt(cls_prob * ctrness) geometric mean, so a confident class on a
    badly centred location ranks low for thresholding and NMS."""

    def decode_scores(concat: torch.Tensor) -> torch.Tensor:
        probs = torch.sqrt(
            torch.sigmoid(concat[..., :-1]) * torch.sigmoid(concat[..., -1:])
        )
        probs[..., 0] = 0.0  # exclude background class before taking max
        return probs.max(dim=-1)[0].unsqueeze(-1)

    def decode_labels(concat: torch.Tensor, pad_value=-1) -> torch.Tensor:
        probs = torch.sqrt(
            torch.sigmoid(concat[..., :-1]) * torch.sigmoid(concat[..., -1:])
        )
        probs[..., 0] = 0.0
        scores, labels = probs.max(dim=-1)
        labels = labels.clone()
        labels[scores <= score_thresh] = int(pad_value)
        return labels.unsqueeze(-1)

    return decode_scores, decode_labels


def sigmoid_focal(alpha: float, gamma: float):
    """Focal loss over integer class ids, as in build_ret_loss: select()
    hands over one id per location (0 for negatives); the sigmoid
    convention wants per-class binary targets with background rows
    all-off."""

    def loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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

    return loss


# The single FCOS loss, trainable and decode-faithful at once, mirroring
# torchvision's FCOSHead.compute_loss: sigmoid focal loss over the class
# channels of every location (unmatched ones are plain negatives -- no
# mining, focal loss does the downweighting), GIoU on the decoded boxes
# and BCE on the centerness logit for foreground locations only, all
# three normalised by the foreground count. The centerness subloss rides
# in the scores slot: its predictions are the concat's last channel and
# its target is derived from the matched GT box (true_field="bboxes"),
# not from the annotations. Channel 0 stays reserved for background by
# the shared l2i convention; decode zeroes it before taking the max.
def build_fcos_loss(
    priors: torch.Tensor,
    score_thresh: float,
    radius: float = 1.5,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> DetectionLoss:
    decode_scores, decode_labels = fcos_decoders(score_thresh)
    sublosses = Sublosses(
        bboxes=WeightedLoss(
            loss=sum_normalized(
                partial(generalized_box_iou_loss, reduction="sum")
            ),
            enc_pred=decode_ltrb,
            dec_pred=partial(decode_ltrb, priors=priors),
        ),
        scores=WeightedLoss(
            loss=sum_normalized(
                partial(F.binary_cross_entropy_with_logits, reduction="sum")
            ),
            true_field="bboxes",
            enc_pred=slice_ctrness,
            enc_true=centerness,
            dec_pred=decode_scores,
        ),
        labels=WeightedLoss(
            loss=positive_normalized(sigmoid_focal(alpha, gamma)),
            needs_negatives=True,
            enc_pred=slice_classes,
            dec_pred=decode_labels,
        ),
    )
    return DetectionLoss(
        priors=priors,
        sublosses=sublosses,
        match=partial(fcos_match, radius=radius),
    )


# The trainable flavor keeps RetinaNet's exact geometry -- the retina
# anchor grid, the standard delta codec and IoU matching with focal loss
# -- so the only difference from RETINANET is the head: FCOS's shared
# GroupNorm conv towers plus the centerness channel. Why consider it a
# RetinaNet alternative? Centerness is a learned localisation-quality
# estimate multiplied into the detection score, so badly centred
# anchors are suppressed before NMS instead of outscoring well-centred
# ones -- typically better boxes at the same anchor budget, and far
# fewer anchors needed at the limit (native FCOS runs one per
# location). Keeping everything else identical to RETINANET makes the
# pair a clean A/B test of exactly that head design.
def build_fcos_delta_loss(
    priors: torch.Tensor,
    score_thresh: float,
    overlap: float = 0.35,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> DetectionLoss:
    decode_scores, decode_labels = fcos_decoders(score_thresh)
    sublosses = Sublosses(
        bboxes=WeightedLoss(
            loss=sum_normalized(partial(F.smooth_l1_loss, reduction="sum")),
            enc_true=encode_boxes,
            dec_pred=partial(decode_boxes, priors=priors),
        ),
        scores=WeightedLoss(
            loss=sum_normalized(
                partial(F.binary_cross_entropy_with_logits, reduction="sum")
            ),
            true_field="bboxes",
            enc_pred=slice_ctrness,
            enc_true=centerness,
            dec_pred=decode_scores,
        ),
        labels=WeightedLoss(
            loss=positive_normalized(sigmoid_focal(alpha, gamma)),
            needs_negatives=True,
            enc_pred=slice_classes,
            dec_pred=decode_labels,
        ),
    )
    return DetectionLoss(
        priors=priors,
        sublosses=sublosses,
        match=partial(match_all_negatives, overalp=overlap),
    )


def build_fcos_delta(n_classes, resolution: tuple[int, int]):
    return FCOSPure(n_classes)


def build_torchvision_fcos(
    n_classes=91,
    resolution: tuple[int, int] = (800, 1088),
):
    return FCOSPure(
        n_classes,
        extra_blocks=LastLevelP6P7(256, 256),
        num_anchors=1,
        signed_boxes=False,
    )


# Trainable configuration on RetinaNet's anchor grid and codec -- see
# build_fcos_delta_loss for why it is a drop-in RetinaNet alternative.
FCOS_DELTA = DetectionRecipe(
    build_model=build_fcos_delta,
    anchors=retina_anchors,
    loss=build_fcos_delta_loss,
)

# Faithful-to-torchvision configuration: FCOS's own point grid, linear
# ltrb codec and centre-sampling matcher, for comparing our inference
# against the reference. Trainable like any other flavor (warm_start
# loading, so the head can be sized for any label set).
TORCHVISION_FCOS = DetectionRecipe(
    build_model=build_torchvision_fcos,
    anchors=fcos_anchors,
    loss=build_fcos_loss,
    reference=torchvision_reference(fcos_resnet50_fpn),
)
