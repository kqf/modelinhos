from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
from torchvision.models.detection import (
    _utils as det_utils,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssdlite import (
    SSDLiteClassificationHead,
    SSDLiteRegressionHead,
    _mobilenet_extractor,
    mobilenet_v3_large,
)

from modelinhos.detector import DetectionRecipe, torchvision_reference
from modelinhos.loss.loss import DetectionLoss
from modelinhos.loss.matching import match
from modelinhos.loss.subloss import (
    Sublosses,
    WeightedLoss,
    positive_normalized,
    sum_normalized,
)
from modelinhos.models.anchors import anchors
from modelinhos.preprocess.boxes import decode_boxes, encode_boxes
from modelinhos.preprocess.image import normalize, rgb_normalized_image_encoder


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


def mobilenet_c3c4c5_extractor(backbone, norm_layer):
    """MobileNetV3-Large extractor tapping strides 8, 16, 32 (C3/C4/C5),
    for use with retina_anchors (steps=[8, 16, 32]).

    torchvision's own _mobilenet_extractor only ever exposes the last two
    native stages (16, 32) plus appended extra downsampling blocks (64,
    128, ...) -- it can't reach anything shallower than stride 16. To get
    a stride-8 map we split the backbone at *two* stride-changing blocks
    instead of one, reusing the same expansion/depthwise trick torchvision
    uses for its single split: each stride-2 InvertedResidual block's own
    block[0] is a stride-1 1x1 expansion, so slicing there taps the
    feature map *before* that block's downsampling, and resuming from
    block[1:] continues seamlessly. For mobilenet_v3_large (reduced_tail),
    these land at blocks 7 (8->16) and 13 (16->32).
    """
    backbone = backbone.features
    stage_indices = (
        [0]
        + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)]
        + [len(backbone) - 1]
    )
    c3_pos, c4_pos = stage_indices[-3], stage_indices[-2]
    if backbone[c3_pos].use_res_connect or backbone[c4_pos].use_res_connect:
        raise ValueError("split blocks must not use a residual connection")

    class MobileNetC3C4C5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Sequential(
                    *backbone[:c3_pos], backbone[c3_pos].block[0]
                ),  # -> stride 8
                torch.nn.Sequential(
                    backbone[c3_pos].block[1:],
                    *backbone[c3_pos + 1 : c4_pos],
                    backbone[c4_pos].block[0],
                ),  # -> stride 16
                torch.nn.Sequential(
                    backbone[c4_pos].block[1:], *backbone[c4_pos + 1 :]
                ),  # -> stride 32
            )

        def forward(self, x):
            out = []
            for block in self.features:
                x = block(x)
                out.append(x)
            return OrderedDict((str(i), v) for i, v in enumerate(out))

    return MobileNetC3C4C5()


class SSDPure(torch.nn.Module):
    def __init__(
        self,
        resolution,
        n_classes,
        num_anchors=2,
        extra=-3,
        backbone_extractor=None,
    ):
        super().__init__()
        self.n_classes = n_classes
        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        backbone = mobilenet_v3_large(
            weights=None,
            progress=True,
            norm_layer=norm_layer,
            reduced_tail=True,
        )
        backbone_extractor = backbone_extractor or (
            lambda b, n: _mobilenet_extractor(b, 6, n)
        )
        self.backbone = backbone_extractor(backbone, norm_layer)
        out_channels = det_utils.retrieve_out_channels(
            self.backbone,
            resolution,
        )[:extra]

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

    # Get real feature map sizes from the backbone. Must run in eval mode:
    # a freshly built module is in train mode, and at small resolutions the
    # deepest feature map is 1x1, where train-mode BatchNorm cannot compute
    # batch statistics (one value per channel) and raises.
    backbone.eval()
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


def torchvision_ssdlite_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors matching build_torchvision_ssdlite. These need real feature
    map sizes, so (unlike ssdlite_anchors) this builds a throwaway backbone
    internally -- only for its feature map shapes, no weights loaded."""
    norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
    backbone = _mobilenet_extractor(
        mobilenet_v3_large(
            weights=None,
            progress=False,
            norm_layer=norm_layer,
            reduced_tail=True,
        ),
        6,
        norm_layer,
    )
    return ssd_anchors(resolution, backbone)


def ssdlite_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors matching build_ssdlite -- a pure function of resolution."""
    return anchors(
        resolution,
        sizes=[[32, 64], [64, 128], [128, 256]],
        steps=[16, 32, 64],
        clip=False,
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
    if weights is not None:
        model = weights(model)
    return model


# This configures retina-net like network
def build_ssdlite(
    resolution: tuple[int, int],
    n_classes: int = 92,
    weights=None,
):
    model = SSDPure(resolution, n_classes=n_classes)
    if weights is not None:
        model = weights(model)
    return model


# weights for encoding/decoding box regression targets, same convention
# as torchvision's SSD; build_ssd_loss below uses this for both encode
# (training) and decode (dec_pred) so the two stay in sync.
SSD_BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def build_ssd_loss(
    priors: torch.Tensor,
    score_thresh: float,
    negpos_ratio: int = 7,
    overlap: float = 0.35,
    ssd_box_weights: tuple[float, float, float, float] = SSD_BOX_WEIGHTS,
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
            enc_true=partial(
                encode_boxes,
                weights=ssd_box_weights,
            ),
            dec_pred=partial(
                decode_boxes,
                priors=priors,
                weights=ssd_box_weights,
            ),
        ),
        # scores are derived from the same logits as labels at decode
        # time, there is nothing to train here
        scores=WeightedLoss(
            loss=None,
            dec_pred=decode_scores,
        ),
        labels=WeightedLoss(
            loss=positive_normalized(
                partial(F.cross_entropy, reduction="sum")
            ),
            needs_negatives=True,
            dec_pred=decode_labels,
        ),
    )
    return DetectionLoss(
        priors=priors,
        sublosses=sublosses,
        match=partial(
            match,
            negpos_ratio=negpos_ratio,
            overalp=overlap,
        ),
    )


# Why the step counts below are what they are -- BatchNorm EMA math:
#
# Training mAP is scored in train() mode (per-batch BN statistics), but
# transform()/predict() run in eval() mode, which normalizes with the BN
# running_mean/running_var buffers instead. Those buffers are NOT trained
# by the optimizer; they only drift toward the data statistics via an EMA,
# once per train-mode forward pass:
#
#     running = (1 - momentum) * running + momentum * batch_stat ~
#
# With the model's momentum = 0.03 and this dataset (identical images, so
# batch_stat is a constant), the leftover fraction of the stale initial
# stats (COCO backbone / mean-0-var-1 fresh head) after N training steps
# is exactly 0.97^N:
#
#     N =  32 -> 38% stale  (mAP 1.0 in train mode, ~0.34 scores in eval,
#                            below the 0.4 threshold -> test fails)
#     N =  64 -> 14% stale  (barely passes, no margin)
#     N = 128 ->  2% stale  (train mode ~= eval mode, safe)
#
# So the test must run ~100+ optimizer steps, where
# N = epochs * n_samples / batch_size. Two non-obvious consequences:
#
#   * Raising batch_size HURTS: EMA updates happen per forward pass, not
#     per sample, so bigger batches mean fewer updates -- and with
#     identical images the batch stats don't improve either. Keep it at 1.
#   * The learning rate is irrelevant here: loss converges in ~1 epoch;
#     the EMA is the only bottleneck, and it ignores the optimizer.
#
# Resolution is kept small (the 32 px dot matches the smallest SSD anchor
# at any image size) so the extra steps stay cheap in CI. If you lower
# epochs or n_samples, redo the 0.97^N arithmetijjk first.

# Trainable configuration: retina-style anchors -- what modelinhos
# trains from scratch / fine-tunes (weights=warm_start(...)).
ssd_normalize = partial(
    normalize,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
)


def retina_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors matching build_retinanet -- a pure function of resolution."""
    return anchors(
        resolution,
        sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
    )


def build_retina_ssdlite(
    resolution: tuple[int, int],
    n_classes: int = 92,
    weights=None,
):
    model = SSDPure(
        resolution,
        n_classes=n_classes,
        extra=None,  # all 3 taps are native and used, nothing to drop
        backbone_extractor=mobilenet_c3c4c5_extractor,
    )
    if weights is not None:
        model = weights(model)
    return model


SSDLARGE = DetectionRecipe(
    build_model=build_retina_ssdlite,
    anchors=retina_anchors,
    loss=build_ssd_loss,
    iencoder=rgb_normalized_image_encoder(ssd_normalize),
)


SSDLITE = DetectionRecipe(
    build_model=build_ssdlite,
    anchors=ssdlite_anchors,
    loss=build_ssd_loss,
    iencoder=rgb_normalized_image_encoder(ssd_normalize),
)

# Faithful-to-torchvision configuration: torchvision's own anchors and
# head shape, for comparing our inference against the reference -- and
# trainable like any other flavor (warm_start loading, so the head can
# be sized for any label set).
TORCHVISION_SSDLITE = DetectionRecipe(
    build_model=build_torchvision_ssdlite,
    anchors=torchvision_ssdlite_anchors,
    loss=build_ssd_loss,
    iencoder=rgb_normalized_image_encoder(ssd_normalize),
    reference=torchvision_reference(ssdlite320_mobilenet_v3_large),
)
