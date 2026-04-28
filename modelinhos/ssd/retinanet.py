from functools import partial

import torch
import torchvision
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import (
    LastLevelP6P7,
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
)
from torchvision.models.resnet import ResNet50_Weights, resnet50

from modelinhos.ssd.anchors import anchors, tvison_anchors
from modelinhos.ssd.inference import decode_boxes
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


def build_vanilla_ssd(n_classes, resolution: tuple[int, int], weights):
    model = RetinaNetPure(n_classes)
    if weights is not None:
        load_with_mismatch_from_weights(model, weights=weights, progress=False)
    return (
        model,
        anchors(
            resolution,
            sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
        ),
    )


def postprocess(preds, priors, resolution, score_thresh=0.4, iou_thresh=0.5):
    raw_deltas, raw_logits = preds
    boxes = decode_boxes(raw_deltas, priors.to(raw_deltas.device), resolution)
    scores, labels = torch.sigmoid(raw_logits).max(dim=-1)

    results = []
    for b in range(scores.shape[0]):
        s, l, bx = scores[b], labels[b], boxes[b]  # noqa
        keep = s > score_thresh
        s, l, bx = s[keep], l[keep], bx[keep]  # noqa
        keep = torchvision.ops.batched_nms(bx, s, l, iou_thresh)
        results.append(
            {
                "boxes": bx[keep],
                "scores": s[keep],
                "labels": l[keep],
            }
        )
    return results


def build_retinanet_torchvision(
    n_classes=91,
    resolution: tuple[int, int] = (800, 1088),
    weights=None,
):
    model = RetinaNetPure(
        n_classes,
        extra_blocks=LastLevelP6P7(2048, 256),
        num_anchors=9,
    )
    priors = tvison_anchors(
        resolution=resolution,
        steps=[8, 16, 32, 64, 128],
        aspect_ratios=[0.5, 1.0, 2.0],
        scales=[1.0, 2 ** (1 / 3), 2 ** (2 / 3)],
        base_sizes=[32, 64, 128, 256, 512],
    )
    if weights is not None:
        model.load_state_dict(weights.get_state_dict())
    return model, priors
