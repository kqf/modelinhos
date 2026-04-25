from functools import partial

import torch
from torchvision.models.detection import (
    _utils as det_utils,
)
from torchvision.models.detection.ssdlite import (
    MobileNet_V3_Large_Weights,
    SSDLiteClassificationHead,
    SSDLiteRegressionHead,
    _mobilenet_extractor,
    mobilenet_v3_large,
)


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
    def __init__(self, resolution, n_classes):
        super().__init__()
        self.n_classes = n_classes
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1

        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        backbone = mobilenet_v3_large(
            weights=weights_backbone,
            progress=True,
            norm_layer=norm_layer,
            reduced_tail=False,
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
        num_anchors = [2 for _ in out_channels]
        self.head = SSDPureHead(
            out_channels=out_channels,
            num_anchors=num_anchors,
            norm_layer=norm_layer,
            n_classes=n_classes,
        )

    def forward(self, images):
        features = self.backbone(images.float())
        features = list(features.values())[:-3]
        return self.head(features)
