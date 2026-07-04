from functools import partial
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import (
    _utils as det_utils,
)
from torchvision.models.detection.ssdlite import (
    SSDLiteClassificationHead,
    SSDLiteRegressionHead,
    _mobilenet_extractor,
    mobilenet_v3_large,
)

from modelinhos.preprocess.boxes import decode_boxes
from modelinhos.preprocess.image import normalize
from modelinhos.sample import Annotation, Sample
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


def ssd_postprocess(
    preds,
    priors,
    resolution,
    score_thresh=0.004,
    iou_thresh=0.5,
) -> list[Sample]:
    raw_deltas, raw_logits = preds

    device = raw_deltas.device
    priors = priors.to(device)

    boxes = decode_boxes(
        raw_deltas,
        priors,
        resolution,
        weights=(10.0, 10.0, 5.0, 5.0),
    )

    scores_all = torch.softmax(raw_logits, dim=-1)

    annotations: list[Annotation] = []

    for b in range(scores_all.shape[0]):
        boxes_b = boxes[b]
        scores_b = scores_all[b]

        image_boxes = []
        image_scores = []
        image_labels = []

        for cls in range(1, scores_b.shape[1]):  # skip background
            cls_scores = scores_b[:, cls]

            keep = cls_scores > score_thresh
            if not keep.any():
                continue

            image_boxes.append(boxes_b[keep])
            image_scores.append(cls_scores[keep])
            image_labels.append(
                torch.full_like(cls_scores[keep], cls, dtype=torch.int64)
            )

        if not image_boxes:
            continue  # just skip empty images

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        keep = torchvision.ops.batched_nms(
            image_boxes, image_scores, image_labels, iou_thresh
        )

        # convert to Annotation objects
        annotations.extend(
            Annotation(
                bboxes=tuple(bbox.cpu().numpy().tolist()),  # type: ignore
                scores=score.cpu().numpy().item(),
                labels=label.cpu().numpy().item(),
            )
            for bbox, score, label in zip(
                image_boxes[keep],
                image_scores[keep],
                image_labels[keep],
            )
        )
    return [
        Sample(
            file_name=Path("fake-file.png"),
            annotations=list(annotations),
        )
    ]


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
