from functools import partial
from itertools import product
from math import ceil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._api import Weights, WeightsEnum
from torchvision.ops import box_iou
from torchvision.transforms._presets import ObjectDetection

from modelinhos.detector import DetectionRecipe
from modelinhos.loss.loss import DetectionLoss
from modelinhos.loss.matching import match_all_negatives
from modelinhos.loss.subloss import (
    Sublosses,
    WeightedLoss,
    positive_normalized,
    sum_normalized,
)
from modelinhos.models.anchors import anchors
from modelinhos.models.ssdlite import build_ssd_loss
from modelinhos.preprocess.image import (
    normalize,
    rgb_normalized_image_encoder,
)
from modelinhos.sample import Annotation, Sample


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            # ceil_mode matches the conv path (and the ceil-based anchor
            # grids) on odd feature maps; a no-op on even ones.
            self.max_pool = nn.MaxPool2d(
                kernel_size=stride, stride=stride, ceil_mode=True
            )
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


def _blaze_stack(
    in_channels: int,
    schedule: list[tuple[int, int]],
) -> list[BlazeBlock]:
    """BlazeBlocks chained along a (out_channels, stride) schedule."""
    blocks = []
    for out_channels, stride in schedule:
        blocks.append(BlazeBlock(in_channels, out_channels, stride=stride))
        in_channels = out_channels
    return blocks


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=2,
                padding=0,
                groups=channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


def _blaze_stem() -> list[nn.Module]:
    """The shared 5x5/2 input conv; its TFLite padding is applied
    manually in BlazeNet.forward."""
    return [
        nn.Conv2d(3, 24, kernel_size=5, stride=2),
        nn.ReLU(inplace=True),
    ]


def _flatten_head(features: torch.Tensor, coords: int) -> torch.Tensor:
    """Flatten a head map to (b, cells * anchors, coords) in NHWC
    order -- the TFLite anchor layout the checkpoints and the repo
    anchors use; the permute is what keeps that parity."""
    b = features.shape[0]
    return features.permute(0, 2, 3, 1).reshape(b, -1, coords)


class BlazeNet(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Two concrete flavors below: BlazeNetFront and BlazeNetBack; each
    owns its feature extractor and its scale (the input side length),
    the heads and output packing are shared here.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """

    # These are the settings from the MediaPipe example graphs
    # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
    # and
    # mediapipe/graphs/face_detection/face_detection_back_mobile_gpu.pbtxt
    num_classes = 1
    num_anchors = 896
    num_coords = 16
    score_clipping_thresh = 100.0
    min_suppression_threshold = 0.3
    # The four x/y/w/h scales of the MediaPipe config are always one
    # and the same number: the flavor's input side length.
    scale: float
    min_score_thresh: float

    def __init__(self, channels_8: int):
        super().__init__()
        # 2 anchors per stride-8 cell, 6 per stride-16 -- the same
        # counts blaze_anchors below restores.
        self.classifier_8 = nn.Conv2d(channels_8, 2 * self.num_classes, 1)
        self.classifier_16 = nn.Conv2d(96, 6 * self.num_classes, 1)
        self.regressor_8 = nn.Conv2d(channels_8, 2 * self.num_coords, 1)
        self.regressor_16 = nn.Conv2d(96, 6 * self.num_coords, 1)

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The two head maps, finest first."""
        raise NotImplementedError

    def forward(self, image):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(image, (1, 2, 1, 2), "constant", 0)
        x8, x16 = self.features(x)
        r = torch.cat(
            [
                _flatten_head(self.regressor_8(x8), self.num_coords),
                _flatten_head(self.regressor_16(x16), self.num_coords),
            ],
            dim=1,
        )  # (b, 896, 16)
        c = torch.cat(
            [
                _flatten_head(self.classifier_8(x8), self.num_classes),
                _flatten_head(self.classifier_16(x16), self.num_classes),
            ],
            dim=1,
        )  # (b, 896, 1)
        return [r, c]


class BlazeNetFront(BlazeNet):
    """The 128x128 front-camera flavor: head maps at strides 8/16, an
    uneven channel ramp up to 88/96."""

    scale = 128.0
    min_score_thresh = 0.75

    def __init__(self):
        super().__init__(channels_8=88)
        self.backbone1 = nn.Sequential(
            *_blaze_stem(),
            *_blaze_stack(
                24,
                [
                    (24, 1),
                    (28, 1),
                    (32, 2),
                    (36, 1),
                    (42, 1),
                    (48, 2),
                    (56, 1),
                    (64, 1),
                    (72, 1),
                    (80, 1),
                    (88, 1),
                ],
            ),
        )
        self.backbone2 = nn.Sequential(
            *_blaze_stack(88, [(96, 2)] + [(96, 1)] * 4),
        )

    def features(self, x):
        x8 = self.backbone1(x)
        return x8, self.backbone2(x8)


class BlazeNetBack(BlazeNet):
    """The 256x256 back-camera flavor: four runs of seven blocks joined
    by stride-2 transitions, so the head maps sit at strides 16/32."""

    scale = 256.0
    min_score_thresh = 0.65

    def __init__(self):
        super().__init__(channels_8=96)
        self.backbone = nn.Sequential(
            *_blaze_stem(),
            *_blaze_stack(
                24,
                [(24, 1)] * 7
                + [(24, 2)]
                + [(24, 1)] * 7
                + [(48, 2)]
                + [(48, 1)] * 7
                + [(96, 2)]
                + [(96, 1)] * 7,
            ),
        )
        self.final = FinalBlazeBlock(96)

    def features(self, x):
        x8 = self.backbone(x)
        return x8, self.final(x8)


_BLAZE_REPO = "https://github.com/hollance/BlazeFace-PyTorch/raw/master"

_COMMON_META = {
    "categories": ["face"],
    "num_anchors": 896,
    "recipe": "https://github.com/hollance/BlazeFace-PyTorch",
    "_docs": "MediaPipe BlazeFace weights converted from TFLite; "
    "batchnorm is already folded into the convolutions.",
}


class BlazeNet_Weights(WeightsEnum):
    FRONT_V1 = Weights(
        url=f"{_BLAZE_REPO}/blazeface.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "resolution": (128, 128),
            "anchors_url": f"{_BLAZE_REPO}/anchors.npy",
            "model": BlazeNetFront,
        },
    )
    BACK_V1 = Weights(
        url=f"{_BLAZE_REPO}/blazefaceback.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "resolution": (256, 256),
            "anchors_url": f"{_BLAZE_REPO}/anchorsback.npy",
            "model": BlazeNetBack,
        },
    )
    DEFAULT = FRONT_V1


def download_blaze_asset(name: str) -> Path:
    """Fetch a non-checkpoint file (anchors, sample images) from the
    BlazeFace-PyTorch repo into the torch hub cache."""
    directory = Path(torch.hub.get_dir()) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if not path.exists():
        torch.hub.download_url_to_file(f"{_BLAZE_REPO}/{name}", str(path))
    return path


def load_repo_anchors(weights: BlazeNet_Weights) -> torch.Tensor:
    """The reference anchors.npy shipped with the repo -- ground truth
    for blaze_anchors below."""
    path = download_blaze_asset(Path(weights.meta["anchors_url"]).name)
    return torch.tensor(np.load(path), dtype=torch.float32)


def blaze_anchors(
    resolution: tuple[int, int],
    steps: tuple[int, ...] = (8, 16),
    anchors_per_cell: tuple[int, ...] = (2, 6),
) -> torch.Tensor:
    """Anchors matching BlazeNetFront -- a pure function of resolution.

    Restores MediaPipe's SsdAnchorsCalculator config for BlazeFace
    (mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt):
    strides [8, 16, 16, 16] with one aspect ratio plus an interpolated
    scale each, so the three stride-16 layers merge into a single 8x8
    map with 6 anchors per cell while the stride-8 map gets 2. With
    fixed_anchor_size=true the scales never reach the output: every
    anchor is (cx, cy, 1.0, 1.0) in normalized coordinates, and only
    the counts and centers matter.
    """
    H, W = resolution
    out = []
    for step, per_cell in zip(steps, anchors_per_cell):
        fm_h, fm_w = ceil(H / step), ceil(W / step)
        for i, j in product(range(fm_h), range(fm_w)):
            cx = (j + 0.5) / fm_w
            cy = (i + 0.5) / fm_h
            out.extend([[cx, cy, 1.0, 1.0]] * per_cell)
    return torch.tensor(out, dtype=torch.float32).view(-1, 4)


def blaze_back_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors for the back-camera model: same layout, but the first conv
    downsamples by 4, so both feature maps sit at double the stride."""
    return blaze_anchors(resolution, steps=(16, 32))


# Blaze preprocessing is x / 127.5 - 1; after the pipeline's /255 that is
# exactly mean/std of 0.5 -- same convention as ssd_normalize.
blaze_normalize = partial(
    normalize,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
)


def build_blazeface(flavor: Callable[[], BlazeNet]):
    """Recipe build_model for a vanilla fixed-head flavor: weights are
    applied by bake() by convention and resolution belongs to the
    anchors, so building is the constructor plus the label-space guard.
    forward() returns (boxes, classes) with boxes carrying 16 coords
    per anchor (bbox + 6 keypoints), unlike the 4-coord SSD/Retina
    heads. The head always emits one sigmoid face channel; in the
    pipeline convention (index 0 reserved for background) that is
    n_classes=2 with an implicit background, so both 1 (raw) and 2
    (background + face) are accepted."""

    def build(
        resolution: tuple[int, int],
        n_classes: int = 2,
    ) -> BlazeNet:
        if n_classes not in (1, 2):
            raise ValueError(
                "vanilla BlazeNet is single-class (face); rebuild the "
                "classifier/regressor convs before asking for more classes"
            )
        return flavor()

    return build


def decode_blaze_points(
    rel_codes: torch.Tensor,  # (..., N, 2k) interleaved x, y offsets
    priors: torch.Tensor,  # (N, 4) normalised cxcywh
    scale: float,
) -> torch.Tensor:  # (..., N, 2k) normalised interleaved x, y points
    """The point half of MediaPipe's codec: every x offset decodes as
    / scale * pw + pcx and every y as / scale * ph + pcy -- the same
    affine map for box centers and keypoints alike, so the interleaved
    lanes broadcast against the priors, any number of points at once."""
    pcx, pcy, pw, ph = (p[:, None] for p in priors.to(rel_codes).unbind(-1))
    points = torch.empty_like(rel_codes)
    points[..., 0::2] = rel_codes[..., 0::2] / scale * pw + pcx
    points[..., 1::2] = rel_codes[..., 1::2] / scale * ph + pcy
    return points


def decode_blaze_boxes(
    rel_codes: torch.Tensor,  # (B, N, 16) raw blaze regression output
    priors: torch.Tensor,  # (N, 4) normalised cxcywh, sizes fixed at 1.0
    scale: float,
) -> torch.Tensor:  # (B, N, 4) normalised xyxy, keypoints dropped
    """MediaPipe's box codec: offsets are denominated in input pixels
    (scale = the input side length) and sizes decode linearly -- no exp,
    unlike decode_boxes in preprocess/boxes.py."""
    _, _, pw, ph = priors.to(rel_codes).unbind(-1)
    cx, cy = decode_blaze_points(rel_codes[..., :2], priors, scale).unbind(-1)
    w = rel_codes[..., 2] / scale * pw
    h = rel_codes[..., 3] / scale * ph
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        dim=-1,
    )


def encode_blaze_boxes(
    boxes: torch.Tensor,  # (..., 4) normalised xyxy ground-truth boxes
    priors: torch.Tensor,  # (..., 4) normalised cxcywh anchors
    scale: float,
) -> torch.Tensor:  # (..., 4) regression targets, inverse of decode
    pcx, pcy, pw, ph = priors.to(boxes).unbind(-1)
    gx1, gy1, gx2, gy2 = boxes.unbind(-1)
    return torch.stack(
        [
            ((gx1 + gx2) / 2 - pcx) * scale / pw,
            ((gy1 + gy2) / 2 - pcy) * scale / ph,
            (gx2 - gx1) * scale / pw,
            (gy2 - gy1) * scale / ph,
        ],
        dim=-1,
    )


def build_blaze_loss(
    priors: torch.Tensor,
    score_thresh: float,
    scale: float = 128.0,
    score_clip: float = 100.0,
    overlap: float = 0.35,
) -> DetectionLoss:
    """Decode-faithful loss for the vanilla single-face-channel head:
    sigmoid scores (clipped like MediaPipe), face at label index 1, and
    the linear pixel-scale box codec above -- the box loss reads only the
    first 4 of the 16 regression outputs, leaving the keypoints alone.
    Matching is match_all_negatives: the mining path assumes softmax
    logits, which a 1-channel head cannot feed. Note the anchors are
    full-image squares (w = h = 1), so IoU matching only ever assigns
    large boxes -- fine for decoding pretrained weights, a known ceiling
    for training."""

    def decode_labels(raw_logits: torch.Tensor, pad_value=-1) -> torch.Tensor:
        scores = torch.sigmoid(raw_logits.clamp(-score_clip, score_clip))
        labels = torch.ones_like(scores, dtype=torch.long)
        labels[scores <= score_thresh] = int(pad_value)
        return labels

    def decode_scores(raw_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw_logits.clamp(-score_clip, score_clip))

    def face_bce(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            y_pred.view(-1),
            (y_true > 0).to(y_pred.dtype),
            reduction="sum",
        )

    sublosses = Sublosses(
        bboxes=WeightedLoss(
            loss=sum_normalized(partial(F.smooth_l1_loss, reduction="sum")),
            enc_pred=lambda raw, _: raw[..., :4],
            enc_true=partial(encode_blaze_boxes, scale=scale),
            dec_pred=partial(decode_blaze_boxes, priors=priors, scale=scale),
        ),
        scores=WeightedLoss(loss=None, dec_pred=decode_scores),
        labels=WeightedLoss(
            loss=positive_normalized(face_bce),
            needs_negatives=True,
            dec_pred=decode_labels,
        ),
    )
    return DetectionLoss(
        priors=priors,
        sublosses=sublosses,
        match=partial(match_all_negatives, overalp=overlap),
    )


def _weighted_non_max_suppression(
    detections,
    min_suppression_threshold: float,
):
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.

    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0:
        return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    remaining = torch.argsort(detections[:, 16], descending=True)

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.) Boxes are y-first here, but IoU is invariant
        # under swapping the axes of both arguments alike.
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = box_iou(first_box.unsqueeze(0), other_boxes).squeeze(0)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.clone()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :16]
            scores = detections[overlapping, 16:17]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(dim=0) / total_score
            weighted_detection[:16] = weighted
            weighted_detection[16] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections


def _decode_boxes(model: BlazeNet, raw, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.

    Both halves are the shared codec: decode_blaze_boxes for the box,
    stored in the upstream's y-first order, and decode_blaze_points for
    the 6 keypoints.
    """
    boxes = torch.zeros_like(raw)

    x1, y1, x2, y2 = decode_blaze_boxes(
        raw, priors=anchors, scale=model.scale
    ).unbind(-1)
    boxes[..., 0] = y1  # ymin
    boxes[..., 1] = x1  # xmin
    boxes[..., 2] = y2  # ymax
    boxes[..., 3] = x2  # xmax

    boxes[..., 4:] = decode_blaze_points(raw[..., 4:], anchors, model.scale)

    return boxes


def _tensors_to_detections(
    model: BlazeNet,
    raw_box_tensor,
    raw_score_tensor,
    anchors,
    min_score_thresh,
):
    """The output of the neural network is a tensor of shape (b, 896, 16)
    containing the bounding box regressor predictions, as well as a tensor
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" tensors into proper detections.
    Returns a list of (num_detections, 17) tensors, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    assert raw_box_tensor.ndimension() == 3
    assert raw_box_tensor.shape[1] == model.num_anchors
    assert raw_box_tensor.shape[2] == model.num_coords

    assert raw_score_tensor.ndimension() == 3
    assert raw_score_tensor.shape[1] == model.num_anchors
    assert raw_score_tensor.shape[2] == model.num_classes

    assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

    detection_boxes = _decode_boxes(model, raw_box_tensor, anchors)

    thresh = model.score_clipping_thresh
    raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
    detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= min_score_thresh

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
        output_detections.append(torch.cat((boxes, scores), dim=-1))

    return output_detections


def _preprocess(x):
    """Converts the image pixels to the range [-1, 1]."""
    return x.float() / 127.5 - 1.0


def predict_on_batch(
    model: BlazeNet,
    x,
    min_suppression_threshold: float,
    min_score_thresh: float,
):
    """Makes a prediction on a batch of images.

    Arguments:
        x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
            shape (b, 3, H, W). The height and width should match
            model.scale.

    Returns:
        A list containing a tensor of face detections for each image in
        the batch. If no faces are found for an image, returns a tensor
        of shape (0, 17).

    Each face detection is a PyTorch tensor consisting of 17 numbers:
        - ymin, xmin, ymax, xmax
        - x,y-coordinates for the 6 keypoints
        - confidence score
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).permute((0, 3, 1, 2))

    assert x.shape[1] == 3
    assert x.shape[2] == x.shape[3] == int(model.scale)

    # 1. Preprocess the images into tensors:
    x = x.to(model.classifier_8.weight.device)
    x = _preprocess(x)

    # 2. Run the neural network:
    with torch.no_grad():
        out = model(x)

    # 3. Postprocess the raw predictions:
    detections = _tensors_to_detections(
        model,
        out[0],
        out[1],
        model.anchors,
        min_score_thresh,
    )

    output = []
    for i in range(len(detections)):
        faces = _weighted_non_max_suppression(
            detections[i],
            min_suppression_threshold=min_suppression_threshold,
        )
        output.append(
            torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17))
        )
    return output


def predict_on_image(
    model: BlazeNet,
    image,
    min_suppression_threshold: float,
    min_score_thresh: float,
):
    """Makes a prediction on a single image.

    Arguments:
        img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                shape (3, H, W). The image's height and width should
                match model.scale.

    Returns:
        A tensor with face detections.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute((2, 0, 1))

    return predict_on_batch(
        model,
        image.unsqueeze(0),
        min_suppression_threshold=min_suppression_threshold,
        min_score_thresh=min_score_thresh,
    )[0]


def blaze_reference(
    weights: BlazeNet_Weights,
    frames: list[np.ndarray],
    th: float = 0.4,
) -> list[Sample[Annotation]]:
    """The original BlazeFace-PyTorch inference path (repo anchors,
    MediaPipe decode, weighted-blend NMS) as a plain predict function --
    DetectionRecipe.reference for parity tests, same contract as
    torchvision_reference: BGR frames in, relative boxes out. The rows
    predict_on_image returns are (ymin, xmin, ymax, xmax, 6 keypoints,
    score); only the box and score survive the Sample conversion."""
    model: BlazeNet = weights.meta["model"]()
    model.load_state_dict(weights.get_state_dict(progress=True))
    model.anchors = load_repo_anchors(weights)
    label = weights.meta["categories"][0]
    results: list[Sample[Annotation]] = []
    for frame in frames:
        detections = predict_on_image(
            model,
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            min_suppression_threshold=model.min_suppression_threshold,
            min_score_thresh=th,
        )
        results.append(
            Sample(
                file_name=Path("fake-file.png"),
                annotations=[
                    Annotation(
                        bbox=(
                            float(d[1]),
                            float(d[0]),
                            float(d[3]),
                            float(d[2]),
                        ),
                        label=label,
                        score=float(d[16]),
                    )
                    for d in detections
                ],
            )
        )
    return results


class BlazePure(nn.Module):
    """A BlazeFace backbone with SSD-style generic heads, sized for
    retina_anchors: steps 8/16/32, 2 anchors per cell, 4 box coords, no
    keypoints. The flavors below provide stages() -- the three head
    maps -- reusing the donor BlazeNet's modules under their original
    names, so warm_start with the matching checkpoint picks up the
    backbone and leaves the fresh heads alone."""

    channels: tuple[int, int, int]

    def __init__(self, n_classes: int, num_anchors: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.classifiers = nn.ModuleList(
            nn.Conv2d(c, num_anchors * n_classes, 1) for c in self.channels
        )
        self.regressors = nn.ModuleList(
            nn.Conv2d(c, num_anchors * 4, 1) for c in self.channels
        )

    def stages(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three head maps at strides 8/16/32."""
        raise NotImplementedError

    def forward(self, images):
        # Same manual first-conv padding as BlazeNet (TFLite convention).
        x = F.pad(images.float(), (1, 2, 1, 2), "constant", 0)
        maps = self.stages(x)
        boxes = torch.cat(
            [_flatten_head(r(f), 4) for r, f in zip(self.regressors, maps)],
            dim=1,
        )
        classes = torch.cat(
            [
                _flatten_head(c(f), self.n_classes)
                for c, f in zip(self.classifiers, maps)
            ],
            dim=1,
        )
        return boxes, classes


class BlazePureFront(BlazePure):
    """The front backbone extended with a fresh stride-32 stage;
    warm_start(BlazeNet_Weights.FRONT_V1) covers backbone1/backbone2,
    backbone3 trains from scratch."""

    channels = (88, 96, 96)

    def __init__(self, n_classes: int, num_anchors: int = 2):
        super().__init__(n_classes, num_anchors)
        base = BlazeNetFront()
        self.backbone1 = base.backbone1  # stride 8, 88 channels
        self.backbone2 = base.backbone2  # stride 16, 96 channels
        self.backbone3 = FinalBlazeBlock(96)  # stride 32

    def stages(self, x):
        c3 = self.backbone1(x)
        c4 = self.backbone2(c3)
        return c3, c4, self.backbone3(c4)


class BlazePureBack(BlazePure):
    """The back backbone tapped at its stride-8 boundary plus its own
    final stride-32 block; backbone and final keep their checkpoint
    names, so warm_start(BlazeNet_Weights.BACK_V1) covers everything
    but the heads."""

    channels = (48, 96, 96)

    def __init__(self, n_classes: int, num_anchors: int = 2):
        super().__init__(n_classes, num_anchors)
        base = BlazeNetBack()
        self.backbone = base.backbone
        self.final = base.final

    def stages(self, x):
        # backbone[24] is the last stride-8 block (48 channels); the
        # tail from backbone[25] descends to stride 16 at 96 channels.
        c3 = self.backbone[:25](x)
        c4 = self.backbone[25:](c3)
        return c3, c4, self.final(c4)


def retina_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors matching the BlazePure flavors -- a pure function of
    resolution, the same grid as models.retinanet/models.ssdlite."""
    return anchors(
        resolution,
        sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
    )


def build_blaze_retina(flavor: type[BlazePure]):
    """Recipe build_model for a BlazePure flavor: the generic heads are
    sized from n_classes, resolution belongs to the anchors, weights
    are applied by bake() by convention."""

    def build(
        resolution: tuple[int, int],
        n_classes: int = 92,
    ) -> BlazePure:
        return flavor(n_classes=n_classes)

    return build


# The vanilla BlazeFace flavors wired through the Recipe/Detector flow:
# MediaPipe anchors and box codec, blaze normalization, sigmoid face
# channel. reference is the original BlazeFace-PyTorch inference path,
# so parity tests can compare the two end to end (the NMS differs:
# weighted blending there, hard NMS here).
BLAZEFACE_F = DetectionRecipe(
    build_model=build_blazeface(BlazeNetFront),
    anchors=blaze_anchors,
    loss=build_blaze_loss,
    iencoder=rgb_normalized_image_encoder(blaze_normalize),
    reference=blaze_reference,
)

# The back-camera flavor: double the strides, and the box offsets are
# denominated in its 256-pixel input side.
BLAZEFACE_B = DetectionRecipe(
    build_model=build_blazeface(BlazeNetBack),
    anchors=blaze_back_anchors,
    loss=partial(build_blaze_loss, scale=256.0),
    iencoder=rgb_normalized_image_encoder(blaze_normalize),
    reference=blaze_reference,
)

# Trainable blaze flavors on the shared retina anchor grid -- the same
# scheme as models.retinanet.RETINANET and models.ssdlite.SSDLARGE, so
# backbone effects can be compared at fixed anchors. No torchvision
# upstream exists for them, hence no reference.
RETINANET_F = DetectionRecipe(
    build_model=build_blaze_retina(BlazePureFront),
    anchors=retina_anchors,
    loss=build_ssd_loss,
    iencoder=rgb_normalized_image_encoder(blaze_normalize),
)

RETINANET_B = DetectionRecipe(
    build_model=build_blaze_retina(BlazePureBack),
    anchors=retina_anchors,
    loss=build_ssd_loss,
    iencoder=rgb_normalized_image_encoder(blaze_normalize),
)
