from functools import partial
from itertools import product
from math import ceil
from pathlib import Path

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
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
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


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()
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

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """

    def __init__(self, back_model=False):
        super(BlazeNet, self).__init__()

        # These are the settings from the MediaPipe example graphs
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        # and
        # mediapipe/graphs/face_detection/face_detection_back_mobile_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.back_model = back_model
        if back_model:
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
            self.min_score_thresh = 0.65
        else:
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
        if self.back_model:
            self.backbone = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=24,
                    kernel_size=5,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24, stride=2),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 48, stride=2),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.final = FinalBlazeBlock(96)
            self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

            self.regressor_8 = nn.Conv2d(96, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        else:
            self.backbone1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=24,
                    kernel_size=5,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )

            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

            self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, image):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(image, (1, 2, 1, 2), "constant", 0)

        b = x.shape[0]  # batch size, needed for reshaping later

        if self.back_model:
            x = self.backbone(x)  # (b, 16, 16, 96)
            h = self.final(x)  # (b, 8, 8, 96)
        else:
            x = self.backbone1(x)  # (b, 88, 16, 16)
            h = self.backbone2(x)  # (b, 96, 8, 8)

        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.

        c1 = self.classifier_8(x)  # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)  # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)  # (b, 512, 1)

        c2 = self.classifier_16(h)  # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)  # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)  # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)  # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)  # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)  # (b, 512, 16)

        r2 = self.regressor_16(h)  # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)  # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)  # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return [r, c]


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
            "back_model": False,
        },
    )
    BACK_V1 = Weights(
        url=f"{_BLAZE_REPO}/blazefaceback.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "resolution": (256, 256),
            "anchors_url": f"{_BLAZE_REPO}/anchorsback.npy",
            "back_model": True,
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
    """Anchors matching build_blazenet -- a pure function of resolution.

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


def build_blazenet(
    resolution: tuple[int, int] = (128, 128),
    n_classes: int = 2,
    weights: BlazeNet_Weights | None = None,
    back_model: bool = False,
):
    """Vanilla BlazeFace: fixed single-class heads, loaded strictly so a
    checkpoint/architecture mismatch fails loudly. forward() returns
    (boxes, classes) with boxes carrying 16 coords per anchor (bbox +
    6 keypoints), unlike the 4-coord SSD/Retina heads. The head always
    emits one sigmoid face channel; in the pipeline convention (index 0
    reserved for background) that is n_classes=2 with an implicit
    background, so both 1 (raw) and 2 (background + face) are accepted."""
    if n_classes not in (1, 2):
        raise ValueError(
            "vanilla BlazeNet is single-class (face); rebuild the "
            "classifier/regressor convs before asking for more classes"
        )
    if weights is not None:
        back_model = weights.meta["back_model"]
    model = BlazeNet(back_model=back_model)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True))
    model.eval()
    return model


def decode_blaze_boxes(
    rel_codes: torch.Tensor,  # (B, N, 16) raw blaze regression output
    priors: torch.Tensor,  # (N, 4) normalised cxcywh, sizes fixed at 1.0
    scale: float,
) -> torch.Tensor:  # (B, N, 4) normalised xyxy, keypoints dropped
    """MediaPipe's box codec: offsets are denominated in input pixels
    (scale = the input side length) and sizes decode linearly -- no exp,
    unlike decode_boxes in preprocess/boxes.py."""
    pcx, pcy, pw, ph = priors.to(rel_codes).unbind(-1)
    cx = rel_codes[..., 0] / scale * pw + pcx
    cy = rel_codes[..., 1] / scale * ph + pcy
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

    The box part is the shared codec (decode_blaze_boxes -- the model's
    x/y/w/h scales are always one and the same number), stored in the
    upstream's y-first order; only the 6 keypoints are decoded here.
    """
    boxes = torch.zeros_like(raw)

    x1, y1, x2, y2 = decode_blaze_boxes(
        raw, priors=anchors, scale=model.x_scale
    ).unbind(-1)
    boxes[..., 0] = y1  # ymin
    boxes[..., 1] = x1  # xmin
    boxes[..., 2] = y2  # ymax
    boxes[..., 3] = x2  # xmax

    for k in range(6):
        offset = 4 + k * 2
        keypoint_x = (
            raw[..., offset] / model.x_scale * anchors[:, 2] + anchors[:, 0]
        )  # noqa
        keypoint_y = (
            raw[..., offset + 1] / model.y_scale * anchors[:, 3]
            + anchors[:, 1]  # noqa
        )
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

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
    back_model,
    min_suppression_threshold: float,
    min_score_thresh: float,
):
    """Makes a prediction on a batch of images.

    Arguments:
        x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
            shape (b, 3, H, W). The height and width should be 128 pixels.

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
    if back_model:
        assert x.shape[2] == 256
        assert x.shape[3] == 256
    else:
        assert x.shape[2] == 128
        assert x.shape[3] == 128

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
    back_model,
    min_suppression_threshold: float,
    min_score_thresh: float,
):
    """Makes a prediction on a single image.

    Arguments:
        img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                shape (3, H, W). The image's height and width should be
                128 pixels.

    Returns:
        A tensor with face detections.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute((2, 0, 1))

    return predict_on_batch(
        model,
        image.unsqueeze(0),
        back_model=back_model,
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
    model = build_blazenet(weights=weights)
    model.anchors = load_repo_anchors(weights)
    label = weights.meta["categories"][0]
    results: list[Sample[Annotation]] = []
    for frame in frames:
        detections = predict_on_image(
            model,
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            back_model=weights.meta["back_model"],
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


class BlazePure(torch.nn.Module):
    """The front BlazeFace backbone extended with a stride-32 stage
    (FinalBlazeBlock) and SSD-style generic heads, sized for
    retina_anchors: steps 8/16/32, 2 anchors per cell, 4 box coords, no
    keypoints. The BlazeNet stages keep their original names
    (backbone1/backbone2), so warm_start(BlazeNet_Weights.FRONT_V1)
    picks up the backbone and leaves the fresh heads alone. The stride-2
    residual blocks need even feature maps, so H and W must be divisible
    by 16."""

    def __init__(self, n_classes: int, num_anchors: int = 2):
        super().__init__()
        self.n_classes = n_classes
        base = BlazeNet(back_model=False)
        self.backbone1 = base.backbone1  # stride 8, 88 channels
        self.backbone2 = base.backbone2  # stride 16, 96 channels
        self.backbone3 = FinalBlazeBlock(96)  # stride 32
        channels = (88, 96, 96)
        self.classifiers = torch.nn.ModuleList(
            torch.nn.Conv2d(c, num_anchors * n_classes, 1) for c in channels
        )
        self.regressors = torch.nn.ModuleList(
            torch.nn.Conv2d(c, num_anchors * 4, 1) for c in channels
        )

    def forward(self, images):
        # Same manual first-conv padding as BlazeNet (TFLite convention).
        x = F.pad(images.float(), (1, 2, 1, 2), "constant", 0)
        c3 = self.backbone1(x)
        c4 = self.backbone2(c3)
        c5 = self.backbone3(c4)

        b = images.shape[0]
        boxes = torch.cat(
            [
                r(f).permute(0, 2, 3, 1).reshape(b, -1, 4)
                for r, f in zip(self.regressors, (c3, c4, c5))
            ],
            dim=1,
        )
        classes = torch.cat(
            [
                c(f).permute(0, 2, 3, 1).reshape(b, -1, self.n_classes)
                for c, f in zip(self.classifiers, (c3, c4, c5))
            ],
            dim=1,
        )
        return boxes, classes


def retina_anchors(resolution: tuple[int, int]) -> torch.Tensor:
    """Anchors matching build_retina_blazenet -- a pure function of
    resolution, the same grid as models.retinanet/models.ssdlite."""
    return anchors(
        resolution,
        sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
    )


def build_retina_blazenet(
    resolution: tuple[int, int],
    n_classes: int = 92,
):
    return BlazePure(n_classes=n_classes)


# The standard front-camera BlazeFace wired through the Recipe/Detector
# flow: MediaPipe anchors and box codec, blaze normalization, sigmoid
# face channel. reference is the original BlazeFace-PyTorch inference
# path, so parity tests can compare the two end to end (the NMS differs:
# weighted blending there, hard NMS here).
BLAZEFACE = DetectionRecipe(
    build_model=build_blazenet,
    anchors=blaze_anchors,
    loss=build_blaze_loss,
    iencoder=rgb_normalized_image_encoder(blaze_normalize),
    reference=blaze_reference,
)

# Trainable blaze flavor on the shared retina anchor grid -- the same
# scheme as models.retinanet.RETINANET and models.ssdlite.SSDLARGE, so
# backbone effects can be compared at fixed anchors. No torchvision
# upstream exists for it, hence no reference.
RETINANET = DetectionRecipe(
    build_model=build_retina_blazenet,
    anchors=retina_anchors,
    loss=build_ssd_loss,
    iencoder=rgb_normalized_image_encoder(blaze_normalize),
)
