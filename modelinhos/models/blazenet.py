from functools import partial
from itertools import product
from math import ceil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models._api import Weights, WeightsEnum
from torchvision.transforms._presets import ObjectDetection

from modelinhos.blaze.blazenet import BlazeNet, FinalBlazeBlock
from modelinhos.blaze.postprocessing import predict_on_image
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
