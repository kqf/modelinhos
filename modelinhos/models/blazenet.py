from functools import partial
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
import torch
from torchvision.models._api import Weights, WeightsEnum
from torchvision.transforms._presets import ObjectDetection

from modelinhos.blaze.blazenet import BlazeNet
from modelinhos.preprocess.image import normalize

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
    downsamples by 4, so both maps sit one octave deeper."""
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
    n_classes: int = 1,
    weights: BlazeNet_Weights | None = None,
    back_model: bool = False,
):
    """Vanilla BlazeFace: fixed single-class heads, loaded strictly so a
    checkpoint/architecture mismatch fails loudly. forward() returns
    (boxes, classes) with boxes carrying 16 coords per anchor (bbox +
    6 keypoints), unlike the 4-coord SSD/Retina heads."""
    if n_classes != 1:
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
