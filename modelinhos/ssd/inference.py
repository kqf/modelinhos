import math
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
import torch
import torchvision
import tqdm

from modelinhos.sample import Annotation, Sample


def to_blob(frame: np.ndarray, weights) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    preprocess = weights.transforms()
    return preprocess(tensor).unsqueeze(0)


def normalize(
    image: torch.Tensor,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
):
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(image_std, dtype=dtype, device=device)
    return (image - mean[:, None, None]) / std[:, None, None]


def decode_boxes(
    rel_codes: torch.Tensor,  # (B, N, 4)
    priors: torch.Tensor,  # (N, 4)  normalised cxcywh
    image_size: tuple[int, int],  # (H, W)
    weights: tuple = (1.0, 1.0, 1.0, 1.0),
    bbox_xform_clip: float = math.log(1000.0 / 16),
) -> torch.Tensor:  # (B, N, 4)  pixel xyxy
    H, W = image_size
    pcx, pcy, pw, ph = (
        priors.to(rel_codes)
        * priors.new_tensor(
            [
                W,
                H,
                W,
                H,
            ]
        )
    ).unbind(-1)

    dx, dy = rel_codes[..., 0] / weights[0], rel_codes[..., 1] / weights[1]
    dw = torch.clamp(rel_codes[..., 2] / weights[2], max=bbox_xform_clip)
    dh = torch.clamp(rel_codes[..., 3] / weights[3], max=bbox_xform_clip)

    cx = dx * pw + pcx
    cy = dy * ph + pcy
    w = torch.exp(dw) * pw
    h = torch.exp(dh) * ph

    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        dim=-1,
    )


def postprocess(preds, priors, resolution, score_thresh=0.4, iou_thresh=0.5):
    raw_deltas, raw_logits = preds
    boxes = decode_boxes(raw_deltas, priors.to(raw_deltas.device), resolution)
    scores, labels = torch.sigmoid(raw_logits).max(dim=-1)

    annotations = []
    for b in range(scores.shape[0]):
        s, l, bx = scores[b], labels[b], boxes[b]  # noqa
        keep = s > score_thresh
        s, l, bx = s[keep], l[keep], bx[keep]  # noqa
        keep = torchvision.ops.batched_nms(bx, s, l, iou_thresh)
        annotations.append(
            Annotation(
                bbox=bbox.tolist(),
                label=label.item(),
                score=score.item(),
            )
            for bbox, score, label in zip(bx[keep], s[keep], l[keep])
        )
    return Sample(file_name=None, annotations=list(annotations[0]))


@runtime_checkable
class LabelEncoderType(Protocol):
    def fit_transform(self, samples: list[Sample]) -> list[Sample]: ...

    def transform(self, samples: list[Sample]) -> list[Sample]: ...

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]: ...


class DoNothingEncoder:
    def fit_transform(self, samples: list[Sample]) -> list[Sample]:
        return samples

    def transform(self, samples: list[Sample]) -> list[Sample]:
        return samples

    def inverse_transform(self, samples: list[Sample]) -> list[Sample]:
        return samples


class Detector:
    def __init__(
        self,
        resolution: tuple[int, int],
        build_model,
        weights,
        th=0.4,
        postprocess=postprocess,
        normalize=normalize,
        lencoder: LabelEncoderType = DoNothingEncoder(),
    ):
        self.model, self.priors = build_model(
            resolution=resolution,
            weights=weights,
        )
        # Needed for postprocessing
        self.weights = weights
        self.postprocess = postprocess
        self.normalize = normalize
        self.th = th
        self.resolution = resolution
        self.label_encoder = lencoder

    def fit(self, samples: list[Sample]) -> "Detector":
        self.label_encoder.fit_transform(samples)
        return self

    def transform(self, samples: list[Sample]) -> list[Sample]:
        return self.label_encoder.inverse_transform(
            [
                self.transform_single(cv2.imread(str(s.file_name)))
                for s in tqdm.tqdm(samples)
            ]
        )

    def transform_single(self, frame: np.ndarray) -> Sample:
        self.model.eval()
        blob = to_blob(frame, self.weights)
        with torch.no_grad():
            return self.postprocess(
                self.model(self.normalize(blob)),
                self.priors,
                resolution=self.resolution,
                score_thresh=self.th,
            )


def torchvison_to_samples(predictions, anchors, resolution, score_thresh):
    pred = predictions[0]
    annotations = [
        Annotation(
            bbox=b.tolist(),
            label=ll.item(),
            score=s.item(),
        )
        for b, s, ll in zip(
            pred["boxes"].numpy(),
            pred["scores"].numpy(),
            pred["labels"].numpy(),
        )
        if s > score_thresh
    ]
    return Sample(file_name="fake-file.png", annotations=list(annotations))


class TorchvisionDetector(Detector):
    def __init__(
        self,
        resolution: tuple[int, int],
        build_model,
        weights,
        th=0.4,
        postprocess=torchvison_to_samples,
        normalize=lambda x: x,
    ):
        self.model = build_model(
            resolution=resolution,
            weights=weights,
        )
        self.priors = None
        # Needed for postprocessing
        self.weights = weights
        self.postprocess = postprocess
        self.normalize = normalize
        self.th = th
        self.resolution = resolution
