"""torchvision transforms.v2 adapter. Boxes are converted relative ->
pixel tv_tensors.BoundingBoxes for the pipeline and back on the way
out. Box identity is tracked by index through the pipeline (the indices
double as the "labels" v2's SanitizeBoundingBoxes filters in sync), so
labels/scores stay attached and min_visibility can compare each box's
pixel area before and after. torchvision is a base dependency; the
guard below is about the v2 API generation, not optionality."""

import numpy as np
import torch

try:
    from torchvision import tv_tensors
    from torchvision.transforms import v2
except ImportError as e:
    raise ImportError(
        "the torchvision augment needs torchvision >= 0.16 "
        "(transforms.v2 and tv_tensors)"
    ) from e

from modelinhos.augment import Augmentation
from modelinhos.sample import TrainAnnotation


def augment(
    ops: list,
    min_visibility: float = 0.3,
) -> Augmentation:
    """Wrap a list of torchvision.transforms.v2 ops into an Augmentation.

    min_visibility matches the albumentations wrapper: the fraction of a
    box's pixel area surviving the pipeline. Exact for crop/flip-style
    ops; ops that rescale the whole frame (Resize) shift every box's
    fraction alike, so pick the threshold for the geometric pipeline you
    actually compose. See albumentations.albumentations_augment for why
    it is exposed here.
    """
    pipeline = v2.Compose([*ops, v2.ClampBoundingBoxes()])

    def augment(
        image: np.ndarray,
        annotations: list[TrainAnnotation],
    ) -> tuple[np.ndarray, list[TrainAnnotation]]:
        h, w = image.shape[:2]
        scale = np.array([w, h, w, h], dtype=np.float32)
        boxes = np.array(
            [a.bboxes for a in annotations],
            dtype=np.float32,
        ).reshape(-1, 4)
        pixel_areas = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (w * h)
        )

        out = pipeline(
            {
                "image": tv_tensors.Image(
                    torch.from_numpy(image).permute(2, 0, 1)
                ),
                "boxes": tv_tensors.BoundingBoxes(
                    torch.from_numpy(boxes * scale),
                    format="XYXY",
                    canvas_size=(h, w),
                ),
                "labels": torch.arange(len(annotations)),
            }
        )

        out_h, out_w = out["boxes"].canvas_size
        out_scale = np.array([out_w, out_h, out_w, out_h], dtype=np.float32)
        out_boxes = np.asarray(out["boxes"], dtype=np.float32).reshape(-1, 4)
        out_areas = (out_boxes[:, 2] - out_boxes[:, 0]) * (
            out_boxes[:, 3] - out_boxes[:, 1]
        )

        survivors = []
        for box, area, index in zip(out_boxes, out_areas, out["labels"]):
            source = annotations[int(index)]
            before = pixel_areas[int(index)]
            if before <= 0 or area / before < min_visibility:
                continue
            survivors.append(
                TrainAnnotation(
                    bboxes=tuple(float(c) for c in box / out_scale),  # type: ignore
                    labels=source.labels,
                    scores=source.scores,
                )
            )

        frame = np.ascontiguousarray(
            out["image"].permute(1, 2, 0).numpy(),
        )
        return frame, survivors

    return augment
