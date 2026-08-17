"""albumentations adapter. The wrapper owns the bbox plumbing: the
'albumentations' bbox format is normalized xyxy -- exactly this
library's relative-coordinates convention, so boxes cross the boundary
untouched -- and labels/scores ride along as label_fields, staying in
sync with boxes the pipeline drops. Importing this module requires
albumentations (pip install modelinhos[albumentations])."""

try:
    import albumentations as A
except ImportError as e:
    raise ImportError(
        "the albumentations augment needs albumentations installed -- "
        "pip install modelinhos[albumentations]"
    ) from e

import numpy as np

from modelinhos.augment import Augmentation
from modelinhos.sample import TrainAnnotation


def augment(
    ops: list,
    min_visibility: float = 0.3,
) -> Augmentation:
    """Wrap a list of albumentations ops into an Augmentation.

    min_visibility is the truncated-box policy: a box keeping less than
    this fraction of its area after cropping is dropped, together with
    its labels/scores. It is a signature-level parameter on purpose --
    it must be chosen against the matcher overlap and anchor floor, so
    it belongs where the pipeline is assembled, not buried in defaults.
    Beware the mined-negatives interaction: a dropped box leaves visible
    unlabeled foreground that hard-negative mining will train against.
    """
    pipeline = A.Compose(
        ops,
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["labels", "scores"],
            min_visibility=min_visibility,
        ),
    )

    def f(
        image: np.ndarray,
        annotations: list[TrainAnnotation],
    ) -> tuple[np.ndarray, list[TrainAnnotation]]:
        result = pipeline(
            # albumentations validates its bbox range strictly; ingest
            # (sample.read_samples) already clamps the float noise that
            # would trip it, so boxes go in as they are and a box still
            # outside [0, 1] here is a genuine bug worth the raise.
            image=image,
            bboxes=[a.bboxes for a in annotations],
            labels=[a.labels for a in annotations],
            scores=[a.scores for a in annotations],
        )
        return result["image"], [
            TrainAnnotation(
                bboxes=tuple(float(c) for c in bbox),  # type: ignore
                labels=tuple(int(v) for v in labels),
                scores=tuple(float(s) for s in scores),
            )
            for bbox, labels, scores in zip(
                result["bboxes"],
                result["labels"],
                result["scores"],
            )
        ]

    return f
