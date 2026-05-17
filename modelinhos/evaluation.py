import numpy as np
from mean_average_precision import MetricBuilder

from modelinhos.sample import Sample


def _annotations_to_true(sample: Sample, l2i: dict[str, int]) -> np.ndarray:
    if not sample.annotations:
        return np.empty((0, 7), dtype=np.float32)

    rows = []
    for ann in sample.annotations:
        xmin, ymin, xmax, ymax = ann.bbox
        class_id = l2i[ann.label]
        rows.append([xmin, ymin, xmax, ymax, class_id, 0, 0])

    return np.array(rows, dtype=np.float32)


def _annotations_to_pred(sample: Sample, l2i: dict[str, int]) -> np.ndarray:
    if not sample.annotations:
        return np.empty((0, 6), dtype=np.float32)

    rows = []
    for ann in sample.annotations:
        xmin, ymin, xmax, ymax = ann.bbox
        class_id = l2i[ann.label]
        confidence = 1.0 if np.isnan(ann.score) else ann.score
        rows.append([xmin, ymin, xmax, ymax, class_id, confidence])

    return np.array(rows, dtype=np.float32)


def mean_average_precision(
    y_true: list[Sample],
    y_pred: list[Sample],
    l2i: dict[str, int],
    iou_thresholds: list[float] | None = None,
    *args,
    **kwargs,
) -> dict:
    iou_thresholds = iou_thresholds or [0.5]

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )
    num_classes = max(l2i.values()) + 1
    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=True, num_classes=num_classes
    )

    for true_sample, pred_sample in zip(y_true, y_pred):
        true = _annotations_to_true(true_sample, l2i)
        pred = _annotations_to_pred(pred_sample, l2i)
        metric_fn.add(pred, true)

    return metric_fn.value(iou_thresholds=iou_thresholds, *args, **kwargs)


def per_sample_ap(
    y_true: list[Sample],
    y_pred: list[Sample],
    l2i: dict[str, int],
    iou_thresholds: list[float] | None = None,
    mpolicy: str = "greedy",
) -> list[float]:
    iou_thresholds = iou_thresholds or [0.5]
    num_classes = max(l2i.values()) + 1
    maps = []
    for true_sample, pred_sample in zip(y_true, y_pred):
        metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d",
            async_mode=False,
            num_classes=num_classes,
        )

        metric_fn.add(
            _annotations_to_pred(pred_sample, l2i),
            _annotations_to_true(true_sample, l2i),
        )
        result = metric_fn.value(
            iou_thresholds=iou_thresholds,
            mpolicy=mpolicy,
        )
        maps.append(float(result["mAP"]))
    return maps
