from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
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

    confidences: dict[int, list[float]] = defaultdict(list)
    for true_sample, pred_sample in zip(y_true, y_pred):
        true = _annotations_to_true(true_sample, l2i)
        pred = _annotations_to_pred(pred_sample, l2i)
        metric_fn.add(pred, true)

        # pred columns: [xmin, ymin, xmax, ymax, class_id, confidence]
        for row in pred:
            confidences[int(row[4])].append(float(row[5]))

    results = metric_fn.value(iou_thresholds=iou_thresholds, *args, **kwargs)
    return _attach_thresholds(results, confidences)


def _attach_thresholds(results: dict, confidences: dict) -> dict:
    sconfidences = {c: sorted(s, reverse=True) for c, s in confidences.items()}
    for _, class_results in results.items():
        if not isinstance(class_results, dict):
            continue

        for class_id, metrics in class_results.items():
            if class_id == "mAP":
                continue

            confs = sconfidences.get(class_id, [])
            n_curve = len(metrics["recall"])

            if not confs:
                metrics["thresholds"] = [0.0] * n_curve
                continue
            sentinel = [confs[0] + 1e-6] if len(confs) + 1 == n_curve else []
            metrics["thresholds"] = (sentinel + confs + [0.0] * n_curve)[
                :n_curve
            ]

    return results


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


def visualize_pr(map_results: dict, i2l: dict[int, str]):
    for iou, class_results in map_results.items():
        if not isinstance(class_results, dict):
            continue
        for class_id, metrics in class_results.items():
            if class_id == "mAP":
                continue

            label = i2l.get(class_id, str(class_id))
            recall = metrics["recall"]
            precision = metrics["precision"]
            thresholds = metrics.get("thresholds", [])
            ap = metrics["ap"]

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, label="Precision")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

            ax_thresh = ax.twinx()
            ax_thresh.plot(
                recall,
                thresholds,
                color="orange",
                linestyle="--",
                label="Confidence",
            )
            ax_thresh.set_ylabel("Confidence threshold")
            ax_thresh.set_ylim(0, 1)

            lines, labels = ax.get_legend_handles_labels()
            t_lines, t_labels = ax_thresh.get_legend_handles_labels()
            ax.legend(lines + t_lines, labels + t_labels)

            ax.set_title(f"{label} — AP={ap:.2f} @ IoU={iou:.2f}")
            ax.grid(True)
            plt.tight_layout()
            plt.show()
