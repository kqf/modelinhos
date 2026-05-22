from collections import defaultdict
from typing import Iterator

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


def _iter_class_results(value: dict) -> Iterator[tuple[float, int, dict]]:
    for iou, class_results in value.items():
        if not isinstance(class_results, dict):
            continue
        for class_id, metrics in class_results.items():
            if class_id == "mAP":
                continue
            yield iou, class_id, metrics


def _per_class_fp_fn(
    value: dict,
    pred: np.ndarray,
    true: np.ndarray,
    threshold: float,
) -> dict[int, dict]:
    filt = pred[pred[:, 5] >= threshold] if len(pred) else pred

    per_class: dict[int, dict] = {}
    for _, class_id, metrics in _iter_class_results(value):
        n_pred = int((filt[:, 4] == class_id).sum()) if len(filt) else 0
        n_true = int((true[:, 4] == class_id).sum()) if len(true) else 0
        recall = np.array(metrics["recall"])
        tp = round(float(recall[-1]) * n_true) if len(recall) else 0
        per_class[class_id] = {
            "tp": tp,
            "fp": max(n_pred - tp, 0),
            "fn": max(n_true - tp, 0),
        }

    return per_class


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

        for row in pred:
            confidences[int(row[4])].append(float(row[5]))

    results = metric_fn.value(iou_thresholds=iou_thresholds, *args, **kwargs)
    return _attach_thresholds(results, confidences)


def _attach_thresholds(results: dict, confidences: dict) -> dict:
    sconfidences = {c: sorted(s, reverse=True) for c, s in confidences.items()}
    for _, class_id, metrics in _iter_class_results(results):
        confs = sconfidences.get(class_id, [])
        n_curve = len(metrics["recall"])

        if not confs:
            metrics["thresholds"] = [0.0] * n_curve
            continue
        sentinel = [confs[0] + 1e-6] if len(confs) + 1 == n_curve else []
        metrics["thresholds"] = (sentinel + confs + [0.0] * n_curve)[:n_curve]

    return results


def per_sample_metrics(
    y_true: list[Sample],
    y_pred: list[Sample],
    l2i: dict[str, int],
    iou_threshold: float = 0.5,
    threshold: float = 0.5,
    mpolicy: str = "greedy",
) -> list[dict]:
    num_classes = max(l2i.values()) + 1
    results = []

    for true_sample, pred_sample in zip(y_true, y_pred):
        true = _annotations_to_true(true_sample, l2i)
        pred = _annotations_to_pred(pred_sample, l2i)

        metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=False, num_classes=num_classes
        )
        metric_fn.add(pred, true)
        value = metric_fn.value(
            iou_thresholds=[iou_threshold], mpolicy=mpolicy
        )

        results.append(
            {
                "map": float(value["mAP"]),
                "classes": _per_class_fp_fn(value, pred, true, threshold),
            }
        )

    return results


def visualize_pr(map_results: dict, i2l: dict[int, str]):
    for iou, class_id, metrics in _iter_class_results(map_results):
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


def visualize_fp_fn(
    fp_fn: list[dict[int, dict]],
    i2l: dict[int, str],
    class_agnostic: bool = False,
):
    class_ids = sorted({cid for r in fp_fn for cid in r})

    if class_agnostic:
        fps = [
            sum(r.get(c, {}).get("fp", 0) for c in class_ids) for r in fp_fn
        ]
        fns = [
            sum(r.get(c, {}).get("fn", 0) for c in class_ids) for r in fp_fn
        ]
        panels = [("All classes", fps, fns)]
    else:
        panels = [
            (
                i2l.get(cid, str(cid)),
                [r.get(cid, {}).get("fp", 0) for r in fp_fn],
                [r.get(cid, {}).get("fn", 0) for r in fp_fn],
            )
            for cid in class_ids
        ]

    for label, fps, fns in panels:
        indices = np.arange(len(fp_fn))
        max_count = max(max(fps, default=0), max(fns, default=0)) + 1
        bins = np.arange(0, max_count + 1) - 0.5

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].bar(indices, fps, color="tomato", label="FP")
        axes[0].bar(indices, fns, bottom=fps, color="steelblue", label="FN")
        axes[0].set_xlabel("Image index")
        axes[0].set_ylabel("Count")
        axes[0].set_title("FP / FN per image")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.4)

        axes[1].hist(fps, bins=bins, color="tomato", edgecolor="white")
        axes[1].set_xlabel("FP count")
        axes[1].set_ylabel("Images")
        axes[1].set_title("FP distribution")
        axes[1].grid(axis="y", alpha=0.4)

        axes[2].hist(fns, bins=bins, color="steelblue", edgecolor="white")
        axes[2].set_xlabel("FN count")
        axes[2].set_ylabel("Images")
        axes[2].set_title("FN distribution")
        axes[2].grid(axis="y", alpha=0.4)

        fig.suptitle(label, fontweight="bold")
        plt.tight_layout()
        plt.show()
