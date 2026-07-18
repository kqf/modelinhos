import warnings
from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mean_average_precision import MetricBuilder
from tqdm import tqdm

from modelinhos.sample import Sample

# The backend concatenates empty match tables on every add(), which
# pandas has deprecated -- one FutureWarning per call floods the
# output. Scoped to warnings issued from inside the package, so our
# own pandas deprecations stay visible.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"mean_average_precision\.",
)


# The mean_average_precision backend computes VOC-style IoU with the
# inclusive +1 pixel convention, so it needs pixel coordinates: fed
# relative boxes directly it would report IoU ~ 1 for any overlapping
# pair. Hence every entry point below scales the relative boxes by an
# explicit evaluation resolution. The +1 also means the numbers depend
# (weakly, ~1/object_size) on that resolution -- pin it to the deploy
# resolution. Goes away once the backend computes geometric IoU (see
# TODO.md).
def _annotations_to_true(
    sample: Sample,
    l2i: dict[str, int],
    resolution: tuple[int, int],
) -> np.ndarray:
    if not sample.annotations:
        return np.empty((0, 7), dtype=np.float32)

    H, W = resolution
    rows = []
    for ann in sample.annotations:
        xmin, ymin, xmax, ymax = ann.bbox
        class_id = l2i[ann.label]
        rows.append([xmin * W, ymin * H, xmax * W, ymax * H, class_id, 0, 0])

    return np.array(rows, dtype=np.float32)


def _annotations_to_pred(
    sample: Sample,
    l2i: dict[str, int],
    resolution: tuple[int, int],
) -> np.ndarray:
    if not sample.annotations:
        return np.empty((0, 6), dtype=np.float32)

    H, W = resolution
    rows = []
    for ann in sample.annotations:
        xmin, ymin, xmax, ymax = ann.bbox
        class_id = l2i[ann.label]
        confidence = 1.0 if np.isnan(ann.score) else ann.score
        rows.append(
            [xmin * W, ymin * H, xmax * W, ymax * H, class_id, confidence]
        )

    return np.array(rows, dtype=np.float32)


def _box_sizes(rows: np.ndarray) -> np.ndarray:
    """sqrt(w * h) per box, in the pixel space the rows are scaled to."""
    if not len(rows):
        return np.empty(0, dtype=np.float32)
    return np.sqrt((rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1]))


def _iter_class_results(value: dict) -> Iterator[tuple[float, int, dict]]:
    for iou, class_results in value.items():
        if not isinstance(class_results, dict):
            continue
        for class_id, metrics in class_results.items():
            if class_id == "mAP":
                continue

            # Exclude empty metrics
            if metrics["precision"].size == 0:
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
        # The recall curve is cumulative over predictions sorted by
        # descending confidence, one point per prediction, so the
        # n_pred-th point is the operating point at `threshold` -- the
        # last point would also count matches below the threshold.
        recall = np.array(metrics["recall"])
        tp = round(float(recall[n_pred - 1]) * n_true) if n_pred else 0
        per_class[class_id] = {
            "tp": tp,
            "fp": max(n_pred - tp, 0),
            "fn": max(n_true - tp, 0),
        }

    # GT classes the model never predicted have no PR curve above, but
    # they still count -- as pure misses.
    gt_classes = {int(c) for c in true[:, 4]} if len(true) else set()
    for class_id in gt_classes - set(per_class):
        n_true = int((true[:, 4] == class_id).sum())
        per_class[class_id] = {"tp": 0, "fp": 0, "fn": n_true}

    return per_class


def _attach_thresholds(results: dict, confidences: dict) -> dict:
    sconfidences = {c: sorted(s, reverse=True) for c, s in confidences.items()}
    for _, class_id, metrics in _iter_class_results(results):
        confs = sconfidences.get(class_id, [])
        # The backend emits exactly one curve point per prediction
        # (difficult/crowd are always 0 here, so nothing gets ignored);
        # a length mismatch means the curve and the collected
        # confidences no longer describe the same predictions.
        if len(confs) != len(metrics["recall"]):
            raise ValueError(
                f"class {class_id}: {len(confs)} confidences for a "
                f"{len(metrics['recall'])}-point PR curve"
            )
        metrics["thresholds"] = confs

    return results


def _mean_ap_over_gt_classes(
    results: dict,
    gt_class_ids: set[int],
) -> float:
    """COCO-style mAP: average AP only over classes that appear in the
    ground truth. Classes without any GT (e.g. a reserved background slot,
    or unused label-space entries) must not dilute the mean -- the metric
    library averages over all num_classes indiscriminately. A GT class the
    model never predicted still counts, as 0."""
    aps = [
        float(class_results[cid]["ap"]) if cid in class_results else 0.0
        for class_results in results.values()
        if isinstance(class_results, dict)
        for cid in gt_class_ids
    ]
    return float(np.mean(aps)) if aps else float("nan")


def _map_results_to_df(results: dict, map_score: float) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for iou, class_id, metrics in _iter_class_results(results):
        recall = metrics["recall"]
        precision = metrics["precision"]
        thresholds = metrics.get("thresholds", [])
        ap = metrics["ap"]
        records.extend(
            {
                "iou": iou,
                "class_id": class_id,
                "recall": float(r),
                "precision": float(p),
                "threshold": float(t),
                "ap": float(ap),
                "mAP": map_score,
            }
            for r, p, t in zip(recall, precision, thresholds)
        )
    if not records:
        # No curves at all (e.g. an epoch with zero confident detections):
        # still expose the mAP so callers like df["mAP"].iloc[0] survive.
        records.append(
            {
                "iou": float("nan"),
                "class_id": float("nan"),
                "recall": float("nan"),
                "precision": float("nan"),
                "threshold": float("nan"),
                "ap": float("nan"),
                "mAP": map_score,
            }
        )
    return pd.DataFrame(records)


def _per_group_to_df(results: list[dict]) -> pd.DataFrame:
    """One record per (group, class): every key except "classes" is the
    group identity (sample_idx, size bin, ...) and repeats per class.
    Groups with no boxes at all produce no records."""
    records: list[dict[str, float]] = []
    for row in results:
        meta = {k: v for k, v in row.items() if k != "classes"}
        records.extend(
            {**meta, "class_id": class_id, **counts}
            for class_id, counts in row["classes"].items()
        )
    return pd.DataFrame(records)


class MetricCollector:
    def __init__(
        self,
        l2i: dict[str, int],
        resolution: tuple[int, int],  # h, w -- pixel space the IoU runs in
        metric_fn=None,
        iou_thresholds: list[float] | None = None,
    ):
        num_classes = max(l2i.values()) + 1

        self.l2i = l2i
        self.resolution = resolution
        self.iou_thresholds = iou_thresholds or [0.5]

        self.metric_fn = metric_fn or MetricBuilder.build_evaluation_metric(
            "map_2d",
            async_mode=True,
            num_classes=num_classes,
        )

        self.confidences: dict[int, list[float]] = defaultdict(list)
        self.gt_class_ids: set[int] = set()

    def __call__(
        self,
        y_true: list[Sample],
        y_pred: list[Sample],
    ) -> None:
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length, "
                f"got {len(y_true)} and {len(y_pred)}"
            )

        for true_sample, pred_sample in zip(y_true, y_pred):
            true = _annotations_to_true(true_sample, self.l2i, self.resolution)
            pred = _annotations_to_pred(pred_sample, self.l2i, self.resolution)

            self.metric_fn.add(pred, true)

            self.gt_class_ids.update(int(row[4]) for row in true)
            for row in pred:
                self.confidences[int(row[4])].append(float(row[5]))

    def value(self, **kwargs) -> pd.DataFrame:
        results = self.metric_fn.value(
            iou_thresholds=self.iou_thresholds,
            **kwargs,
        )
        results = _attach_thresholds(results, self.confidences)
        map_score = _mean_ap_over_gt_classes(results, self.gt_class_ids)
        return _map_results_to_df(results, map_score=map_score)


def mean_average_precision(
    y_true: list[Sample],
    y_pred: list[Sample],
    l2i: dict[str, int],
    resolution: tuple[int, int],  # h, w
    iou_thresholds: list[float] | None = None,
    **kwargs,
) -> pd.DataFrame:
    metric = MetricCollector(
        l2i=l2i,
        resolution=resolution,
        iou_thresholds=iou_thresholds,
    )
    metric(y_true, y_pred)
    return metric.value(**kwargs)


def per_sample_metrics(
    y_true: list[Sample],
    y_pred: list[Sample],
    l2i: dict[str, int],
    resolution: tuple[int, int],  # h, w
    iou_threshold: float = 0.5,
    threshold: float = 0.5,
    mpolicy: str = "greedy",
) -> pd.DataFrame:
    num_classes = max(l2i.values()) + 1
    results = []

    pairs = tqdm(
        zip(y_true, y_pred),
        total=len(y_true),
        desc="Per-sample metrics",
    )
    for idx, (true_sample, pred_sample) in enumerate(pairs):
        true = _annotations_to_true(true_sample, l2i, resolution)
        pred = _annotations_to_pred(pred_sample, l2i, resolution)

        metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=False, num_classes=num_classes
        )
        metric_fn.add(pred, true)
        value = metric_fn.value(
            iou_thresholds=[iou_threshold],
            mpolicy=mpolicy,
        )
        gt_class_ids = {int(row[4]) for row in true}
        results.append(
            {
                "sample_idx": idx,
                "mAP": _mean_ap_over_gt_classes(value, gt_class_ids),
                "classes": _per_class_fp_fn(value, pred, true, threshold),
            }
        )

    return _per_group_to_df(results)


def per_size_metrics(
    y_true: list[Sample],
    y_pred: list[Sample],
    l2i: dict[str, int],
    resolution: tuple[int, int],  # h, w
    bins: list[float],  # sqrt(w * h) edges, pixels at `resolution`
    iou_threshold: float = 0.5,
    threshold: float = 0.5,
    mpolicy: str = "greedy",
) -> pd.DataFrame:
    """Same metrics as per_sample_metrics, grouped by object size
    instead of by image. GT boxes go to the bin of their own size,
    predictions to the bin of theirs, so a prediction whose size falls
    in the wrong bin counts as FP there and FN in the GT's bin --
    boundary noise, acceptable for a histogram. Bins with no boxes at
    all are absent from the output."""
    num_classes = max(l2i.values()) + 1
    trues = [_annotations_to_true(s, l2i, resolution) for s in y_true]
    preds = [_annotations_to_pred(s, l2i, resolution) for s in y_pred]

    results = []
    edges = tqdm(
        zip(bins[:-1], bins[1:]),
        total=len(bins) - 1,
        desc="Per-size metrics",
    )
    for lo, hi in edges:
        metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=False, num_classes=num_classes
        )
        # Seeds keep np.concatenate happy when there are no samples.
        bin_trues = [np.empty((0, 7), dtype=np.float32)]
        bin_preds = [np.empty((0, 6), dtype=np.float32)]
        for true, pred in zip(trues, preds):
            bin_true = true[(_box_sizes(true) >= lo) & (_box_sizes(true) < hi)]
            bin_pred = pred[(_box_sizes(pred) >= lo) & (_box_sizes(pred) < hi)]
            # Matching stays per image: only boxes from the same frame
            # may pair up, the size filter just narrows the population.
            metric_fn.add(bin_pred, bin_true)
            bin_trues.append(bin_true)
            bin_preds.append(bin_pred)

        true = np.concatenate(bin_trues)
        pred = np.concatenate(bin_preds)
        value = metric_fn.value(
            iou_thresholds=[iou_threshold],
            mpolicy=mpolicy,
        )
        gt_class_ids = {int(row[4]) for row in true}
        results.append(
            {
                "size_lo": lo,
                "size_hi": hi,
                "mAP": _mean_ap_over_gt_classes(value, gt_class_ids),
                "classes": _per_class_fp_fn(value, pred, true, threshold),
            }
        )

    return _per_group_to_df(results)


def visualize_pr(
    map_results: pd.DataFrame,
    i2l: dict[int, str],
):
    for (iou, class_id), group in map_results.groupby(["iou", "class_id"]):
        label = i2l.get(class_id, str(class_id))
        recall = group["recall"].tolist()
        precision = group["precision"].tolist()
        thresholds = group["threshold"].tolist()
        ap = group["ap"].iloc[0]

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


def visualize_map_size(per_size: pd.DataFrame):
    # mAP repeats per (bin, class): one value per bin, support summed
    # over classes.
    bins = (
        per_size.groupby(["size_lo", "size_hi"])
        .agg(mAP=("mAP", "first"), tp=("tp", "sum"), fn=("fn", "sum"))
        .reset_index()
    )
    labels = [
        f"{lo:g}–{hi:g}" for lo, hi in zip(bins["size_lo"], bins["size_hi"])
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, bins["mAP"], color="steelblue", edgecolor="white")
    supports = bins["tp"] + bins["fn"]
    for x, (m_ap, support) in enumerate(zip(bins["mAP"], supports)):
        y = 0.0 if np.isnan(m_ap) else m_ap
        ax.annotate(
            f"n={int(support)}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel("Object size $\\sqrt{w \\cdot h}$, px")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1.05)
    ax.set_title("mAP vs object size")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.show()


def visualize_fp_fn(
    per_sample: pd.DataFrame,
    i2l: dict[int, str],
    class_agnostic: bool = False,
):
    if class_agnostic:
        # Rows are per (image, class): collapse to per image, or an
        # image with errors in two classes would show up as two bars.
        per_sample = (
            per_sample.groupby("sample_idx", as_index=False)[["fp", "fn"]]
            .sum()
            .assign(class_id="all classes")
        )

    for class_id, group in per_sample.groupby("class_id"):
        label = i2l.get(class_id, str(class_id))
        fps = group["fp"].to_numpy()
        fns = group["fn"].to_numpy()
        indices = group["sample_idx"].to_numpy()
        max_count = max(fps.max(initial=0), fns.max(initial=0)) + 1
        bins = np.arange(0, max_count + 1) - 0.5

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].bar(indices, fps, color="tomato", label="FP")
        axes[0].bar(indices, fns, bottom=fps, color="steelblue", label="FN")
        axes[0].set_xlabel("Image index")
        axes[0].set_ylabel("Count")
        axes[0].set_title("FP / FN per image")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.4)

        axes[1].hist(
            fps[fps > 0],
            bins=bins,
            color="tomato",
            edgecolor="white",
        )
        axes[1].set_xlabel("FP count")
        axes[1].set_ylabel("Images")
        axes[1].set_title("FP distribution")
        axes[1].grid(axis="y", alpha=0.4)
        axes[2].hist(
            fns[fns > 0],
            bins=bins,
            color="steelblue",
            edgecolor="white",
        )
        axes[2].set_xlabel("FN count")
        axes[2].set_ylabel("Images")
        axes[2].set_title("FN distribution")
        axes[2].grid(axis="y", alpha=0.4)

        fig.suptitle(label, fontweight="bold")
        plt.tight_layout()
        plt.show()
