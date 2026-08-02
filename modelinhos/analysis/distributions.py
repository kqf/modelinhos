"""Distribution facts, the drift verdict, and views between splits.
Facts are unary (list[Sample]) -> DataFrame; divergence takes
(reference, other) fact frames, the visualize_* views a dict of them
keyed by split name -- splits stay caller-owned either way. Purely
data-side: no model, no torch (model-facing checks live in
modelinhos.infos)."""

from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from modelinhos.sample import Annotation, Sample


def bboxes(samples: list[Sample[Annotation]]) -> pd.DataFrame:
    """Fact: one row per box -- relative geometry (w, h, area, aspect)
    with the label and source file."""
    return pd.DataFrame(
        [
            {
                "file": str(sample.file_name),
                "label": annotation.label,
                "w": annotation.bbox[2] - annotation.bbox[0],
                "h": annotation.bbox[3] - annotation.bbox[1],
            }
            for sample in samples
            for annotation in sample.annotations
        ]
    ).assign(
        area=lambda df: df.w * df.h,
        aspect=lambda df: df.w / df.h,
    )


def labels(samples: list[Sample[Annotation]]) -> pd.DataFrame:
    """Fact: instance count and share per observed label. Whether the
    labels cover the task's classes is a verdict (class_feasibility)
    -- it owns the label space, this table only reports the data."""
    counts = Counter(
        annotation.label
        for sample in samples
        for annotation in sample.annotations
    )
    return pd.DataFrame(
        [{"label": label, "count": count} for label, count in counts.items()]
    ).assign(share=lambda df: df["count"] / df["count"].sum())


def divergence(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    threshold: float = 0.2,
) -> pd.DataFrame:
    """Verdict: drift between two fact frames, one row per shared
    numeric column. ks is the two-sample Kolmogorov-Smirnov statistic
    (largest gap between the empirical CDFs: 0 identical, 1 disjoint);
    drifted flags columns where it exceeds the threshold. There is no
    recipe knob for drift -- it is a property of the data split."""
    columns = reference.select_dtypes("number").columns.intersection(
        other.select_dtypes("number").columns
    )
    rows = []
    for column in columns:
        a = np.sort(reference[column].to_numpy(dtype=float))
        b = np.sort(other[column].to_numpy(dtype=float))
        grid = np.concatenate([a, b])
        gap = np.abs(
            np.searchsorted(a, grid, side="right") / len(a)
            - np.searchsorted(b, grid, side="right") / len(b)
        )
        rows.append(
            {
                "column": column,
                "ks": float(gap.max()),
                "reference_mean": a.mean(),
                "other_mean": b.mean(),
            }
        )
    return pd.DataFrame(rows).assign(drifted=lambda df: df.ks > threshold)


def visualize_labels(frames: dict[str, pd.DataFrame]):
    """View: grouped share bars over the union of labels, ordered by
    the first frame's count. Shares, not counts -- the frames differ
    in size. A bar next to a gap is the coverage finding: a class one
    split has and another misses. Keys name the bars in the legend."""
    first, *rest = frames.values()
    union = first.sort_values("count", ascending=False).label.tolist()
    for frame in rest:
        union += [label for label in frame.label if label not in union]
    x = np.arange(len(union))
    width = 0.8 / len(frames)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, frame) in enumerate(frames.items()):
        shares = frame.set_index("label").share.reindex(union, fill_value=0)
        shift = (i - (len(frames) - 1) / 2) * width
        ax.bar(x + shift, shares, width=width, label=name)
    ax.set_xticks(x, union)
    ax.set_ylabel("share")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_bboxes(
    frames: dict[str, pd.DataFrame],
    resolution: tuple[int, int] = (1, 1),  # h, w -- the model's
    bins: int = 20,
):
    """View: overlaid share histograms of box geometry on bins shared
    by all frames; keys name the curves in the legend. resolution is
    the resolution the model consumes
    (not the images' own): it scales w/h into the pixel space where
    anchor sizes and the matcher floor live. scale and aspect are
    recomputed after scaling, because the fact frame's relative aspect
    is distorted by the image's own aspect ratio -- it is only a shape
    when the pixel grid is square."""
    H, W = resolution
    unit = "px" if H != 1 and W != 1 else "relative"
    views = {
        name: pd.DataFrame({"w": frame.w * W, "h": frame.h * H}).assign(
            scale=lambda df: np.sqrt(df.w * df.h),
            aspect=lambda df: df.w / df.h,
        )
        for name, frame in frames.items()
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, column in zip(axes, ("w", "h", "scale", "aspect")):
        combined = np.concatenate(
            [view[column].to_numpy() for view in views.values()]
        )
        # Box sizes are log-distributed and anchor levels double in
        # size, so log2 bins keep both readable; aspect is a ratio --
        # linear
        if column == "aspect":
            edges = np.histogram_bin_edges(combined, bins=bins)
            ax.set_xlabel("aspect")
        else:
            edges = np.geomspace(combined.min(), combined.max(), bins + 1)
            ax.set_xscale("log", base=2)
            ax.set_xlabel(f"{column} [{unit}]")
        for name, view in views.items():
            ax.hist(
                view[column],
                bins=edges,
                weights=np.full(len(view), 1 / len(view)),
                histtype="step",
                label=name,
            )
        ax.set_ylabel("share")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()
