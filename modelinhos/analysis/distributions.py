"""Distribution facts and the drift verdict between splits. Facts are
unary (list[Sample]) -> DataFrame; divergence(reference, other) is the
only two-frame function -- "split" stays a caller-owned column. Purely
data-side: no model, no torch (model-facing checks live in
modelinhos.inspect)."""

from collections import Counter

import numpy as np
import pandas as pd

from modelinhos.sample import Annotation, Sample


def boxes(samples: list[Sample[Annotation]]) -> pd.DataFrame:
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
