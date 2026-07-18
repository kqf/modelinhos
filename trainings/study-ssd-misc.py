import random
from pathlib import Path

import pandas as pd
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)

from modelinhos.analysis.distributions import (
    bboxes,  # fact: per-box geometry (w, h, area, aspect, label, file)
    divergence,  # verdict: (reference df, other df) -> drift table
    labels,  # fact: per-class counts and shares
    visualize_bboxes,  # view: overlaid geometry histograms
    visualize_labels,  # view: paired share bars over the label union
)
from modelinhos.analysis.lint import lint
from modelinhos.coco import load_samples
from modelinhos.infos import (
    anchor_advice,  # verdict: matchability df -> ceilings + knob advice
    matchability,  # fact: matcher simulation, per-box matched-anchors
    summarize,  # params / FLOPs / measured latency, data-independent
)
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.zoo import coco_label_encoder


def main(
    annotations: Path = Path("datasets/coco/annotations.json"),
    resolution: tuple[int, int] = (320, 320),  # the recipe's native
    holdout: float = 0.2,
    seed: int = 0,
):
    # 0. The label space is the task definition -- here the pretrained
    # checkpoint's own COCO categories, since TORCHVISION_SSDLITE is
    # studied exactly as it was trained.
    lencoder = coco_label_encoder(
        SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
    )

    # 1. One dataset, nothing downloaded beyond the evaluation set: a
    # shuffled holdout stands in for train/test -- enough to run every
    # fact and verdict against a real distribution.
    samples = load_samples(annotations)
    random.Random(seed).shuffle(samples)
    edge = int(len(samples) * holdout)
    parts = {"train": samples[edge:], "test": samples[:edge]}

    # 2. Lint: unreadable files land in .corrupt, images without a
    # single visible box in .problematic; only .good flows on. Plot the
    # problematic samples to review what gets dropped.
    linted = {name: lint(part) for name, part in parts.items()}
    for name, part in linted.items():
        print(
            name,
            part.classes,
            f"problematic: {len(part.problematic)}",
            f"corrupt: {len(part.corrupt)}",
        )
    splits = {name: linted[name].good for name in parts}

    # 3. Facts: same functions on every split, split is just a column.
    counts = pd.concat(
        labels(s).assign(split=name) for name, s in splits.items()
    )
    # Pure label counts: do we have enough boxes for each class?
    print(counts.to_string())

    geometry = pd.concat(
        bboxes(s).assign(split=name) for name, s in splits.items()
    )
    # Does the data drift between what we fit and what we grade on?
    # This verdict is model independent.
    print(
        divergence(
            geometry[geometry.split == "train"],
            geometry[geometry.split == "test"],
        )
    )

    # 4. Views: the same train-vs-test comparison as divergence, for
    # eyes -- box geometry at the resolution the model consumes.
    visualize_labels(
        counts[counts.split == "train"],
        counts[counts.split == "test"],
    )
    visualize_bboxes(
        geometry[geometry.split == "train"],
        geometry[geometry.split == "test"],
        resolution=resolution,
    )

    # 5. Model facts: data-independent, once.
    print(
        summarize(
            TORCHVISION_SSDLITE,
            (1, 3, *resolution),
            n_classes=lencoder.n_classes,
        )
    )

    # 6. The matcher simulation: the recipe's own priors and loss over
    # every GT box of every split.
    matched = pd.concat(
        matchability(s, TORCHVISION_SSDLITE, resolution, lencoder).assign(
            split=name
        )
        for name, s in splits.items()
    )

    # 7. Verdicts read facts, never samples -- and they are as
    # split-blind as the facts: run per split and stack, the split
    # column stays ours. Solvability: ceilings on ALL splits.
    print(
        pd.concat(
            anchor_advice(part, TORCHVISION_SSDLITE, resolution).assign(
                split=name
            )
            for name, part in matched.groupby("split")
        ).to_string()
    )
    # Class feasibility is a read, not a function: counts x matched in
    # the advice table above. The one thing no table can show is a task
    # class with zero boxes anywhere -- absent classes have no row:
    print(
        "task classes without data:",
        set(lencoder.l2i) - {"__background__"} - set(matched.label),
    )


if __name__ == "__main__":
    main()
