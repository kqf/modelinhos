from pathlib import Path

import albumentations as A
import pandas as pd

from modelinhos.analysis import (
    anchor_advice,  # verdict: matchability df -> recall ceilings + knob advice
    boxes,  # fact: per-box geometry (w, h, area, aspect, label, file)
    class_feasibility,  # verdict: labels df x matchability df
    # -> per-class verdict
    divergence,  # verdict: (reference df, other df) -> drift table
    labels,  # fact: per-class counts and shares
    matchability,  # fact: matcher simulation, per-box matched-anchor count
    materialize,  # sample a stochastic augmentation into a concrete dataset
    model_facts,  # params / FLOPs / measured latency, data-independent
)
from modelinhos.analysis.lint import check_visible, lint
from modelinhos.augment.albumentations import augment
from modelinhos.models.ssdlite import SSDLITE
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import Annotation, Sample, read_samples


def main(
    root: Path = Path("/mnt/data/misc"),
    resolution: tuple[int, int] = (480, 640),
):
    # 1. Lint: unreadable files land in .corrupt, images without a
    # single visible box in .problematic; only .good flows on. Plot the
    # problematic samples to review what gets dropped.
    linted = {
        name: lint(read_samples(root / f"{name}.json"))
        for name in ("train", "valid", "test")
    }
    for name, part in linted.items():
        print(
            name,
            part.classes,
            f"problematic: {len(part.problematic)}",
            f"corrupt: {len(part.corrupt)}",
        )

    # 2. Sanitize: drop sub-floor boxes (accepting that their objects
    # stay in the image as unlabeled background) and collapse every
    # label into the single "object" class the task models.
    train, valid, test_ = (
        [
            Sample(
                file_name=sample.file_name,
                annotations=[
                    Annotation(bbox=ann.bbox, label="object", score=ann.score)
                    for ann in sample.annotations
                    if check_visible(ann, resolution)
                ],
            )
            for sample in linted[name].good
        ]
        for name in ("train", "valid", "test")
    )

    # 3. The label space is the task definition, stated explicitly: we
    # model the test set with the help of the train set, so
    # class_feasibility checks that train covers it, not the reverse.
    lencoder = LabelEncoder(l2i={"__background__": 0, "object": 1}).fit(train)

    # 4. The virtual split: the augmentation sampled into concrete data.
    # From here on nothing distinguishes it from a real split.
    augmented = materialize(
        train,
        augment(
            [
                A.HorizontalFlip(),
                A.RandomSizedBBoxSafeCrop(
                    480,
                    640,
                ),
            ],
            min_visibility=0.3,
        ),
        draws=8,
        seed=137,
    )

    # 5. Facts: same functions on every split, split is just a column.
    splits = {
        "train": train,
        "train-aug": augmented,
        "valid": valid,
        "test": test_,
    }
    geometry = pd.concat(
        boxes(s).assign(split=name) for name, s in splits.items()
    )
    counts = pd.concat(
        labels(s, lencoder).assign(split=name) for name, s in splits.items()
    )
    matched = pd.concat(
        matchability(s, SSDLITE, resolution).assign(split=name)
        for name, s in splits.items()
    )

    # 6. Model facts: data-independent, once.
    print(model_facts(SSDLITE, resolution, n_classes=lencoder.n_classes))

    # 7. Verdicts read facts, never samples.
    print(anchor_advice(matched))  # solvability: ceilings on ALL splits
    print(class_feasibility(counts, matched))

    # Does the data drift between what we fit and what we grade on?
    print(
        divergence(
            geometry[geometry.split == "train"],
            geometry[geometry.split == "test"],
        )
    )
    # Does the augmentation close that gap or blind the model?
    print(
        divergence(
            geometry[geometry.split == "train-aug"],
            geometry[geometry.split == "test"],
        )
    )


if __name__ == "__main__":
    main()
