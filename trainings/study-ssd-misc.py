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
    lint,  # fact: one row per problem; empty df == clean
    matchability,  # fact: matcher simulation, per-box matched-anchor count
    materialize,  # sample a stochastic augmentation into a concrete dataset
    model_facts,  # params / FLOPs / measured latency, data-independent
)
from modelinhos.augment.albumentations import augment
from modelinhos.models.ssdlite import SSDLITE
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import read_samples


def main(
    root: Path = Path("/mnt/data/misc"),
    resolution: tuple[int, int] = (480, 640),
):
    train = read_samples(root / "train.json")
    valid = read_samples(root / "valid.json")
    test_ = read_samples(root / "test.json")

    # 1. Sanitize the real data first; everything below assumes it passed.
    # This should be a separate scrip
    problems = pd.concat(
        [
            lint(train, resolution).assign(split="train"),
            lint(valid, resolution).assign(split="valid"),
            lint(test_, resolution).assign(split="test"),
        ]
    )

    # Don't asset but just sanitize
    assert problems.empty, problems

    # 2. Label space comes from train only, like in a real run --
    # test labels unseen at fit time surface in class_feasibility.
    # TODO: This is not true. We model the test set, with the help of train set
    # not vice versa
    # This should belong the same script
    lencoder = LabelEncoder(l2i={"__background__": 0, "object": 1}).fit(train)

    # 3. The virtual split: the augmentation sampled into concrete data.
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

    # 4. Facts: same functions on every split, split is just a column.
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

    # 5. Model facts: data-independent, once.
    print(model_facts(SSDLITE, resolution, n_classes=lencoder.n_classes))

    # 6. Verdicts read facts, never samples.
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
