from pathlib import Path

import pandas as pd

from modelinhos.analysis.distributions import (
    bboxes,  # fact: per-box geometry (w, h, area, aspect, label, file)
    divergence,  # verdict: (reference df, other df) -> drift table
    labels,  # fact: per-class counts and shares
)
from modelinhos.analysis.lint import lint
from modelinhos.infos import (
    anchor_advice,  # verdict: matchability df -> ceilings + knob advice
    matchability,  # fact: matcher simulation, per-box matched-anchors
    summarize,  # params / FLOPs / measured latency, data-independent
)
from modelinhos.models.ssdlite import SSDLITE
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import read_samples


def main(
    root: Path = Path("/mnt/data/misc"),
    resolution: tuple[int, int] = (480, 640),
):
    # 0. The label space is the task definition, stated explicitly: we
    # model the test set with the help of the train set, so
    # class_feasibility checks that train covers it, not the reverse.
    lencoder = LabelEncoder(
        l2i={
            "__background__": 0,
            "object": 1,
        }
    )

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
    splits = {n: linted[n].good for n in ("train", "valid", "test")}

    # 3. Check the label distribution for each of the splits
    counts = pd.concat(
        labels(s).assign(split=name) for name, s in splits.items()
    )
    # Print pure label counts, do we have enough labels for each class
    print(counts)

    # Check the box distributions
    geometry = pd.concat(
        bboxes(s).assign(split=name) for name, s in splits.items()
    )
    # Does the data drift between what we fit and what we grade on?
    # This verdict is model independent
    print(
        divergence(
            geometry[geometry.split == "train"],
            geometry[geometry.split == "test"],
        )
    )
    # 6. Model facts: data-independent, once.
    print(
        summarize(SSDLITE, (1, 3, *resolution), n_classes=lencoder.n_classes)
    )

    # 4. The virtual split: the augmentation sampled into concrete data.
    # From here on nothing distinguishes it from a real split.
    # TODO: Implement me later: augmented = materialize(...), the max
    # tries should reflect roughly the number of epochs
    # 5. Facts: same functions on every split, split is just a column.
    # The task lencoder only knows "object" -- this needs step 2 to be a
    # real sanitize (collapse labels) before it stops raising KeyError.
    matched = pd.concat(
        matchability(s, SSDLITE, resolution, lencoder).assign(split=name)
        for name, s in splits.items()
    )

    # 7. Verdicts read facts, never samples -- and they are as
    # split-blind as the facts: run per split and stack, the split
    # column stays ours. Solvability: ceilings on ALL splits.
    print(
        pd.concat(
            anchor_advice(part, SSDLITE, resolution).assign(split=name)
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
