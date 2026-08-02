import random
from pathlib import Path

from modelinhos.analysis.distributions import (
    bboxes,  # fact: per-box geometry (w, h, area, aspect, label, file)
    divergence,  # verdict: (reference df, other df) -> drift table
    labels,  # fact: per-class counts and shares
    visualize_bboxes,  # view: overlaid geometry histograms
    visualize_labels,  # view: paired share bars over the label union
)
from modelinhos.analysis.lint import lint, rename
from modelinhos.coco import load_samples
from modelinhos.infos import (
    anchor_advice,  # verdict: matchability df -> ceilings + knob advice
    matchability,  # fact: matcher simulation, per-box matched-anchors
    summarize,  # params / FLOPs / measured latency, data-independent
)
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.preprocess.lables import LabelEncoder


def main(
    annotations: Path = Path("datasets/coco/annotations.json"),
    resolution: tuple[int, int] = (320, 320),  # the recipe's native
    holdout: float = 0.2,
    seed: int = 0,
):
    # 0. The label space is the task definition -- the misc task keeps
    # person and car, everything else collapses into "other".
    lencoder = LabelEncoder(
        l2i={"__background__": 0, "person": 1, "car": 2, "other": 3},
    )

    # 1. One dataset, nothing downloaded beyond the evaluation set: a
    # shuffled holdout stands in for train/test -- enough to run every
    # fact and verdict against a real distribution. Labels collapse at
    # ingest, so every fact and verdict below sees the task's classes.
    samples = load_samples(annotations)
    samples = rename(
        samples,
        lencoder,
        new={
            annotation.label: "other"
            for sample in samples
            for annotation in sample.annotations
            if annotation.label not in lencoder.l2i
        },
    )
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

    # 3. Facts: same functions on every split, split stays the dict key.
    counts = {name: labels(part) for name, part in splits.items()}
    # Pure label counts: do we have enough boxes for each class?
    for name, frame in counts.items():
        print(name)
        print(frame.to_string())

    geometry = {name: bboxes(part) for name, part in splits.items()}
    # Does the data drift between what we fit and what we grade on?
    # This verdict is model independent.
    print(divergence(geometry["train"], geometry["test"]))

    # 4. Views: the same train-vs-test comparison as divergence, for
    # eyes -- box geometry at the resolution the model consumes.
    visualize_labels(counts)
    visualize_bboxes(geometry, resolution=resolution)

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
    matched = {
        name: matchability(part, TORCHVISION_SSDLITE, resolution, lencoder)
        for name, part in splits.items()
    }

    # 7. Verdicts read facts, never samples -- and they are as
    # split-blind as the facts: run per split, the split stays our dict
    # key. Solvability: ceilings on ALL splits.
    for name, part in matched.items():
        print(name)
        print(anchor_advice(part, TORCHVISION_SSDLITE, resolution).to_string())
    # Class feasibility is a read, not a function: counts x matched in
    # the advice table above. The one thing no table can show is a task
    # class with zero boxes anywhere -- absent classes have no row:
    print(
        "task classes without data:",
        set(lencoder.l2i)
        - {"__background__"}
        - {label for part in matched.values() for label in part.label},
    )


if __name__ == "__main__":
    main()
