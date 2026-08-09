"""Simulation side of augmentations: materialize samples a stochastic
augmentation into concrete data, so every fact and verdict function
(analysis and infos alike) runs on augmented data exactly as on a real
split -- training simulated at the sample level, no pipeline needed."""

import pathlib
import random
from typing import Callable

import numpy as np
import torch

from modelinhos.augment import Augmentation
from modelinhos.data import opencv_read
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import Annotation, Sample


def materialize(
    samples: list[Sample[Annotation]],
    augment: Augmentation,
    draws: int = 1,
    seed: int = 0,
    read_image: Callable[[pathlib.Path], np.ndarray] = opencv_read,
) -> list[Sample[Annotation]]:
    """Sample the augmentation into a virtual split: every draw re-rolls
    it over the whole input (draws roughly plays the number of epochs),
    and from the returned samples nothing distinguishes a virtual split
    from a real one. Images are read because the Augmentation contract
    consumes pixels -- inject read_image (SampleDataset convention) to
    feed synthetic frames instead. Labels round-trip through a throwaway
    LabelEncoder, a lossless codec into the TrainAnnotations the
    contract speaks, not a task definition. Seeds the random / numpy /
    torch RNGs globally -- a simulation entry point, not a pure
    function."""
    lencoder = LabelEncoder().fit(samples)
    encoded = lencoder.transform(samples)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = []
    for _ in range(draws):
        for sample in encoded:
            _, annotations = augment(
                read_image(sample.file_name),
                sample.annotations,
            )
            out.append(
                lencoder.inverse_transform(
                    [
                        Sample(
                            file_name=sample.file_name,
                            annotations=annotations,
                        )
                    ]
                )[0]
            )
    return out
