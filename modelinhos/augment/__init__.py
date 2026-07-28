"""Augmentations: joint (image, annotations) transforms applied by the
training dataset only. The contract is framework-free -- uint8 BGR HWC
image plus relative-coordinate TrainAnnotations in, same out -- so
augmentations compose, and any stage can be reviewed by running it in a
loop and plotting the result (inverse-transform the TrainAnnotations
back to Annotations first). One adapter module per framework
(albumentations / torchvision), same optionality convention as the
engines: importing modelinhos.augment.albumentations requires
albumentations, importing this package does not.
"""

from typing import Protocol

import numpy as np

from modelinhos.sample import TrainAnnotation


class Augmentation(Protocol):
    """Joint train-time transform; any plain function with this shape
    satisfies it. A callback Protocol rather than a Callable alias --
    same convention as Engine/SampleEncoder, and the pinned pre-commit
    mypy (v0.921) collapses Callable-typed dataclass fields to their
    return type on attribute access."""

    def __call__(
        self,
        image: np.ndarray,
        annotations: list[TrainAnnotation],
    ) -> tuple[np.ndarray, list[TrainAnnotation]]: ...


def identity(
    image: np.ndarray,
    annotations: list[TrainAnnotation],
) -> tuple[np.ndarray, list[TrainAnnotation]]:
    return image, annotations


def no_augment(resolution: tuple[int, int]) -> Augmentation:
    """The DetectionRecipe.augment default: augment nothing."""
    return identity
