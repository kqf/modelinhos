import pathlib
from typing import Callable

import cv2
import numpy as np
import torch

from modelinhos.augment import Augmentation, identity
from modelinhos.containers import anno2tensors
from modelinhos.sample import Sample, TrainAnnotation
from modelinhos.tasks.standard import PerImage


def opencv_read(file_name: pathlib.Path) -> np.ndarray:
    return cv2.imread(str(file_name))


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[Sample[TrainAnnotation]],
        encode_images,
        read_image: Callable[[pathlib.Path], np.ndarray] = opencv_read,
        augment: Augmentation = identity,
    ):
        self.samples = samples
        self.encode_images = encode_images
        self.read_image = read_image
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, PerImage]:
        sample = self.samples[idx]
        bgr = self.read_image(sample.file_name)
        # Every __getitem__ re-rolls the augmentation, epoch to epoch
        bgr, annotations = self.augment(bgr, sample.annotations)
        image = self.encode_images(bgr)
        batch = anno2tensors(annotations)
        return image, batch
