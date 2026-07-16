import cv2
import torch

from modelinhos.containers import anno2tensors
from modelinhos.sample import Sample, TrainAnnotation
from modelinhos.tasks.standard import PerImage


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample[TrainAnnotation]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, PerImage]:
        sample = self.samples[idx]
        bgr = cv2.imread(str(sample.file_name))
        image = self.transform(bgr)
        batch = anno2tensors(sample.annotations)
        return image, batch
