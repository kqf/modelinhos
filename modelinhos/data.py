import cv2
import torch

from modelinhos.containers import anno2tensors
from modelinhos.sample import Sample, TrainAnnotation
from modelinhos.tasks.standard import PerImage


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[Sample[TrainAnnotation]],
        encode_images,
    ):
        self.samples = samples
        self.encode_images = encode_images

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, PerImage]:
        sample = self.samples[idx]
        bgr = cv2.imread(str(sample.file_name))
        image = self.encode_images(bgr)
        batch = anno2tensors(sample.annotations)
        return image, batch
