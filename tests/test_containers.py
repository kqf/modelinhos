from pathlib import Path

import numpy as np
import pytest
import torch

from modelinhos.containers import Collate, field_specs
from modelinhos.data import SampleDataset
from modelinhos.sample import Sample, TrainAnnotation


def sample(n_boxes: int) -> Sample[TrainAnnotation]:
    return Sample(
        file_name=Path("fake-file.png"),
        annotations=[
            TrainAnnotation(
                bboxes=(0.2, 0.2, 0.7, 0.7), labels=(1,), scores=(1.0,)
            )
            for _ in range(n_boxes)
        ],
    )


@pytest.fixture
def dataset():
    def build(samples, augment=None):
        return SampleDataset(
            samples,
            encode_images=lambda image: torch.from_numpy(image).float(),
            read_image=lambda file_name: np.zeros((4, 4, 3), dtype=np.uint8),
            **({"augment": augment} if augment else {}),
        )

    return build


def test_specs_come_from_the_first_annotated_sample():
    assert field_specs([sample(0), sample(2)]) == {
        "bboxes": (torch.float32, 4),
        "labels": (torch.int64, 1),
        "scores": (torch.float32, 1),
    }


def test_dataset_without_a_single_box_explodes_at_construction(dataset):
    data = [sample(0), sample(0)]
    with pytest.raises(ValueError, match="no annotated sample"):
        dataset(data)


# The two ways a batch ends up with nothing to infer dtypes from. Both
# used to yield float32 label ids in (0, 1) boxes, which the loss then
# either cast away or choked on.
@pytest.mark.parametrize(
    "samples, augment",
    [
        pytest.param([sample(2), sample(0)], None, id="background-image"),
        pytest.param([sample(2)], lambda i, a: (i, []), id="augmented-empty"),
    ],
)
def test_annotation_free_batches_keep_the_dataset_dtypes(
    dataset, samples, augment
):
    data = dataset(samples, augment=augment)
    _, batch = Collate().collate([data[len(samples) - 1]])

    assert batch.labels.dtype == torch.int64
    assert batch.labels.shape == (1, 0, 1)
    assert batch.bboxes.dtype == torch.float32
    assert batch.bboxes.shape == (1, 0, 4)
