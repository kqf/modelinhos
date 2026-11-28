from pathlib import Path

import numpy as np
import pytest
import torch

from modelinhos.containers import Collate
from modelinhos.data import SampleDataset
from modelinhos.sample import Sample, TrainAnnotation
from modelinhos.tasks.standard import STANDARD_SPECS


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
def dataset():  # noqa: DJL002
    def build(samples, augment=None):
        return SampleDataset(
            samples,
            encode_images=lambda image: torch.from_numpy(image).float(),
            specs=STANDARD_SPECS,
            read_image=lambda file_name: np.zeros((4, 4, 3), dtype=np.uint8),
            **({"augment": augment} if augment else {}),
        )

    return build


def test_standard_specs_match_what_annotations_produce(dataset):
    # The given specs must agree with what a real annotated image
    # yields, or an empty image would not stack with an annotated one.
    _, per_image = dataset([sample(2)])[0]
    assert {
        name: (column.dtype, column.shape[1])
        for name, column in vars(per_image).items()
    } == STANDARD_SPECS


def test_dataset_without_a_single_box_builds(dataset):
    # Unlabelled images (inference) need no annotated sample anymore
    _, per_image = dataset([sample(0), sample(0)])[0]
    assert per_image.labels.shape == (0, 1)


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
