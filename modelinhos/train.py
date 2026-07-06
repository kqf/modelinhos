from pathlib import Path

import cv2
from dadinhos.generate import make_detection_task
from joblib import Memory
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)

from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.plot import plot
from modelinhos.postprocess import ssd_postprocess
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import Sample, read_samples
from modelinhos.ssd.inference import Detector, custom_model
from modelinhos.ssd.lite import build_ssdlite, ssd_normalize

memory = Memory(location=".cache", verbose=0)


def build_model(
    resolution: tuple[int, int],
):
    # TODO: Fix the constructor, I want to basically train SSD network architeture
    # As if it was a retina-net one
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    return Detector(
        build_model=custom_model(
            resolution=resolution,
            build_model=build_ssdlite,
            weights=weights,
            postprocess=ssd_postprocess,
            normalize=ssd_normalize,
        ),
        lencoder=LabelEncoder(),
        # TODO: Use the simpler trainer
        trainer=None,
    )


@memory.cache
def infer(
    resolution: tuple[int, int],
    samples: list[Sample],
) -> tuple[list[Sample], LabelEncoder]:
    model = build_model(resolution)
    model.fit(samples)
    return model.transform(samples), model.label_encoder


def main(
    path: Path = Path("data/blobs/annotations.json"),
    resolution: tuple[int, int] = (480, 640),  # height, width
    n_samples: int = 1000,
):
    if not path.exists():
        make_detection_task(
            path,
            resolution=resolution,
            n_samples=n_samples,
            n_classes=3,
        )

    train = read_samples(path)
    y_pred, le = infer(resolution, train)

    for i, (true, pred) in enumerate(zip(train, y_pred)):
        if i > 1:
            continue
        frame = cv2.imread(str(true.file_name))
        cv2.imshow("frame", plot(frame, pred))
        cv2.waitKey()

    l2i = le.label_encoder
    m_ap = mean_average_precision(train, y_pred, l2i=l2i)
    print(m_ap["mAP"].iloc[0])

    aps = per_sample_metrics(train, y_pred, l2i=l2i)
    visualize_fp_fn(aps, i2l={0: "any"})


if __name__ == "__main__":
    main()
