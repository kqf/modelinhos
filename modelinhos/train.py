from functools import partial
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
from modelinhos.ssd.lite import build_ssd_loss, build_ssdlite, ssd_normalize
from modelinhos.trainer.simple import build_trainer

memory = Memory(location=".cache", verbose=0)


def build_model(
    resolution: tuple[int, int],
    epochs: int = 10,
):
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    # anchors only depend on the resolution, not on the weights, so this
    # is a cheap way to get the priors needed to build the loss below
    _, priors = build_ssdlite(resolution=resolution)
    return Detector(
        build_model=partial(
            custom_model,
            resolution=resolution,
            build_model=build_ssdlite,
            weights=weights,
            postprocess=ssd_postprocess,
            normalize=ssd_normalize,
        ),
        lencoder=LabelEncoder(),
        trainer=build_trainer(
            loss_fn=build_ssd_loss(priors, resolution),
            epochs=epochs,
        ),
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

    m_ap = mean_average_precision(train, y_pred, l2i=le.l2i)
    print(m_ap["mAP"].iloc[0])

    aps = per_sample_metrics(train, y_pred, l2i=le.l2i)
    visualize_fp_fn(aps, i2l=le.i2l)


if __name__ == "__main__":
    main()
