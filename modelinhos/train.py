from pathlib import Path

import cv2
from dadinhos.generate import make_detection_task
from joblib import Memory

from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.models.ssdlite import SSDLITE
from modelinhos.plot import plot
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import Sample, read_samples
from modelinhos.zoo import build_ssd

memory = Memory(location=".cache", verbose=0)


def infer(
    resolution: tuple[int, int],
    samples: list[Sample],
) -> tuple[list[Sample], LabelEncoder]:
    lencoder = LabelEncoder().fit(samples)
    model = build_ssd(resolution, lencoder=lencoder, arch=SSDLITE)
    model.fit(samples)
    return model.transform(samples), lencoder


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

    m_ap = mean_average_precision(
        train, y_pred, l2i=le.l2i, resolution=resolution
    )
    print(m_ap["mAP"].iloc[0])

    aps = per_sample_metrics(train, y_pred, l2i=le.l2i, resolution=resolution)
    visualize_fp_fn(aps, i2l=le.i2l)


if __name__ == "__main__":
    main()
