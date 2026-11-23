from pathlib import Path

import cv2
from dadinhos.generate import make_detection_task
from joblib import Memory

from modelinhos.detector import build_detector
from modelinhos.engine.simple import simple_engine
from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
)
from modelinhos.models.load import warm_start
from modelinhos.models.ssdlite import SSDLITE
from modelinhos.plot import plot
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import Sample, read_samples

memory = Memory(location=".cache", verbose=0)


def infer(
    resolution: tuple[int, int],
    samples: list[Sample],
) -> tuple[list[Sample], LabelEncoder]:
    lencoder = LabelEncoder().fit(samples)
    model = build_detector(
        resolution,
        lencoder=lencoder,
        arch=SSDLITE,
        weights=warm_start(SSDLITE.pretraining, fresh=SSDLITE.diff),
        engine=simple_engine(max_epochs=10),
    )
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
        if frame is None:
            raise OSError(f"The files are missing {true.file_name=}")
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
