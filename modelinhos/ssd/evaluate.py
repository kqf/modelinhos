import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import joblib
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from modelinhos.coco import load_samples
from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
    visualize_pr,
)
from modelinhos.plot import plot
from modelinhos.processing import LabelEncoder
from modelinhos.sample import Sample
from modelinhos.ssd.inference import SampleEncoder, TorchvisionDetector

memory = joblib.Memory("./cachedir", verbose=0)


def build_model(
    resolution: tuple[int, int] = (300, 300),
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
):
    le = LabelEncoder(
        l2i={label: i for i, label in enumerate(weights.meta["categories"])}
    )
    return TorchvisionDetector(
        resolution=resolution,
        build_model=ssdlite320_mobilenet_v3_large,
        weights=weights,
        lencoder=le,
    )


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(
        "{color}[{name}] done in {et:.0f} s{nocolor}".format(
            name=name,
            et=time.time() - t0,
            color="\033[1;33m",
            nocolor="\033[0m",
        )
    )


@memory.cache()
def infer(samples: list[Sample]) -> tuple[list[Sample], SampleEncoder]:
    model = build_model()
    with timer("inference"):
        y_pred = model.transform(samples)
    return y_pred, model.label_encoder


def main():
    annotations = Path("datasets/coco/annotations.json")
    samples = load_samples(annotations)
    for i, sample in enumerate(samples):
        if i > 10:
            continue
        frame = cv2.imread(str(sample.file_name))
        cv2.imshow("frame", plot(frame, sample))

    y_pred, le = infer(samples)

    with timer("mAP calculation"):
        m_ap = mean_average_precision(samples, y_pred, l2i=le.l2i)

    with timer("Per sample calculation"):
        per_sample = per_sample_metrics(samples, y_pred, l2i=le.l2i)

    visualize_fp_fn(per_sample, i2l=le.i2l, class_agnostic=True)

    with timer("Visualize"):
        visualize_pr(m_ap, i2l=le.i2l)

    print("mAP", m_ap["mAP"])


if __name__ == "__main__":
    main()
