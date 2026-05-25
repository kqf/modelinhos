import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import joblib
import tqdm
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
from modelinhos.ssd.inference import TorchvisionDetector

memory = joblib.Memory("./cachedir", verbose=0)


def build_model(resolution: tuple[int, int] = (300, 300)):
    return TorchvisionDetector(
        resolution=resolution,
        build_model=ssdlite320_mobilenet_v3_large,
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
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
def infer(samples: list[Sample], annotations: Path) -> list[Sample]:
    model = build_model()
    with timer("inference"):
        y_pred = model.transform(
            [
                cv2.imread(str(annotations.parent / s.file_name))
                for s in tqdm.tqdm(samples)
            ]
        )
    return y_pred


def main():
    annotations = Path("datasets/coco/annotations.json")
    samples = load_samples(annotations)
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    for i, sample in enumerate(samples):
        if i > 10:
            continue
        frame = cv2.imread(str(annotations.parent / sample.file_name))
        cv2.imshow("frame", plot(frame, sample))

    le = LabelEncoder(
        l2i={label: i for i, label in enumerate(weights.meta["categories"])}
    )
    y_pred = infer(samples, annotations)

    with timer("inverse transform"):
        y_pred = le.inverse_transform(y_pred)

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
