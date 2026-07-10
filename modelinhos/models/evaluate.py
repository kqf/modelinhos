import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import joblib
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
)

from modelinhos.coco import load_samples
from modelinhos.evaluation import (
    mean_average_precision,
    per_sample_metrics,
    visualize_fp_fn,
    visualize_pr,
)
from modelinhos.models.ssdlite import TORCHVISION_SSDLITE
from modelinhos.plot import plot
from modelinhos.preprocess.lables import SampleEncoder
from modelinhos.sample import Sample
from modelinhos.zoo import coco_label_encoder

memory = joblib.Memory("./cachedir", verbose=0)


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
def infer(
    samples: list[Sample],
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
) -> tuple[list[Sample], SampleEncoder]:
    frames = [cv2.imread(str(sample.file_name)) for sample in samples]
    assert TORCHVISION_SSDLITE.reference is not None
    with timer("inference"):
        y_pred = TORCHVISION_SSDLITE.reference(weights, frames)
    # torchvision-native predictions are in pixel space end to end, so
    # the encoder is only consulted for its l2i/i2l mappings here.
    return y_pred, coco_label_encoder(weights, resolution=(1, 1))


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
