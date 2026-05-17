from pathlib import Path

import cv2
import tqdm
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from modelinhos.coco import load_samples
from modelinhos.evaluation import mean_average_precision
from modelinhos.plot import plot
from modelinhos.processing import LabelEncoder
from modelinhos.ssd.inference import TorchvisionDetector


def build_model(resolution: tuple[int, int] = (300, 300)):
    return TorchvisionDetector(
        resolution=resolution,
        build_model=ssdlite320_mobilenet_v3_large,
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
    )


def main():
    annotations = Path("datasets/coco/annotations.json")
    samples = load_samples(annotations)
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    for i, sample in enumerate(samples):
        if i > 10:
            continue
        frame = cv2.imread(str(annotations.parent / sample.file_name))
        cv2.imshow("frame", plot(frame, sample))
        cv2.waitKey()

    model = build_model()
    y_pred = model.transform(
        [
            cv2.imread(str(annotations.parent / s.file_name))
            for s in tqdm.tqdm(samples)
        ]
    )
    le = LabelEncoder(
        l2i={label: i for i, label in enumerate(weights.meta["categories"])}
    )
    y_pred = le.inverse_transform(y_pred)
    m_ap = mean_average_precision(samples, y_pred, l2i=le.l2i)
    print(m_ap["mAP"])


if __name__ == "__main__":
    main()
