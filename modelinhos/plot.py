import cv2
import numpy as np

from modelinhos.sample import Sample


def plot(image_bgr: np.ndarray, sample: Sample) -> np.ndarray:
    for ann in sample.annotations:
        x1, y1, x2, y2 = ann.bbox
        cv2.rectangle(
            image_bgr,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

    cv2.imshow("Predictions", image_bgr)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return image_bgr
