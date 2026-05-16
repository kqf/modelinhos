from pathlib import Path

import cv2

from modelinhos.coco import load_samples
from modelinhos.plot import plot


def main():
    annotations = Path("datasets/coco/annotations.json")
    samples = load_samples(annotations)
    print(len(samples))
    for i, sample in enumerate(samples):
        if i > 10:
            continue
        frame = cv2.imread(str(annotations.parent / sample.file_name))
        cv2.imshow("frame", plot(frame, sample))
        cv2.waitKey()


if __name__ == "__main__":
    main()
