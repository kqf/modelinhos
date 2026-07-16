from functools import partial
from typing import Callable, Literal

import cv2
import numpy as np

from modelinhos.sample import Sample

AbsoluteXYXY = tuple[float, float, float, float]
LPLOT = Callable[
    [
        np.ndarray,
        str,
        AbsoluteXYXY,
    ],
    np.ndarray,
]


HAlign = Literal["left", "right"]
VAlign = Literal["top", "bottom"]


def plot_label(
    frame: np.ndarray,
    label: str,
    bbox: AbsoluteXYXY,
    halign: HAlign = "left",
    valign: VAlign = "top",
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.4,
    thickness: int = 1,
    color_bg: tuple[int, int, int] = (0, 255, 0),
    color_text: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    x1, y1, x2, y2 = (int(b) for b in bbox)
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    tx = int(x1) if halign == "left" else int(x2) - tw
    if valign == "top":
        ty = int(y1) - 4
        if ty - th < 0:
            ty = int(y2) + th + 4
    else:
        ty = int(y2) + th + 4
        if ty + baseline > frame.shape[0]:
            ty = int(y1) - 4
    cv2.rectangle(
        frame,
        (tx, ty - th - baseline),
        (tx + tw, ty + baseline),
        color_bg,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        label,
        (tx, ty),
        font,
        font_scale,
        color_text,
        thickness,
        cv2.LINE_AA,
    )
    return frame


plot_label_top_left = partial(
    plot_label,
    halign="left",
    valign="top",
)
plot_label_bottom_right = partial(
    plot_label,
    halign="right",
    valign="bottom",
)


def plot(
    image_bgr: np.ndarray,
    sample: Sample,
    plot_label: LPLOT = plot_label_bottom_right,
) -> np.ndarray:
    for ann in sample.annotations:
        x1, y1, x2, y2 = (int(v) for v in ann.bboxes)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        image_bgr = plot_label(image_bgr, ann.labels, ann.bboxes)

    return image_bgr
