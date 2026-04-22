from torchvision.models.detection.retinanet import (
    LastLevelP6P7,
)

from modelinhos.ssd.anchors import anchors
from modelinhos.ssd.retinanet import RetinaNetPure


def build_retinanet_torchvision(resolution: tuple[int, int]):
    model = RetinaNetPure(
        91,
        extra_blocks=LastLevelP6P7(2048, 256),
        num_anchors=9,
    )
    priors = anchors(
        [
            [16, 20],  # P3  (stride 8)
            [32, 40],  # P4  (stride 16)
            [64, 80],  # P5  (stride 32)
            [128, 160],  # P6  (stride 64)
            [256, 320],  # P7  (stride 128)
        ],
        steps=[
            8,
            16,
            32,
            64,
            128,
        ],
        image_size=resolution,
        clip=False,
    )

    return model, priors
