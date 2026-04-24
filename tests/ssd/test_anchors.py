import numpy as np

from modelinhos.ssd.anchors import anchors, anchors2


def test_anchors(resolution):
    priors = anchors(
        image_size=resolution,
        sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
    )

    priors2 = anchors2(
        sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        image_size=resolution,
        aspect_ratios=[1.0],
        scales=[1.0],
        clip=False,
    )

    np.testing.assert_almost_equal(
        priors.cpu().numpy(),
        priors2.cpu().numpy(),
        decimal=4,
    )
