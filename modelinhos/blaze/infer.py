import cv2
import numpy as np
import torch

from modelinhos.blaze.postprocessing import predict_on_image
from modelinhos.models.blazenet import (
    BlazeNet_Weights,
    blaze_anchors,
    build_blazenet,
    download_blaze_asset,
)

# Reference output of the original BlazeFace-PyTorch implementation
# (hollance's blazeface.py, weights + anchors.npy from that repo) on its
# 1face.png sample: ymin, xmin, ymax, xmax, 6 keypoints (x, y), score.
EXPECTED = np.array(
    [
        [
            0.27143508195877075,
            0.31713399291038513,
            0.44155359268188477,
            0.48725229501724243,
            0.3863072991371155,
            0.3126678764820099,
            0.46129563450813293,
            0.3186052143573761,
            0.43995076417922974,
            0.355654776096344,
            0.4327559769153595,
            0.3914948105812073,
            0.3153791129589081,
            0.32960063219070435,
            0.4778282344341278,
            0.3367067277431488,
            0.9959920644760132,
        ]
    ],
)


def plot(image, detections, with_keypoints=True):
    visualized = image.copy()

    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])

    for i in range(detections.shape[0]):
        ymin = int(detections[i, 0] * image.shape[0])
        xmin = int(detections[i, 1] * image.shape[1])
        ymax = int(detections[i, 2] * image.shape[0])
        xmax = int(detections[i, 3] * image.shape[1])

        cv2.rectangle(
            visualized,
            (xmin, ymin),
            (xmax, ymax),
            color=(0, 0, 255),  # red in BGR
            thickness=1,
        )

        if with_keypoints:
            for k in range(6):
                kp_x = int(detections[i, 4 + k * 2] * image.shape[1])
                kp_y = int(detections[i, 4 + k * 2 + 1] * image.shape[0])

                cv2.circle(
                    visualized,
                    (kp_x, kp_y),
                    radius=2,
                    color=(255, 200, 100),  # light-sky-blue-ish in BGR
                    thickness=1,
                )

    return visualized


def main():
    weights = BlazeNet_Weights.FRONT_V1
    front_net = build_blazenet(weights=weights)
    front_net.anchors = blaze_anchors(weights.meta["resolution"])
    image = cv2.imread(str(download_blaze_asset("1face.png")))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictions = predict_on_image(
        front_net,
        image,
        back_model=False,
        min_suppression_threshold=front_net.min_suppression_threshold,
        min_score_thresh=front_net.min_score_thresh,
    )
    np.testing.assert_almost_equal(
        predictions.cpu().numpy(),
        EXPECTED,
    )
    visualized = plot(image, predictions)
    cv2.imshow("Detections", visualized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
