import cv2
import numpy as np
import torch

from modelinhos.blaze.blazenet import BlazeNet

EXPECTED = np.array(
    [
        [
            0.2763,
            0.3182,
            0.4465,
            0.4884,
            0.3830,
            0.3150,
            0.4561,
            0.3202,
            0.4309,
            0.3526,
            0.4229,
            0.3913,
            0.3182,
            0.3373,
            0.4769,
            0.3464,
            0.9308,
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
    front_net = BlazeNet()
    front_net.load_weights("blazeface.pth")
    front_net.load_anchors("anchors.npy")
    front_net.min_score_thresh = 0.75
    front_net.min_suppression_threshold = 0.3
    image = cv2.imread("1face.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictions = front_net.predict_on_image(image)
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
