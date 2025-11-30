from modelinhos.blaze.postprocessing import jaccard
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()
        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=2,
                padding=0,
                groups=channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeNet(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """

    def __init__(self, back_model=False):
        super(BlazeNet, self).__init__()

        # These are the settings from the MediaPipe example graphs
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        # and
        # mediapipe/graphs/face_detection/face_detection_back_mobile_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.back_model = back_model
        if back_model:
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
            self.min_score_thresh = 0.65
        else:
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
        if self.back_model:
            self.backbone = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=24,
                    kernel_size=5,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24, stride=2),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 48, stride=2),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.final = FinalBlazeBlock(96)
            self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

            self.regressor_8 = nn.Conv2d(96, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        else:
            self.backbone1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=24,
                    kernel_size=5,
                    stride=2,
                    padding=0,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )

            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

            self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

    def forward(self, image):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(image, (1, 2, 1, 2), "constant", 0)

        b = x.shape[0]  # batch size, needed for reshaping later

        if self.back_model:
            x = self.backbone(x)  # (b, 16, 16, 96)
            h = self.final(x)  # (b, 8, 8, 96)
        else:
            x = self.backbone1(x)  # (b, 88, 16, 16)
            h = self.backbone2(x)  # (b, 96, 8, 8)

        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.

        c1 = self.classifier_8(x)  # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)  # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)  # (b, 512, 1)

        c2 = self.classifier_16(h)  # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)  # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)  # (b, 384, 1)

        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)  # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)  # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)  # (b, 512, 16)

        r2 = self.regressor_16(h)  # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)  # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)  # (b, 384, 16)

        r = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        return [r, c]


def _device(self):
    """Which device (CPU or GPU) is being used by this model?"""
    return self.classifier_8.weight.device


def load_weights(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()


def load_anchors(self, path):
    self.anchors = torch.tensor(
        np.load(path), dtype=torch.float32, device=self._device()
    )
    assert self.anchors.ndimension() == 2
    assert self.anchors.shape[0] == self.num_anchors
    assert self.anchors.shape[1] == 4


def _preprocess(self, x):
    """Converts the image pixels to the range [-1, 1]."""
    return x.float() / 127.5 - 1.0


def predict_on_image(self, img):
    """Makes a prediction on a single image.

    Arguments:
        img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                shape (3, H, W). The image's height and width should be
                128 pixels.

    Returns:
        A tensor with face detections.
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).permute((2, 0, 1))

    return self.predict_on_batch(img.unsqueeze(0))[0]


def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
    """The output of the neural network is a tensor of shape (b, 896, 16)
    containing the bounding box regressor predictions, as well as a tensor
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" tensors into proper detections.
    Returns a list of (num_detections, 17) tensors, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    assert raw_box_tensor.ndimension() == 3
    assert raw_box_tensor.shape[1] == self.num_anchors
    assert raw_box_tensor.shape[2] == self.num_coords

    assert raw_score_tensor.ndimension() == 3
    assert raw_score_tensor.shape[1] == self.num_anchors
    assert raw_score_tensor.shape[2] == self.num_classes

    assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

    detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

    thresh = self.score_clipping_thresh
    raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
    detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= self.min_score_thresh

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
        output_detections.append(torch.cat((boxes, scores), dim=-1))

    return output_detections
