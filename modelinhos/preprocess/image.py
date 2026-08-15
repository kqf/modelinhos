import cv2
import torch
import torchvision.transforms as T


def normalize(
    image: torch.Tensor,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
):
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(image_std, dtype=dtype, device=device)
    return (image - mean[:, None, None]) / std[:, None, None]


def rgb_normalized_image_encoder(normalize=normalize):
    return T.Compose(
        [
            T.Lambda(
                lambda frame: cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )
            ),
            T.Lambda(
                lambda frame: (
                    torch.from_numpy(
                        frame,
                    )
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
            ),
            T.Lambda(normalize),
        ]
    )


def grayscale_image_encoder(resolution: tuple[int, int]):
    """Grayscale without normalization: raw (0, 255) intensities go
    straight into the network, whose first BatchNorm learns the input
    statistics itself. Gray is replicated to 3 channels so RGB backbones
    (and their pretrained weights) fit unchanged. Also resizes to
    `resolution` (height, width) -- boxes are relative everywhere, so
    only pixels move and datasets of mixed image sizes (COCO) batch
    cleanly."""
    H, W = resolution
    return T.Compose(
        [
            T.Lambda(lambda frame: cv2.resize(frame, (W, H))),
            T.Lambda(
                lambda frame: cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )
            ),
            T.Lambda(
                lambda frame: (
                    torch.from_numpy(frame)
                    .float()
                    .unsqueeze(0)
                    .repeat(3, 1, 1)
                )
            ),
        ]
    )
