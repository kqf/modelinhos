import cv2
import torch
import torchvision.transforms as T


def build_transform(weights, normalize):
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
            weights.transforms(),
            T.Lambda(normalize),
        ]
    )


def normalize(
    image: torch.Tensor,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
):
    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(image_mean, dtype=dtype, device=device)
    std = torch.as_tensor(image_std, dtype=dtype, device=device)
    return (image - mean[:, None, None]) / std[:, None, None]
