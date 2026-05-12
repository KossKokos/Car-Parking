from __future__ import annotations

import cv2
import numpy as np
import torch


def prepare_plate_tensor(
    image: np.ndarray,
    image_height: int,
    image_width: int,
    grayscale: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Runtime preprocessing for one cropped plate image.

    This intentionally avoids Albumentations so the API environment does not
    need the full training dependency stack.

    Returns:
        Tensor[1, C, H, W]
    """
    if image is None:
        raise ValueError("image cannot be None.")

    if grayscale:
        if image.ndim == 2:
            processed = image
        elif image.ndim == 3 and image.shape[2] == 1:
            processed = image[:, :, 0]
        elif image.ndim == 3 and image.shape[2] == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        processed = cv2.resize(
            processed,
            (image_width, image_height),
            interpolation=cv2.INTER_AREA,
        )

        processed = processed.astype(np.float32) / 255.0
        processed = (processed - 0.5) / 0.5
        processed = processed[None, :, :]

    else:
        if image.ndim == 2:
            processed = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        processed = cv2.resize(
            processed,
            (image_width, image_height),
            interpolation=cv2.INTER_AREA,
        )

        processed = processed.astype(np.float32) / 255.0
        processed = (processed - 0.5) / 0.5
        processed = processed.transpose(2, 0, 1)

    return (
        torch.from_numpy(processed)
        .unsqueeze(0)
        .float()
        .to(device)
    )