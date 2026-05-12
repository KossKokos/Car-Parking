import numpy as np
import torch

from anpr.inference.predict import PlateRecognizer
from anpr.models.plate_cnn import PlateCNN


def test_plate_recognizer_prepare_image_tensor_shape():
    model = PlateCNN(
        in_channels=1,
        dropout=0.1,
        pooled_width=10,
    )

    recognizer = PlateRecognizer(
        model=model,
        image_height=48,
        image_width=160,
        grayscale=True,
        device="cpu",
    )

    image = np.zeros((80, 240, 1), dtype=np.uint8)

    tensor = recognizer._prepare_image_tensor(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 48, 160)
    assert tensor.dtype == torch.float32