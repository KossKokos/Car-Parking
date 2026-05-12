from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from anpr.data.encoders import encode_label


class PlateDataset(Dataset):
    """
    PyTorch Dataset for cropped UK license plate images.

    Returns:
        {
            "image": Tensor[C, H, W],
            "target": Tensor[7],
            "label": str,
            "image_path": str,
        }
    """

    def __init__(
        self,
        metadata: str | Path | pd.DataFrame,
        project_root: str | Path,
        split: str | None = None,
        transform=None,
        image_path_col: str = "image_path",
        label_col: str = "label",
        grayscale: bool = True,
    ) -> None:
        if isinstance(metadata, pd.DataFrame):
            self.df = metadata.copy()
        else:
            self.df = pd.read_csv(metadata)

        if split is not None:
            self.df = self.df[self.df["split"] == split].copy()

        if self.df.empty:
            raise ValueError(f"No rows found for split={split!r}.")

        self.df = self.df.reset_index(drop=True)

        self.project_root = Path(project_root)
        self.transform = transform
        self.image_path_col = image_path_col
        self.label_col = label_col
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]

        image_path = self.project_root / row[self.image_path_col]
        label = row[self.label_col]

        image = self._load_image(image_path)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = image.astype(np.float32) / 255.0

        if image.ndim == 2:
            image = image[..., None]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        target_tensor = torch.tensor(encode_label(label), dtype=torch.long)

        return {
            "image": image_tensor,
            "target": target_tensor,
            "label": label,
            "image_path": str(image_path),
        }

    def _load_image(self, image_path: Path) -> np.ndarray:
        if self.grayscale:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            image = image[..., None]
            return image

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image