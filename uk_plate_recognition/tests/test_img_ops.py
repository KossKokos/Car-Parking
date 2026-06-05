import numpy as np

from anpr.preprocessing.image_ops import (
    BoundingBoxXYXY,
    clip_box_to_image,
    crop_image_xyxy,
)


def test_clip_box_to_image():
    box = BoundingBoxXYXY(
        x1=-10,
        y1=-5,
        x2=220,
        y2=120,
    )

    clipped = clip_box_to_image(
        box=box,
        image_shape=(100, 200, 3),
    )

    assert clipped == (0, 0, 200, 100)


def test_crop_image_xyxy():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    box = BoundingBoxXYXY(
        x1=80,
        y1=40,
        x2=120,
        y2=60,
    )

    crop = crop_image_xyxy(
        image=image,
        box=box,
        padding=0.0,
    )

    assert crop.shape == (20, 40, 3)
