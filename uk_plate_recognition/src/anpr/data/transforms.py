from __future__ import annotations

import albumentations as A


def get_eval_transforms(
    image_height: int = 48,
    image_width: int = 160,
) -> A.Compose:
    """
    Deterministic preprocessing for validation/test/inference.

    No augmentation here.
    """
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=(0.5,),
                std=(0.5,),
                max_pixel_value=255.0,
            ),
        ]
    )


def get_train_transforms(
    image_height: int = 48,
    image_width: int = 160,
) -> A.Compose:
    """
    Mild realistic augmentation for cropped plate recognition.

    Keep this conservative. The plate must remain readable.
    """
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),

            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.02, 0.02),
                rotate=(-3.0, 3.0),
                shear=(-2.0, 2.0),
                p=0.35,
            ),

            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                ],
                p=0.20,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.35,
            ),

            A.GaussNoise(
                p=0.20,
            ),

            A.ImageCompression(
                p=0.20,
            ),

            A.Normalize(
                mean=(0.5,),
                std=(0.5,),
                max_pixel_value=255.0,
            ),
        ]
    )


# Backwards-compatible alias for old notebook code.
def get_basic_transforms(
    image_height: int = 48,
    image_width: int = 160,
) -> A.Compose:
    return get_eval_transforms(
        image_height=image_height,
        image_width=image_width,
    )


def get_stress_test_transforms(
    image_height: int = 48,
    image_width: int = 160,
) -> A.Compose:
    """
    Deterministic-style stress evaluation with random realistic distortions.

    This is NOT for final clean test accuracy.
    It is for comparing robustness between models.
    """
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),

            A.Affine(
                scale=(0.94, 1.06),
                translate_percent=(-0.025, 0.025),
                rotate=(-4, 4),
                shear=(-3, 3),
                p=0.50,
            ),

            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                ],
                p=0.35,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.50,
            ),

            A.GaussNoise(
                var_limit=(8.0, 35.0),
                p=0.30,
            ),

            A.ImageCompression(
                quality_lower=45,
                quality_upper=90,
                p=0.35,
            ),

            A.Normalize(
                mean=(0.5,),
                std=(0.5,),
                max_pixel_value=255.0,
            ),
        ]
    )
