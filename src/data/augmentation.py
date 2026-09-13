import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),

            A.HorizontalFlip(p=0.5),

            A.Affine(
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                scale=(0.95, 1.05),
                rotate=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=0,
                fill_mask=0,
                p=0.3,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),

            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),

            ToTensorV2(),
        ]
    )


def get_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),

            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=255.0,
            ),

            ToTensorV2(),
        ]
    )