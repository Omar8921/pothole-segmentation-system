import pandas as pd
import albumentations as A

from torch.utils.data import DataLoader
from .dataset import SegmentationDataset

def get_dataloader(
    images_dir: str,
    masks_dir: str,
    df: pd.DataFrame,
    split: str,
    transform: A.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True
):
    dataset = SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        df=df,
        split=split,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )