import torch
from torch.utils.data import Dataset
import albumentations as A

from typing import Tuple
from pathlib import Path

import pandas as pd
import cv2

class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        df: pd.DataFrame,
        split: str,
        transform: A.Compose = None
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.df = df.loc[df['Split'] == split, :].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file, mask_file = self.df.loc[index, ['Image', 'Mask']]

        image_path = self.images_dir / image_file
        mask_path = self.masks_dir / mask_file

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out['image']
            mask = out['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float() / 255.0
        
        return image, mask