from pathlib import Path

import pandas as pd
import torch

from src.data.augmentation import get_train_transform, get_eval_transform
from src.data.dataloader import get_dataloader

from src.models.factory import get_model

from src.train.loop import train
from src.train.loss import BCEDiceLoss

from src.utils.seed import set_seed

import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

def main():
    config = {
        'seed': 42,
        'model': 'unet',
        'image_size': 512,
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'epochs': 50,
        'patience': 10,
    }

    set_seed(config['seed'])

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    images_dir = Path('data/processed/images')
    masks_dir = Path('data/processed/masks')
    splits_path = Path('data/processed/splits.csv')

    df = pd.read_csv(splits_path)

    train_transform = get_train_transform(
        config['image_size']
    )

    val_transform = get_eval_transform(
        config['image_size']
    )

    train_loader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        df=df,
        split='train',
        transform=train_transform,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )

    val_loader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        df=df,
        split='val',
        transform=val_transform,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )

    model = get_model(
        model_name=config['model'],
        num_classes=1,
    ).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
    )

    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=device.type == 'cuda',
    )

    train(
        epochs=config['epochs'],
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        patience=config['patience'],
        experiments_dir=Path('experiments'),
        experiment_name='unet',
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )


if __name__ == '__main__':
    main()