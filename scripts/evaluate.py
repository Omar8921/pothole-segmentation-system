from pathlib import Path
import argparse
import json

import pandas as pd
import torch

from src.data.augmentation import get_eval_transform
from src.data.dataloader import get_dataloader

from src.models.factory import get_model

from src.train.evaluation import evaluate_model
from src.train.loss import BCEDiceLoss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load checkpoint first
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    config = checkpoint["config"]

    # Dataset
    images_dir = Path("data/processed/images")
    masks_dir = Path("data/processed/masks")
    splits_path = Path("data/processed/splits.csv")

    df = pd.read_csv(splits_path)

    test_transform = get_eval_transform(
        config["image_size"]
    )

    test_loader = get_dataloader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        df=df,
        split="test",
        transform=test_transform,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # Model
    model = get_model(
        model_name=config["model"],
        num_classes=1,
    )

    model.load_state_dict(
        checkpoint["model_weights"]
    )

    model = model.to(device)

    # Same loss used during training
    criterion = BCEDiceLoss()

    # Evaluate
    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nTest Results")
    print("=" * 40)
    print(f"Loss:     {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"IoU:      {metrics['iou']:.4f}")

    # Save alongside checkpoint
    output_path = (
        args.checkpoint.parent.parent
        / "test_metrics.json"
    )

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()