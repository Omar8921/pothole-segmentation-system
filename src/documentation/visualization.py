import torch
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Sequence

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


def denormalize_image(image: torch.Tensor):
    image = image.cpu()

    image = image * IMAGENET_STD + IMAGENET_MEAN

    return image.clamp(0, 1)


def save_prediction_grid(
    images,
    masks,
    predictions,
    save_dir: Path,
    epoch: int,
    num_samples: int = 4,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    images = images.cpu()
    masks = masks.cpu()
    predictions = predictions.cpu()

    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(
        num_samples,
        3,
        figsize=(12, 4 * num_samples),
    )

    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        image = denormalize_image(images[i])
        image = image.permute(1, 2, 0).numpy()

        axes[i, 0].imshow(image)

        mask = masks[i]
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        prediction = predictions[i]
        if prediction.ndim == 3:
            prediction = prediction.squeeze(0)

        axes[i, 0].imshow(image)
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 2].imshow(prediction, cmap="gray")

        axes[i, 0].set_title("Image")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].set_title("Prediction")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"epoch_{epoch:03d}.png")
    plt.close(fig)


def save_metric_curve(
    train_values: Sequence[float],
    val_values: Sequence[float],
    save_dir: Path | str,
    name: str,
    metric_name: str,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(train_values) != len(val_values):
        raise ValueError(
            f"train_values and val_values must have the same length. "
            f"Got {len(train_values)} and {len(val_values)}."
        )

    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, label=f"Train {metric_name}")
    plt.plot(epochs, val_values, label=f"Validation {metric_name}")

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Train vs Validation {metric_name}")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_dir / name, dpi=300, bbox_inches="tight")
    plt.close()