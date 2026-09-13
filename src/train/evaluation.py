import torch
from torch.utils.data import DataLoader

from typing import Tuple

from .metrics import calculate_accuracy, calculate_iou

@torch.no_grad()
def eval_batch(
    model,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion,
) -> Tuple[float, float, float]:

    model.eval()

    with torch.amp.autocast('cuda'):
        outputs = model(images)
        loss = criterion(outputs, masks)
    
    acc = calculate_accuracy(outputs.detach(), masks)
    iou = calculate_iou(outputs.detach(), masks)

    return loss.item(), acc.item(), iou.item()


def evaluate_model(
    model,
    data_loader: DataLoader,
    criterion,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, masks in data_loader:
            images = images.to(
                device,
                non_blocking=True,
            ).float()

            masks = masks.to(
                device,
                non_blocking=True,
            ).unsqueeze(1).float()

            loss, acc, iou = eval_batch(
                model=model,
                images=images,
                masks=masks,
                criterion=criterion,
            )

            batch_size = images.size(0)

            total_loss += loss * batch_size
            total_acc += acc * batch_size
            total_iou += iou * batch_size

            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_acc / total_samples,
        "iou": total_iou / total_samples,
    }