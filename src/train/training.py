import torch
from typing import Tuple

from .metrics import calculate_accuracy, calculate_iou

def train_batch(
    model,
    images: torch.Tensor,
    masks: torch.Tensor,
    criterion,
    optimizer,
    scaler: torch.amp.GradScaler
) -> Tuple[float, float, float]:

    model.train()

    with torch.amp.autocast('cuda'):
        outputs = model(images)
        loss = criterion(outputs, masks)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    acc = calculate_accuracy(outputs.detach(), masks)
    iou = calculate_iou(outputs.detach(), masks)

    return loss.item(), acc.item(), iou.item()