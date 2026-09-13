import torch

def calculate_iou(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    if from_logits:
        y_pred = torch.sigmoid(y_pred)

    y_true = y_true.float()
    y_pred = y_pred.float()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_pred has shape {y_pred.shape}"
        )

    # Convert probabilities to hard binary mask
    y_pred = (y_pred >= threshold).float()

    dims = (1, 2, 3)

    intersection = torch.sum(y_pred * y_true, dim=dims)

    union = (
        torch.sum(y_pred, dim=dims)
        + torch.sum(y_true, dim=dims)
        - intersection
    )

    iou = (intersection + eps) / (union + eps)

    return iou.mean()

def calculate_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
):
    if from_logits:
        y_pred = torch.sigmoid(y_pred)

    y_true = y_true.float()
    y_pred = y_pred.float()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_pred has shape {y_pred.shape}"
        )

    y_pred = (y_pred > threshold).float()

    n_correct = (y_pred == y_true).sum()
    n_total = torch.numel(y_true)

    return n_correct / n_total