import torch
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        y_true = y_true.float()
        y_pred = y_pred.float()

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, "
                f"but y_pred has shape {y_pred.shape}"
            )
        
        dims = (1, 2, 3)

        intersection = torch.sum(y_true * y_pred, dim=dims)
        denominator = torch.sum(y_true, dim=dims) + torch.sum(y_pred, dim=dims)

        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)

        return (1 - dice).mean()
    
class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1e-6,
        from_logits: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.from_logits = from_logits

        if from_logits:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCELoss()

        self.dice = BinaryDiceLoss(
            smooth=smooth,
            from_logits=from_logits,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = y_true.float()

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, "
                f"but y_pred has shape {y_pred.shape}"
            )

        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)

        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss
