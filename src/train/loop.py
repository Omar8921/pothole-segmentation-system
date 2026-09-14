import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from src.inference import predict_batch
from src.train.training import train_batch
from src.train.evaluation import eval_batch
from src.documentation.experiment import make_experiment
from src.documentation.visualization import (
    save_metric_curve,
    save_prediction_grid,
)


def train(
    epochs: int,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    patience: int,
    experiments_dir: Path,
    experiment_name: str,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
):
    # ---------------------------------------------------------
    # Experiment setup
    # ---------------------------------------------------------
    experiment_dir = make_experiment(
        experiments_dir,
        experiment_name,
        config,
    )

    df_path = experiment_dir / "history.csv"
    history_df = pd.read_csv(df_path)

    train_losses = []
    train_accs = []
    train_ious = []

    val_losses = []
    val_accs = []
    val_ious = []

    patience_counter = 0
    best_val_loss = np.inf

    # ---------------------------------------------------------
    # Epoch progress bar
    # ---------------------------------------------------------
    epoch_pbar = tqdm(
        range(epochs),
        desc="Epochs",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    for epoch in epoch_pbar:

        # =====================================================
        # TRAINING
        # =====================================================
        train_loss = 0.0
        train_acc = 0.0
        train_iou = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{epochs}",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (images, masks) in enumerate(train_pbar):
            images = images.to(
                device,
                non_blocking=True,
            ).float()

            masks = masks.to(
                device,
                non_blocking=True,
            ).unsqueeze(1).float()

            loss, acc, iou = train_batch(
                model=model,
                images=images,
                masks=masks,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
            )

            train_loss += loss
            train_acc += acc
            train_iou += iou

            # Running averages
            running_loss = train_loss / (batch_idx + 1)
            running_acc = train_acc / (batch_idx + 1)
            running_iou = train_iou / (batch_idx + 1)

            train_pbar.set_postfix(
                loss=f"{running_loss:.3f}",
                acc=f"{running_acc:.3f}",
                iou=f"{running_iou:.3f}",
            )

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_iou /= len(train_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_ious.append(train_iou)

        # =====================================================
        # VALIDATION
        # =====================================================
        val_loss = 0.0
        val_acc = 0.0
        val_iou = 0.0

        val_pbar = tqdm(
            val_loader,
            desc=f"Val   {epoch + 1}/{epochs}",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (images, masks) in enumerate(val_pbar):
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

            val_loss += loss
            val_acc += acc
            val_iou += iou

            # Running averages
            running_loss = val_loss / (batch_idx + 1)
            running_acc = val_acc / (batch_idx + 1)
            running_iou = val_iou / (batch_idx + 1)

            val_pbar.set_postfix(
                loss=f"{running_loss:.3f}",
                acc=f"{running_acc:.3f}",
                iou=f"{running_iou:.3f}",
            )

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_iou /= len(val_loader)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_ious.append(val_iou)

        # =====================================================
        # EARLY STOPPING
        # =====================================================
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # -----------------------------------------------------
        # Update main epoch progress bar
        # -----------------------------------------------------
        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.3f}",
            val_loss=f"{val_loss:.3f}",
            train_iou=f"{train_iou:.3f}",
            val_iou=f"{val_iou:.3f}",
            patience=f"{patience_counter}/{patience}",
        )

        # -----------------------------------------------------
        # Print epoch summary
        # -----------------------------------------------------
        tqdm.write(f"\nEpoch #{epoch + 1}")

        tqdm.write(
            f"Train Loss: {train_loss:.3f}, "
            f"Train Acc: {train_acc:.3f}, "
            f"Train IoU: {train_iou:.3f}"
        )

        tqdm.write(
            f"Val Loss: {val_loss:.3f}, "
            f"Val Acc: {val_acc:.3f}, "
            f"Val IoU: {val_iou:.3f}"
        )

        # =====================================================
        # SAVE HISTORY
        # =====================================================
        entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 3),
            "train_accuracy": round(train_acc, 3),
            "train_iou": round(train_iou, 3),
            "val_loss": round(val_loss, 3),
            "val_accuracy": round(val_acc, 3),
            "val_iou": round(val_iou, 3),
        }

        history_df.loc[len(history_df)] = entry
        history_df.to_csv(
            df_path,
            index=False,
        )

        # =====================================================
        # VISUALIZE PREDICTIONS
        # =====================================================
        vis_images, vis_masks = next(iter(val_loader))

        preds_batch = predict_batch(
            model=model,
            images=vis_images,
            device=device,
        )

        save_prediction_grid(
            images=vis_images,
            masks=vis_masks,
            predictions=preds_batch,
            save_dir=experiment_dir / "grids",
            epoch=epoch + 1,
        )

        # =====================================================
        # SAVE METRIC CURVES
        # =====================================================
        save_metric_curve(
            train_accs,
            val_accs,
            experiment_dir / "plots",
            "accuracy_curve.png",
            "Accuracy",
        )

        save_metric_curve(
            train_ious,
            val_ious,
            experiment_dir / "plots",
            "iou_curve.png",
            "IoU",
        )

        save_metric_curve(
            train_losses,
            val_losses,
            experiment_dir / "plots",
            "loss_curve.png",
            "Loss",
        )

        # =====================================================
        # LEARNING RATE SCHEDULER
        # =====================================================
        scheduler.step(val_loss)

        # =====================================================
        # CHECKPOINTING
        # =====================================================
        history = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "train_ious": train_ious,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "val_ious": val_ious,
        }

        checkpoint = {
            "model_weights": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch + 1,
            "history": history,
            "best_val_loss": best_val_loss,
            "config": config,
        }

        checkpoint_dir = experiment_dir / "checkpoints"

        # Always save latest checkpoint
        torch.save(
            checkpoint,
            checkpoint_dir / "last_checkpoint.pth",
        )

        # Save best checkpoint separately
        if is_best:
            torch.save(
                checkpoint,
                checkpoint_dir / "best_checkpoint.pth",
            )

            tqdm.write(
                f"Best checkpoint saved at epoch {epoch + 1}"
            )

        else:
            tqdm.write(
                f"No validation improvement. "
                f"Patience: {patience_counter}/{patience}"
            )

        tqdm.write("=" * 84)

        # =====================================================
        # EARLY STOP
        # =====================================================
        if patience_counter >= patience:
            tqdm.write(
                f"Early stopping at epoch {epoch + 1}"
            )
            break