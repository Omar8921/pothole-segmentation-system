from pathlib import Path
import pandas as pd
import json

def make_experiment(
    dir: Path,
    name: str,
    config: dict      
) -> Path:
    # Ensure parent directory exists 
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    # Create experiment directory
    experiment_dir = dir / name
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir / 'checkpoints'
    plots_dir = experiment_dir / 'plots'
    grids_dir = experiment_dir / 'grids'

    checkpoints_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    grids_dir.mkdir(exist_ok=True)

    history = pd.DataFrame(columns=['epoch', 'train_loss', 'train_accuracy', 'train_iou', 'val_loss', 'val_accuracy', 'val_iou'])
    history.to_csv(experiment_dir / 'history.csv', index=False)

    with open(experiment_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return experiment_dir