"""
Phase 1: Train CNN classifier (EfficientNet-B0) on the Nickparvar 4-class brain tumor dataset.

Usage:
    python -m src.train.train_cnn --config configs/cnn_classifier.yaml
"""
import os
import argparse
import yaml
import json
import time
import random
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, base_config_path: str = None) -> dict:
    """Load experiment config, merged with base config."""
    config = {}
    if base_config_path and Path(base_config_path).exists():
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f) or {}

    # Merge: experiment config overrides base config
    for key, val in exp_config.items():
        if isinstance(val, dict) and key in config and isinstance(config[key], dict):
            config[key].update(val)
        else:
            config[key] = val
    return config


def save_confusion_matrix(cm: np.ndarray, classes: list, output_path: Path):
    """Save confusion matrix as a heatmap image."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN Classifier for Brain Tumor Detection")
    parser.add_argument("--config", type=str, default="configs/cnn_classifier.yaml",
                        help="Path to experiment config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configs
    project_root = Path(__file__).resolve().parent.parent.parent
    base_config_path = project_root / "configs" / "base.yaml"
    config = load_config(args.config, str(base_config_path))

    seed = config.get('project', {}).get('seed', 42)
    set_seed(seed)

    # Device setup
    device_cfg = config.get('project', {}).get('device', 'auto')
    if device_cfg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info(f"Using device: {device}")

    # Paths
    data_dir = project_root / config.get('datasets', {}).get('nickparvar', {}).get('raw_dir', 'data/raw/nickparvar')
    checkpoint_dir = project_root / config.get('paths', {}).get('checkpoints', 'checkpoints')
    results_dir = project_root / 'results' / 'phase1'
    runs_dir = results_dir / 'runs' / 'cnn_classifier'

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(runs_dir))

    # Model config
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})

    # Import here after path setup
    from src.data.classification_dataset import BrainTumorClassificationDataset
    from src.models.cnn import BrainTumorCNN

    # Dataset config
    ds_cfg = config.get('datasets', {}).get('nickparvar', {})
    image_size = ds_cfg.get('image_size', 224)
    val_ratio = ds_cfg.get('val_split', 0.15)

    # Create datasets
    train_dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir),
        split='train',
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed
    )
    val_dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir),
        split='val',
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Class distribution (train): {train_dataset.get_sample_count_per_class()}")

    # DataLoaders
    num_workers = config.get('project', {}).get('num_workers', 4)
    batch_size = train_cfg.get('batch_size', 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Model
    model = BrainTumorCNN(
        model_name=model_cfg.get('name', 'efficientnet_b0'),
        num_classes=model_cfg.get('num_classes', 4),
        pretrained=model_cfg.get('pretrained', True),
        dropout=model_cfg.get('dropout', 0.3)
    ).to(device)

    total_params, trainable_params = model.count_parameters()
    logger.info(f"Model: {model_cfg.get('name', 'efficientnet_b0')}")
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss with class weights
    class_weights = train_dataset.get_class_weights().to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    lr = train_cfg.get('learning_rate', 1e-4)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    epochs = train_cfg.get('epochs', 25)
    warmup_epochs = train_cfg.get('warmup_epochs', 2)

    # Cosine annealing after warmup
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )

    # Mixed precision
    use_amp = config.get('project', {}).get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Resume from checkpoint
    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = checkpoint.get('epoch', 0)
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            main_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f"Resumed from epoch {start_epoch}, best val F1: {best_val_f1:.4f}")

    # Training
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}
    freeze_epochs = model_cfg.get('freeze_backbone_epochs', 3)
    patience = train_cfg.get('early_stopping_patience', 7)
    checkpoint_every = train_cfg.get('checkpoint_every', 5)
    grad_clip = train_cfg.get('gradient_clip_max_norm', 1.0)

    # Freeze backbone initially
    if start_epoch < freeze_epochs:
        model.freeze_backbone()
        logger.info(f"Backbone frozen for first {freeze_epochs} epochs")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Unfreeze backbone at the right epoch
        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            logger.info("Backbone unfrozen — all parameters are now trainable")

        # === Training ===
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        # Step scheduler (skip during warmup — use linear warmup instead)
        if epoch >= warmup_epochs:
            main_scheduler.step()

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        current_lr = optimizer.param_groups[0]['lr']

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_preds, val_targets_list = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets_list, val_preds)
        val_f1 = f1_score(val_targets_list, val_preds, average='weighted')

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}/{epochs} ({elapsed:.0f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # === Checkpointing ===
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': main_scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'config': config,
        }
        if scaler:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_data['best_val_f1'] = best_val_f1
            patience_counter = 0
            torch.save(checkpoint_data, checkpoint_dir / "cnn_best.pth")
            logger.info(f"  -> New best model! Val F1: {val_f1:.4f}")
        else:
            patience_counter += 1

        if (epoch + 1) % checkpoint_every == 0:
            torch.save(checkpoint_data, checkpoint_dir / f"cnn_epoch_{epoch + 1}.pth")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    # === Final evaluation on best model ===
    logger.info("Loading best model for final evaluation...")
    best_ckpt = torch.load(checkpoint_dir / "cnn_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    final_preds, final_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            final_preds.extend(predicted.cpu().numpy())
            final_targets.extend(targets.cpu().numpy())

    final_acc = accuracy_score(final_targets, final_preds)
    final_f1 = f1_score(final_targets, final_preds, average='weighted')
    cm = confusion_matrix(final_targets, final_preds)
    classes = list(BrainTumorClassificationDataset.class_to_idx.keys())

    # Save confusion matrix
    save_confusion_matrix(cm, classes, results_dir / 'confusion_matrix.png')

    # Save metrics
    final_metrics = {
        'best_val_accuracy': float(final_acc),
        'best_val_f1_weighted': float(final_f1),
        'best_epoch': int(best_ckpt['epoch']),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_name': model_cfg.get('name', 'efficientnet_b0'),
        'confusion_matrix': cm.tolist(),
        'classes': classes,
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

    # Save training history
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    # Save training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()

    axes[1].plot(history['val_acc'], label='Accuracy')
    axes[1].plot(history['val_f1'], label='F1 (weighted)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()

    axes[2].plot(history['lr'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    writer.close()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Val Accuracy: {final_acc:.4f}")
    logger.info(f"Best Val F1 (weighted): {final_f1:.4f}")
    logger.info(f"Best epoch: {best_ckpt['epoch']}")
    logger.info(f"Checkpoint: {checkpoint_dir / 'cnn_best.pth'}")
    logger.info(f"Results: {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
