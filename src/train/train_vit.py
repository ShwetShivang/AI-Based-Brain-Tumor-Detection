"""
Phase 4: Train Vision Transformer (ViT-Base) on the Nickparvar 4-class brain tumor dataset.

Usage:
    python -m src.train.train_vit --config configs/vit_classifier.yaml
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str, base_config_path: str = None) -> dict:
    config = {}
    if base_config_path and Path(base_config_path).exists():
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f) or {}
    for key, val in exp_config.items():
        if isinstance(val, dict) and key in config and isinstance(config[key], dict):
            config[key].update(val)
        else:
            config[key] = val
    return config


def param_groups_lrd(model, lr, weight_decay, layer_decay=0.65):
    """Layer-wise learning rate decay for ViT."""
    param_groups = []
    # Determine number of layers
    num_layers = 12  # ViT-Base default

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer id
        if 'blocks' in name:
            try:
                parts = name.split('.')
                block_idx = None
                for k, part in enumerate(parts):
                    if part == 'blocks' and k + 1 < len(parts):
                        block_idx = int(parts[k + 1])
                        break
                layer_id = block_idx + 1 if block_idx is not None else 1
            except (ValueError, IndexError):
                layer_id = 1
        elif 'head' in name or 'classifier' in name:
            layer_id = num_layers
        elif 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
            layer_id = 0
        else:
            layer_id = 0

        this_lr = lr * (layer_decay ** (num_layers - layer_id))
        this_wd = 0.0 if ('bias' in name or 'norm' in name) else weight_decay

        param_groups.append({
            'params': [param],
            'lr': this_lr,
            'weight_decay': this_wd,
            'name': name,
        })

    return param_groups


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT Classifier")
    parser.add_argument("--config", type=str, default="configs/vit_classifier.yaml")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    base_config_path = project_root / "configs" / "base.yaml"
    config = load_config(args.config, str(base_config_path))

    seed = config.get('project', {}).get('seed', 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_dir = project_root / config.get('datasets', {}).get('nickparvar', {}).get('raw_dir', 'data/raw/nickparvar')
    checkpoint_dir = project_root / config.get('paths', {}).get('checkpoints', 'checkpoints')
    results_dir = project_root / 'results' / 'phase4'
    runs_dir = results_dir / 'runs' / 'vit_classifier'

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(runs_dir))

    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    ds_cfg = config.get('datasets', {}).get('nickparvar', {})

    from src.data.classification_dataset import BrainTumorClassificationDataset
    from src.models.vit import BrainTumorViT

    image_size = ds_cfg.get('image_size', 224)
    val_ratio = ds_cfg.get('val_split', 0.15)

    train_dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir), split='train', image_size=image_size,
        val_ratio=val_ratio, seed=seed
    )
    val_dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir), split='val', image_size=image_size,
        val_ratio=val_ratio, seed=seed
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    num_workers = config.get('project', {}).get('num_workers', 4)
    batch_size = train_cfg.get('batch_size', 32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = BrainTumorViT(
        model_name=model_cfg.get('name', 'vit_base_patch16_224'),
        num_classes=model_cfg.get('num_classes', 4),
        pretrained=model_cfg.get('pretrained', True),
        dropout=model_cfg.get('dropout', 0.1)
    ).to(device)

    total_params, trainable_params = model.count_parameters()
    logger.info(f"ViT params: {total_params:,} total, {trainable_params:,} trainable")

    # Loss with label smoothing
    label_smoothing = train_cfg.get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Layer-wise LR decay
    lr = train_cfg.get('learning_rate', 5e-5)
    weight_decay = train_cfg.get('weight_decay', 0.01)
    layer_decay = train_cfg.get('layer_decay', 0.65)

    param_groups = param_groups_lrd(model, lr=lr, weight_decay=weight_decay, layer_decay=layer_decay)
    optimizer = optim.AdamW(param_groups)

    epochs = train_cfg.get('epochs', 25)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = config.get('project', {}).get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    patience = train_cfg.get('early_stopping_patience', 7)
    checkpoint_every = train_cfg.get('checkpoint_every', 5)
    mixup_alpha = train_cfg.get('mixup_alpha', 0.2)

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = ckpt.get('epoch', 0)
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        logger.info(f"Resumed from epoch {start_epoch}")

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # === Training ===
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Mixup
            if mixup_alpha > 0:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha, device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    if mixup_alpha > 0:
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if mixup_alpha > 0:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

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

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}/{epochs} ({elapsed:.0f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        ckpt_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'config': config,
        }
        if scaler:
            ckpt_data['scaler_state_dict'] = scaler.state_dict()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_data['best_val_f1'] = best_val_f1
            patience_counter = 0
            torch.save(ckpt_data, checkpoint_dir / "vit_best.pth")
            logger.info(f"  -> New best! F1: {val_f1:.4f}")
        else:
            patience_counter += 1

        if (epoch + 1) % checkpoint_every == 0:
            torch.save(ckpt_data, checkpoint_dir / f"vit_epoch_{epoch + 1}.pth")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Final eval
    best_ckpt = torch.load(checkpoint_dir / "vit_best.pth", map_location=device, weights_only=False)
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('ViT Confusion Matrix')
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    final_metrics = {
        'best_val_accuracy': float(final_acc),
        'best_val_f1_weighted': float(final_f1),
        'best_epoch': int(best_ckpt['epoch']),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_name': model_cfg.get('name', 'vit_base_patch16_224'),
        'confusion_matrix': cm.tolist(),
        'classes': classes,
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    writer.close()
    logger.info("=" * 60)
    logger.info(f"ViT TRAINING COMPLETE | Acc: {final_acc:.4f}, F1: {final_f1:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
