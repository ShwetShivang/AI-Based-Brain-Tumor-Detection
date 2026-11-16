"""
Phase 2: Train U-Net segmentation model on BraTS 2021 dataset.

Usage:
    python -m src.train.train_unet --config configs/unet_segmentation.yaml
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(pred, target, smooth=1e-6):
    """Differentiable Dice loss."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def dice_coeff(pred, target, threshold=0.5, smooth=1e-6):
    """Non-differentiable Dice coefficient for evaluation."""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_coeff(pred, target, threshold=0.5, smooth=1e-6):
    """IoU (Jaccard Index) for evaluation."""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def save_overlay_images(model, dataloader, device, output_dir, num_samples=10, use_amp=True):
    """Save overlay visualizations of predictions vs ground truth."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for inputs, masks in dataloader:
            if saved >= num_samples:
                break
            inputs, masks = inputs.to(device), masks.to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            pred_masks = (torch.sigmoid(outputs) > 0.5).float()

            for b in range(inputs.size(0)):
                if saved >= num_samples:
                    break

                img = inputs[b, 0].cpu().numpy()
                gt_mask = masks[b, 0].cpu().numpy()
                pred_mask = pred_masks[b, 0].cpu().numpy()

                d = dice_coeff(outputs[b:b+1], masks[b:b+1]).item()
                iou = iou_coeff(outputs[b:b+1], masks[b:b+1]).item()

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('MRI Slice')
                axes[0].axis('off')

                axes[1].imshow(gt_mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                # Overlay
                axes[3].imshow(img, cmap='gray')
                axes[3].imshow(pred_mask, cmap='Reds', alpha=0.4)
                axes[3].contour(gt_mask, colors='lime', linewidths=1)
                axes[3].set_title(f'Overlay (Dice: {d:.3f}, IoU: {iou:.3f})')
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(output_dir / f'overlay_{saved:03d}.png', dpi=120, bbox_inches='tight')
                plt.close()
                saved += 1

    logger.info(f"Saved {saved} overlay images to {output_dir}")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net Segmentation for Brain Tumors")
    parser.add_argument("--config", type=str, default="configs/unet_segmentation.yaml")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    base_config_path = project_root / "configs" / "base.yaml"
    config = load_config(args.config, str(base_config_path))

    seed = config.get('project', {}).get('seed', 42)
    set_seed(seed)

    device_cfg = config.get('project', {}).get('device', 'auto')
    device = torch.device("cuda" if (device_cfg == 'auto' and torch.cuda.is_available()) else device_cfg if device_cfg != 'auto' else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    data_dir = project_root / config.get('datasets', {}).get('brats2021', {}).get('raw_dir', 'data/raw/brats2021')
    checkpoint_dir = project_root / config.get('paths', {}).get('checkpoints', 'checkpoints')
    results_dir = project_root / 'results' / 'phase2'
    runs_dir = results_dir / 'runs' / 'unet_segmentation'

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(runs_dir))

    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    ds_cfg = config.get('datasets', {}).get('brats2021', {})

    from src.data.segmentation_dataset import BraTSSegmentationDataset
    from src.models.unet import BrainTumorUNet

    image_size = ds_cfg.get('image_size', 256)
    modality = ds_cfg.get('primary_modality', 'flair')
    min_tumor_frac = data_cfg.get('min_tumor_fraction', 0.001)

    train_dataset = BraTSSegmentationDataset(
        root_dir=str(data_dir),
        split='train',
        image_size=image_size,
        modality=modality,
        min_tumor_fraction=min_tumor_frac
    )
    val_dataset = BraTSSegmentationDataset(
        root_dir=str(data_dir),
        split='val',
        image_size=image_size,
        modality=modality,
        min_tumor_fraction=min_tumor_frac
    )

    logger.info(f"Train slices: {len(train_dataset)}")
    logger.info(f"Val slices: {len(val_dataset)}")

    num_workers = config.get('project', {}).get('num_workers', 4)
    batch_size = train_cfg.get('batch_size', 16)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Model
    model = BrainTumorUNet(
        encoder_name=model_cfg.get('encoder', 'resnet34'),
        encoder_pretrained=model_cfg.get('encoder_pretrained', True),
        in_channels=model_cfg.get('in_channels', 1),
        out_channels=model_cfg.get('out_channels', 1)
    ).to(device)

    total_params, trainable_params = model.count_parameters()
    logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    bce_weight = train_cfg.get('bce_weight', 0.5)
    dice_weight_cfg = train_cfg.get('dice_weight', 0.5)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer
    lr = train_cfg.get('learning_rate', 1e-4)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # AMP
    use_amp = config.get('project', {}).get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Resume
    epochs = train_cfg.get('epochs', 30)
    start_epoch = 0
    best_val_dice = 0.0
    patience_counter = 0
    patience = train_cfg.get('early_stopping_patience', 10)
    checkpoint_every = train_cfg.get('checkpoint_every', 5)
    grad_clip = 1.0

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = ckpt.get('epoch', 0)
        best_val_dice = ckpt.get('best_val_dice', 0.0)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        logger.info(f"Resumed at epoch {start_epoch}")

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # === Training ===
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    l_bce = bce_loss_fn(outputs, targets)
                    l_dice = dice_loss(outputs, targets)
                    loss = bce_weight * l_bce + dice_weight_cfg * l_dice
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                l_bce = bce_loss_fn(outputs, targets)
                l_dice = dice_loss(outputs, targets)
                loss = bce_weight * l_bce + dice_weight_cfg * l_dice
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        l_bce = bce_loss_fn(outputs, targets)
                        l_dice = dice_loss(outputs, targets)
                        loss = bce_weight * l_bce + dice_weight_cfg * l_dice
                else:
                    outputs = model(inputs)
                    l_bce = bce_loss_fn(outputs, targets)
                    l_dice = dice_loss(outputs, targets)
                    loss = bce_weight * l_bce + dice_weight_cfg * l_dice

                val_loss += loss.item()
                val_dice_sum += dice_coeff(outputs, targets).item()
                val_iou_sum += iou_coeff(outputs, targets).item()

        val_loss /= len(val_loader)
        val_dice = val_dice_sum / len(val_loader)
        val_iou = val_iou_sum / len(val_loader)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/Dice', val_dice, epoch)
        writer.add_scalar('Metrics/IoU', val_iou, epoch)

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch + 1}/{epochs} ({elapsed:.0f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}"
        )

        # Checkpointing
        ckpt_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
            'config': config,
        }
        if scaler:
            ckpt_data['scaler_state_dict'] = scaler.state_dict()

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_data['best_val_dice'] = best_val_dice
            patience_counter = 0
            torch.save(ckpt_data, checkpoint_dir / "unet_best.pth")
            logger.info(f"  -> New best! Dice: {val_dice:.4f}")
        else:
            patience_counter += 1

        if (epoch + 1) % checkpoint_every == 0:
            torch.save(ckpt_data, checkpoint_dir / f"unet_epoch_{epoch + 1}.pth")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # === Final: save overlays ===
    logger.info("Generating overlay images from best model...")
    best_ckpt = torch.load(checkpoint_dir / "unet_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    save_overlay_images(model, val_loader, device, results_dir / 'overlays', num_samples=12, use_amp=use_amp)

    # Save metrics
    final_metrics = {
        'best_val_dice': float(best_val_dice),
        'best_val_iou': float(max(history['val_iou'])) if history['val_iou'] else 0.0,
        'best_epoch': int(best_ckpt.get('epoch', 0)),
        'total_params': total_params,
        'trainable_params': trainable_params,
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()

    axes[1].plot(history['val_dice'], label='Dice')
    axes[1].plot(history['val_iou'], label='IoU')
    axes[1].set_title('Segmentation Metrics')
    axes[1].legend()

    axes[2].plot(history['val_dice'])
    axes[2].set_title('Val Dice Over Epochs')
    axes[2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    writer.close()

    logger.info("=" * 60)
    logger.info("SEGMENTATION TRAINING COMPLETE")
    logger.info(f"Best Dice: {best_val_dice:.4f}")
    logger.info(f"Checkpoint: {checkpoint_dir / 'unet_best.pth'}")
    logger.info(f"Results: {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
