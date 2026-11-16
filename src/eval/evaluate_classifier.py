"""
Phase 1/4: Evaluate a trained classifier on the test set.
Produces metrics JSON, confusion matrix, and classification report.

Usage:
    python -m src.eval.evaluate_classifier --checkpoint checkpoints/cnn_best.pth --model-type cnn
    python -m src.eval.evaluate_classifier --checkpoint checkpoints/vit_best.pth --model-type vit
"""
import argparse
import json
import time
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, f1_score
)
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'vit'], default='cnn')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--split', type=str, default='test', help='Data split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent

    # Auto-detect output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        phase = 'phase1' if args.model_type == 'cnn' else 'phase4'
        output_dir = project_root / 'results' / phase

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(project_root / 'configs' / 'base.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    ds_cfg = base_config.get('datasets', {}).get('nickparvar', {})
    data_dir = project_root / ds_cfg.get('raw_dir', 'data/raw/nickparvar')
    image_size = ds_cfg.get('image_size', 224)

    # Load dataset
    from src.data.classification_dataset import BrainTumorClassificationDataset

    dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir),
        split=args.split,
        image_size=image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    classes = list(BrainTumorClassificationDataset.class_to_idx.keys())

    logger.info(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.model_type == 'cnn':
        from src.models.cnn import BrainTumorCNN
        ckpt_config = checkpoint.get('config', {}).get('model', {})
        model = BrainTumorCNN(
            model_name=ckpt_config.get('name', 'efficientnet_b0'),
            num_classes=ckpt_config.get('num_classes', 4),
            pretrained=False,
            dropout=ckpt_config.get('dropout', 0.3)
        )
    else:
        from src.models.vit import BrainTumorViT
        ckpt_config = checkpoint.get('config', {}).get('model', {})
        model = BrainTumorViT(
            model_name=ckpt_config.get('name', 'vit_base_patch16_224'),
            num_classes=ckpt_config.get('num_classes', 4),
            pretrained=False,
            dropout=ckpt_config.get('dropout', 0.1)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    total_params, _ = model.count_parameters()

    # Evaluate
    all_preds = []
    all_targets = []
    all_probs = []

    start_time = time.time()
    num_images = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            num_images += inputs.size(0)

    total_time = time.time() - start_time
    inference_time_ms = (total_time / num_images) * 1000

    # Metrics
    overall_acc = accuracy_score(all_targets, all_preds)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    cm = confusion_matrix(all_targets, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    report_dict = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    report_str = classification_report(all_targets, all_preds, target_names=classes)

    metrics = {
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': num_images,
        'overall_accuracy': float(overall_acc),
        'weighted_f1': float(f1_weighted),
        'macro_f1': float(f1_macro),
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'inference_time_ms_per_image': float(inference_time_ms),
        'total_params': total_params,
        'per_class_accuracy': {c: float(a) for c, a in zip(classes, per_class_acc)},
        'classification_report': report_dict,
    }

    # Save metrics
    prefix = args.model_type
    with open(output_dir / f'{prefix}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / f'{prefix}_classification_report.txt', 'w') as f:
        f.write(f"Model: {args.model_type.upper()}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}, Samples: {num_images}\n\n")
        f.write(report_str)

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'{args.model_type.upper()} Confusion Matrix ({args.split})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print summary
    logger.info("=" * 60)
    logger.info(f"EVALUATION SUMMARY ({args.model_type.upper()})")
    logger.info(f"Accuracy:     {overall_acc:.4f}")
    logger.info(f"Weighted F1:  {f1_weighted:.4f}")
    logger.info(f"Macro F1:     {f1_macro:.4f}")
    logger.info(f"Inference:    {inference_time_ms:.2f} ms/image")
    logger.info(f"Parameters:   {total_params:,}")
    logger.info("=" * 60)
    logger.info(f"\n{report_str}")


if __name__ == '__main__':
    main()
