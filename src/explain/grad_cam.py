"""
Phase 3: Grad-CAM explainability using Captum.
Generates heatmap overlays for correct and incorrect predictions.
Includes sanity check: verifies heatmaps localize to tumor regions.

Usage:
    python -m src.explain.grad_cam --checkpoint checkpoints/cnn_best.pth
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from captum.attr import LayerGradCam, LayerAttribution
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_target_layer(model, model_name: str):
    """Get the target convolutional layer for Grad-CAM."""
    if 'efficientnet' in model_name:
        # For timm EfficientNet, target the last conv features
        return model.model.conv_head
    elif 'resnet' in model_name:
        return model.model.layer4[-1]
    elif 'densenet' in model_name:
        return model.model.features[-1]
    else:
        # Fallback: try common names
        if hasattr(model.model, 'conv_head'):
            return model.model.conv_head
        elif hasattr(model.model, 'layer4'):
            return model.model.layer4[-1]
        raise ValueError(f"Cannot determine target layer for model: {model_name}")


def overlay_heatmap(image_np, heatmap_np, alpha=0.4):
    """Overlay a Grad-CAM heatmap on an image."""
    # image_np: (H, W, 3) float [0, 1]
    # heatmap_np: (H, W) float [0, 1]
    colormap = cm.jet(heatmap_np)[:, :, :3]  # (H, W, 3)
    overlay = (1 - alpha) * image_np + alpha * colormap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize a tensor image."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor.cpu() * std + mean


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/cnn_best.pth')
    parser.add_argument('--model-type', type=str, default='cnn', choices=['cnn', 'vit'])
    parser.add_argument('--output-dir', type=str, default='results/phase3')
    parser.add_argument('--num-correct', type=int, default=10)
    parser.add_argument('--num-incorrect', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / args.output_dir
    correct_dir = output_dir / 'correct_predictions'
    incorrect_dir = output_dir / 'incorrect_predictions'
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(project_root / 'configs' / 'base.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    ds_cfg = base_config.get('datasets', {}).get('nickparvar', {})
    data_dir = project_root / ds_cfg.get('raw_dir', 'data/raw/nickparvar')

    # Load dataset (test split)
    from src.data.classification_dataset import BrainTumorClassificationDataset
    dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir), split='test',
        image_size=ds_cfg.get('image_size', 224)
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    classes = list(BrainTumorClassificationDataset.class_to_idx.keys())

    # Load model
    checkpoint = torch.load(project_root / args.checkpoint, map_location=device, weights_only=False)

    if args.model_type == 'cnn':
        from src.models.cnn import BrainTumorCNN
        ckpt_config = checkpoint.get('config', {}).get('model', {})
        model_name = ckpt_config.get('name', 'efficientnet_b0')
        model = BrainTumorCNN(
            model_name=model_name,
            num_classes=ckpt_config.get('num_classes', 4),
            pretrained=False,
            dropout=ckpt_config.get('dropout', 0.3)
        )
    else:
        logger.warning("Grad-CAM for ViT requires attention-based methods. Using CNN model.")
        return

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get target layer for Grad-CAM
    target_layer = get_target_layer(model, model_name)
    grad_cam = LayerGradCam(model, target_layer)

    correct_count = 0
    incorrect_count = 0
    sanity_results = []

    logger.info(f"Generating Grad-CAM heatmaps: {args.num_correct} correct + {args.num_incorrect} incorrect")

    for idx, (input_tensor, target_label) in enumerate(loader):
        if correct_count >= args.num_correct and incorrect_count >= args.num_incorrect:
            break

        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        # Forward pass
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

        is_correct = (pred_class == target_label.item())

        if is_correct and correct_count >= args.num_correct:
            continue
        if not is_correct and incorrect_count >= args.num_incorrect:
            continue

        # Generate Grad-CAM attribution
        attribution = grad_cam.attribute(input_tensor, target=pred_class)

        # Upsample to input size
        attribution_upsampled = LayerAttribution.interpolate(
            attribution, (input_tensor.shape[2], input_tensor.shape[3])
        )

        # Convert to numpy heatmap
        heatmap = attribution_upsampled[0, 0].detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Denormalize image for visualization
        img_denorm = denormalize(input_tensor[0].detach())
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # Create overlay
        overlay = overlay_heatmap(img_np, heatmap, alpha=0.5)

        # Sanity check: does heatmap focus on a reasonable region?
        # Check if top 20% activation covers a concentrated area
        threshold = 0.8
        high_activation_mask = (heatmap > threshold).astype(float)
        activation_fraction = high_activation_mask.mean()
        localized = 0.01 < activation_fraction < 0.5  # should be a focused region, not everywhere

        true_name = classes[target_label.item()]
        pred_name = classes[pred_class]

        # Save
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title('Original MRI', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f'True: {true_name} | Pred: {pred_name}\nConf: {confidence:.2%}', fontsize=11)
        axes[2].axis('off')

        plt.tight_layout()

        if is_correct:
            save_path = correct_dir / f'gradcam_correct_{correct_count:03d}.png'
            correct_count += 1
        else:
            save_path = incorrect_dir / f'gradcam_incorrect_{incorrect_count:03d}.png'
            incorrect_count += 1

        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

        sanity_results.append({
            'sample_idx': idx,
            'true_class': true_name,
            'pred_class': pred_name,
            'confidence': float(confidence),
            'correct': bool(is_correct),
            'heatmap_localized': bool(localized),
            'activation_fraction': float(activation_fraction),
        })

        if not localized and is_correct:
            logger.warning(
                f"⚠ Sanity check FAILED for correct prediction (sample {idx}): "
                f"heatmap doesn't appear to localize to a specific region "
                f"(activation fraction: {activation_fraction:.3f})"
            )

    # Save sanity check results
    with open(output_dir / 'sanity_check_results.json', 'w') as f:
        json.dump(sanity_results, f, indent=4)

    # Create summary grid
    correct_images = sorted(correct_dir.glob('*.png'))[:6]
    incorrect_images = sorted(incorrect_dir.glob('*.png'))[:3]
    all_grid_images = correct_images + incorrect_images

    if all_grid_images:
        n = len(all_grid_images)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes_flat = np.array(axes).flatten()

        for i, img_path in enumerate(all_grid_images):
            img = plt.imread(str(img_path))
            axes_flat[i].imshow(img)
            axes_flat[i].axis('off')

        for i in range(len(all_grid_images), len(axes_flat)):
            axes_flat[i].axis('off')

        plt.suptitle('Grad-CAM Summary (Top: Correct, Bottom: Incorrect)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'grad_cam_summary.png', dpi=120, bbox_inches='tight')
        plt.close()

    # Summary stats
    n_localized = sum(1 for r in sanity_results if r['heatmap_localized'] and r['correct'])
    n_correct_total = sum(1 for r in sanity_results if r['correct'])

    logger.info("=" * 60)
    logger.info("GRAD-CAM GENERATION COMPLETE")
    logger.info(f"Correct predictions:   {correct_count}")
    logger.info(f"Incorrect predictions: {incorrect_count}")
    logger.info(f"Localization sanity:   {n_localized}/{n_correct_total} correct predictions show focused heatmaps")
    logger.info(f"Results saved to:      {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
