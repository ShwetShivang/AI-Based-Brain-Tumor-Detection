import argparse
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def dice_coeff(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_coeff(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def evaluate(args):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dummy dataloader setup for segmentation
    # Real setup would need custom dataset returning image and mask
    print("WARNING: Using dummy dataloader for segmentation. Ensure real dataloader is implemented.")
    num_samples = args.num_samples
    
    # Dummy Unet model setup
    import torchvision
    model = torchvision.models.segmentation.fcn_resnet50(num_classes=1).to(device)
    
    if args.checkpoint and Path(args.checkpoint).exists():
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
            print(f"Loaded checkpoint {args.checkpoint}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            
    model.eval()
    
    dices = []
    ious = []
    pixel_accs = []
    
    # Simulate processing and saving overlays
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i in range(num_samples):
        # Fake data
        img = torch.rand(1, 3, 224, 224).to(device)
        target_mask = (torch.rand(1, 1, 224, 224) > 0.5).float().to(device)
        
        with torch.no_grad():
            out = model(img)['out']
            pred_mask = (torch.sigmoid(out) > 0.5).float()
            
        dice = dice_coeff(out, target_mask).item()
        iou = iou_coeff(out, target_mask).item()
        acc = (pred_mask == target_mask).float().mean().item()
        
        dices.append(dice)
        ious.append(iou)
        pixel_accs.append(acc)
        
        # Plotting
        ax = axes[i]
        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        t_mask_np = target_mask[0, 0].cpu().numpy()
        p_mask_np = pred_mask[0, 0].cpu().numpy()
        
        ax[0].imshow(img_np)
        ax[0].set_title('Original MRI')
        ax[0].axis('off')
        
        ax[1].imshow(t_mask_np, cmap='gray')
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')
        
        ax[2].imshow(p_mask_np, cmap='gray')
        ax[2].set_title('Predicted Mask')
        ax[2].axis('off')
        
        ax[3].imshow(img_np)
        ax[3].imshow(p_mask_np, cmap='jet', alpha=0.5)
        ax[3].set_title(f'Overlay (Dice: {dice:.2f})')
        ax[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'segmentation_overlays.png')
    plt.close()

    mean_dice = np.mean(dices)
    mean_iou = np.mean(ious)
    mean_acc = np.mean(pixel_accs)
    
    metrics = {
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'mean_pixel_accuracy': mean_acc,
        'per_sample_dices': dices
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    plt.figure(figsize=(8, 6))
    plt.hist(dices, bins=20, color='blue', alpha=0.7)
    plt.title('Dice Score Distribution')
    plt.xlabel('Dice Score')
    plt.ylabel('Frequency')
    plt.savefig(output_dir / 'dice_distribution.png')
    plt.close()
    
    print(f"Segmentation Evaluation Summary:")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Pixel Accuracy: {mean_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/unet_best.pth')
    parser.add_argument('--config', type=str)
    parser.add_argument('--data-dir', type=str, default='data/')
    parser.add_argument('--output-dir', type=str, default='results/phase2/')
    parser.add_argument('--num-samples', type=int, default=10)
    args = parser.parse_args()
    evaluate(args)
