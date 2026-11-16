"""
Phase 5: Train DDPM (Diffusion Model) on minority-class brain tumor images.

Usage:
    python -m src.train.train_diffusion --config configs/diffusion.yaml
"""
import argparse
import yaml
import json
import random
import logging
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path, base_path=None):
    config = {}
    if base_path and Path(base_path).exists():
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    with open(config_path, 'r') as f:
        exp = yaml.safe_load(f) or {}
    for k, v in exp.items():
        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
            config[k].update(v)
        else:
            config[k] = v
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    config = load_config(args.config, str(project_root / "configs" / "base.yaml"))

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    ds_cfg = config.get('datasets', {}).get('nickparvar', {})

    data_dir = project_root / ds_cfg.get('raw_dir', 'data/raw/nickparvar')
    output_dir = project_root / 'results' / 'phase5' / 'diffusion'
    checkpoint_dir = project_root / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    image_size = model_cfg.get('image_size', 64)
    channels = model_cfg.get('channels', 3)

    # Transform: normalize to [-1, 1] for diffusion
    diff_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    from src.data.classification_dataset import BrainTumorClassificationDataset
    from src.models.diffusion import BrainTumorDDPM

    dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir), split='train',
        transform=diff_transform, image_size=image_size,
    )

    # Find minority class
    class_counts = dataset.get_sample_count_per_class()
    minority_class_name = min(class_counts, key=class_counts.get)
    minority_class_idx = dataset.class_to_idx[minority_class_name]
    logger.info(f"Minority class: {minority_class_name} ({class_counts[minority_class_name]} samples)")

    minority_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == minority_class_idx]
    minority_dataset = Subset(dataset, minority_indices)

    batch_size = train_cfg.get('batch_size', 16)
    dataloader = DataLoader(minority_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

    # Model
    ddpm_config = {
        'block_out_channels': model_cfg.get('unet', {}).get('block_out_channels', [64, 128, 256, 256]),
        'num_train_timesteps': model_cfg.get('num_train_timesteps', 1000),
    }
    model = BrainTumorDDPM(
        image_size=image_size,
        channels=channels,
        config=ddpm_config
    ).to(device)

    optimizer = optim.Adam(model.unet.parameters(), lr=train_cfg.get('learning_rate', 1e-4))

    epochs = train_cfg.get('epochs', 100)
    checkpoint_every = train_cfg.get('checkpoint_every', 25)
    sample_every = train_cfg.get('sample_every', 10)
    num_inference_steps = train_cfg.get('num_inference_steps', 50)

    logger.info(f"Training DDPM for {epochs} epochs on {len(minority_dataset)} minority-class images")

    for epoch in range(epochs):
        model.unet.train()
        total_loss = 0.0
        n_batches = 0

        for images, _ in dataloader:
            images = images.to(device)

            # Use the model's train_step method (handles noise scheduling)
            loss = model.train_step(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % sample_every == 0:
            logger.info(f"Generating samples at epoch {epoch + 1}...")
            samples = model.sample(
                num_samples=16,
                num_inference_steps=num_inference_steps,
                device=str(device)
            )
            # Rescale from [-1, 1] to [0, 1]
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            vutils.save_image(samples.cpu(), output_dir / f"samples_epoch_{epoch + 1}.png", nrow=4)

        if (epoch + 1) % checkpoint_every == 0:
            model.save(str(checkpoint_dir / f"ddpm_epoch_{epoch + 1}"))

    # Final save
    model.save(str(checkpoint_dir / "ddpm_final"))

    # Final samples
    samples = model.sample(num_samples=16, num_inference_steps=num_inference_steps, device=str(device))
    samples = ((samples + 1) / 2).clamp(0, 1)
    vutils.save_image(samples.cpu(), output_dir / "final_generated_samples.png", nrow=4)

    writer.close()
    logger.info(f"DDPM training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
