"""
Phase 5: Train VAE on minority-class brain tumor images.

Usage:
    python -m src.train.train_vae --config configs/vae.yaml
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
    parser.add_argument("--config", type=str, default="configs/vae.yaml")
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
    output_dir = project_root / 'results' / 'phase5' / 'vae'
    checkpoint_dir = project_root / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    image_size = model_cfg.get('image_size', 64)
    channels = model_cfg.get('channels', 3)

    # VAE uses [0, 1] normalization (Sigmoid decoder output)
    vae_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # [0, 1] range
        ToTensorV2(),
    ])

    from src.data.classification_dataset import BrainTumorClassificationDataset
    from src.models.vae import BrainTumorVAE

    dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir), split='train',
        transform=vae_transform, image_size=image_size,
    )

    # Find minority class
    class_counts = dataset.get_sample_count_per_class()
    minority_class_name = min(class_counts, key=class_counts.get)
    minority_class_idx = dataset.class_to_idx[minority_class_name]
    logger.info(f"Minority class: {minority_class_name} ({class_counts[minority_class_name]} samples)")

    minority_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == minority_class_idx]
    minority_dataset = Subset(dataset, minority_indices)

    batch_size = train_cfg.get('batch_size', 64)
    dataloader = DataLoader(minority_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

    # Model
    model = BrainTumorVAE(
        latent_dim=model_cfg.get('latent_dim', 128),
        image_size=image_size,
        channels=channels,
        encoder_channels=model_cfg.get('encoder_channels', [32, 64, 128, 256])
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg.get('learning_rate', 3e-4))

    epochs = train_cfg.get('epochs', 100)
    checkpoint_every = train_cfg.get('checkpoint_every', 25)
    sample_every = train_cfg.get('sample_every', 10)
    kl_annealing = train_cfg.get('kl_annealing', True)
    kl_annealing_epochs = train_cfg.get('kl_annealing_epochs', 20)
    kl_weight_base = train_cfg.get('kl_weight', 0.0001)

    logger.info(f"Training VAE for {epochs} epochs on {len(minority_dataset)} minority-class images")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        n_batches = 0

        # KL weight annealing
        if kl_annealing:
            kl_weight = kl_weight_base * min(1.0, (epoch + 1) / kl_annealing_epochs)
        else:
            kl_weight = kl_weight_base

        for images, _ in dataloader:
            images = images.to(device).float()

            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = model.compute_loss(images, recon, mu, logvar, kl_weight=kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_recon = recon_loss_sum / max(n_batches, 1)
        avg_kl = kl_loss_sum / max(n_batches, 1)

        writer.add_scalar('Loss/Total', avg_loss, epoch)
        writer.add_scalar('Loss/Recon', avg_recon, epoch)
        writer.add_scalar('Loss/KL', avg_kl, epoch)
        writer.add_scalar('KL_Weight', kl_weight, epoch)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Total: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, KL_w: {kl_weight:.5f}"
            )

        if (epoch + 1) % sample_every == 0:
            model.eval()
            with torch.no_grad():
                # Random samples
                z = torch.randn(16, model_cfg.get('latent_dim', 128)).to(device)
                samples = model.decode(z)
                vutils.save_image(samples.cpu(), output_dir / f"samples_epoch_{epoch + 1}.png",
                                  nrow=4, normalize=False)

                # Reconstruction comparison
                n = min(images.size(0), 8)
                comparison = torch.cat([images[:n], recon[:n]])
                vutils.save_image(comparison.cpu(), output_dir / f"recon_epoch_{epoch + 1}.png",
                                  nrow=n, normalize=False)

        if (epoch + 1) % checkpoint_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'minority_class': minority_class_name,
                'latent_dim': model_cfg.get('latent_dim', 128),
            }, checkpoint_dir / f"vae_epoch_{epoch + 1}.pth")

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
        'minority_class': minority_class_name,
        'latent_dim': model_cfg.get('latent_dim', 128),
        'config': model_cfg,
    }, checkpoint_dir / "vae_final.pth")

    model.eval()
    with torch.no_grad():
        z = torch.randn(16, model_cfg.get('latent_dim', 128)).to(device)
        samples = model.decode(z)
        vutils.save_image(samples.cpu(), output_dir / "final_generated_samples.png", nrow=4)

    writer.close()
    logger.info(f"VAE training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
