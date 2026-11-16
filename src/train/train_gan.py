"""
Phase 5: Train DCGAN on minority-class brain tumor images.

Usage:
    python -m src.train.train_gan --config configs/gan.yaml
"""
import argparse
import yaml
import json
import random
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--config", type=str, default="configs/gan.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    config = load_config(args.config, str(project_root / "configs" / "base.yaml"))

    set_seed(config.get('project', {}).get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    ds_cfg = config.get('datasets', {}).get('nickparvar', {})

    data_dir = project_root / ds_cfg.get('raw_dir', 'data/raw/nickparvar')
    output_dir = project_root / 'results' / 'phase5' / 'gan'
    checkpoint_dir = project_root / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    # GAN-specific transform: resize to 64x64, normalize to [-1, 1]
    gan_image_size = model_cfg.get('image_size', 64)
    gan_transform = A.Compose([
        A.Resize(gan_image_size, gan_image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # [-1, 1]
        ToTensorV2(),
    ])

    from src.data.classification_dataset import BrainTumorClassificationDataset
    from src.models.gan import Generator, Discriminator

    dataset = BrainTumorClassificationDataset(
        root_dir=str(data_dir), split='train',
        transform=gan_transform, image_size=gan_image_size,
    )

    # Find minority class
    class_counts = dataset.get_sample_count_per_class()
    minority_class_name = min(class_counts, key=class_counts.get)
    minority_class_idx = dataset.class_to_idx[minority_class_name]
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Minority class: {minority_class_name} ({class_counts[minority_class_name]} samples)")

    # Filter to minority class
    minority_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == minority_class_idx]
    minority_dataset = Subset(dataset, minority_indices)

    batch_size = train_cfg.get('batch_size', 64)
    dataloader = DataLoader(minority_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

    # Model setup
    latent_dim = model_cfg.get('latent_dim', 100)
    channels = model_cfg.get('channels', 3)
    gen_features = model_cfg.get('gen_features', 64)
    disc_features = model_cfg.get('disc_features', 64)

    netG = Generator(latent_dim=latent_dim, features=gen_features, out_channels=channels).to(device)
    netD = Discriminator(in_channels=channels, features=disc_features).to(device)

    criterion = nn.BCELoss()

    lr_g = train_cfg.get('lr_generator', 0.0002)
    lr_d = train_cfg.get('lr_discriminator', 0.0002)
    beta1 = train_cfg.get('beta1', 0.5)

    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_dim, device=device)
    epochs = train_cfg.get('epochs', 200)
    checkpoint_every = train_cfg.get('checkpoint_every', 50)
    sample_every = train_cfg.get('sample_every', 10)

    logger.info(f"Training GAN for {epochs} epochs on {len(minority_dataset)} minority-class images")

    for epoch in range(epochs):
        netG.train()
        netD.train()

        g_losses = []
        d_losses = []

        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # === Train Discriminator ===
            netD.zero_grad()
            label_real = torch.full((b_size,), 1.0, device=device)
            label_fake = torch.full((b_size,), 0.0, device=device)

            output_real = netD(real_images)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()

            noise = torch.randn(b_size, latent_dim, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            # === Train Generator ===
            netG.zero_grad()
            output_fake = netD(fake)
            errG = criterion(output_fake, label_real)
            errG.backward()
            optimizerG.step()

            g_losses.append(errG.item())
            d_losses.append(errD.item())

        avg_g = np.mean(g_losses)
        avg_d = np.mean(d_losses)
        writer.add_scalar('Loss/G', avg_g, epoch)
        writer.add_scalar('Loss/D', avg_d, epoch)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} | G Loss: {avg_g:.4f}, D Loss: {avg_d:.4f}")

        if (epoch + 1) % sample_every == 0:
            netG.eval()
            with torch.no_grad():
                fake_samples = netG(fixed_noise).cpu()
            vutils.save_image(fake_samples, output_dir / f"samples_epoch_{epoch + 1}.png",
                              normalize=True, nrow=8)

        if (epoch + 1) % checkpoint_every == 0:
            torch.save({
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'epoch': epoch + 1,
                'minority_class': minority_class_name,
            }, checkpoint_dir / f"gan_epoch_{epoch + 1}.pth")

    # Save final
    torch.save({
        'generator': netG.state_dict(),
        'discriminator': netD.state_dict(),
        'epoch': epochs,
        'minority_class': minority_class_name,
        'latent_dim': latent_dim,
    }, checkpoint_dir / "gan_final.pth")

    # Generate final samples grid
    netG.eval()
    with torch.no_grad():
        final_samples = netG(fixed_noise).cpu()
    vutils.save_image(final_samples, output_dir / "final_generated_samples.png",
                      normalize=True, nrow=8)

    writer.close()
    logger.info(f"GAN training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
