import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler
import os
from typing import Optional, Dict, Any

class BrainTumorDDPM(nn.Module):
    """
    DDPM wrapper using diffusers library for generating brain MRIs.
    """
    def __init__(
        self, 
        image_size: int = 64, 
        channels: int = 3, 
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        
        # Configure UNet
        block_out_channels = [64, 128, 256, 256]
        if config and 'block_out_channels' in config:
            block_out_channels = config['block_out_channels']
            
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=channels,
            out_channels=channels,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        # Configure Scheduler
        num_train_timesteps = 1000
        if config and 'num_train_timesteps' in config:
            num_train_timesteps = config['num_train_timesteps']
            
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    def train_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for a single training step.
        batch: Images tensor of shape (B, C, H, W) normalized to [-1, 1].
        """
        device = batch.device
        noise = torch.randn(batch.shape, device=device)
        bs = batch.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bs,), device=device, dtype=torch.int64
        )
        
        # Add noise to clean images
        noisy_images = self.scheduler.add_noise(batch, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self.unet(noisy_images, timesteps).sample
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(
        self, 
        num_samples: int, 
        num_inference_steps: int = 50, 
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generates images by denoising.
        Returns tensor of shape (N, C, H, W) in range [-1, 1].
        """
        self.unet.eval()
        self.unet.to(device)
        
        # Start from pure noise
        image = torch.randn(
            (num_samples, self.channels, self.image_size, self.image_size), 
            device=device
        )
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            # Predict noise residual
            model_output = self.unet(image, t).sample
            
            # Compute previous image x_{t-1}
            image = self.scheduler.step(model_output, t, image).prev_sample
            
        return image

    def save(self, path: str) -> None:
        """Saves the UNet model."""
        os.makedirs(path, exist_ok=True)
        self.unet.save_pretrained(path)
        
    def load(self, path: str) -> None:
        """Loads the UNet model."""
        self.unet = UNet2DModel.from_pretrained(path)
