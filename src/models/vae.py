import torch
import torch.nn as nn
from typing import Tuple, List

class BrainTumorVAE(nn.Module):
    """
    Convolutional Variational Autoencoder.
    """
    def __init__(
        self, 
        latent_dim: int = 128, 
        image_size: int = 64, 
        channels: int = 3, 
        encoder_channels: List[int] = [32, 64, 128, 256]
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        
        # Encoder
        enc_layers = []
        in_c = channels
        for out_c in encoder_channels:
            enc_layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            ])
            in_c = out_c
        self.encoder_conv = nn.Sequential(*enc_layers)
        
        # Compute flattened size
        self._flat_size = (image_size // (2 ** len(encoder_channels))) ** 2 * encoder_channels[-1]
        
        self.fc_mu = nn.Linear(self._flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_size, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self._flat_size)
        
        dec_layers = []
        decoder_channels = encoder_channels[::-1]
        
        for i in range(len(decoder_channels) - 1):
            dec_layers.extend([
                nn.ConvTranspose2d(
                    decoder_channels[i], 
                    decoder_channels[i+1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(decoder_channels[i+1]),
                nn.ReLU()
            ])
            
        dec_layers.extend([
            nn.ConvTranspose2d(
                decoder_channels[-1], 
                channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid()
        ])
        
        self.decoder_conv = nn.Sequential(*dec_layers)
        self.decoder_channels = decoder_channels

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(-1, self.decoder_channels[0], 
                   self.image_size // (2 ** len(self.decoder_channels)), 
                   self.image_size // (2 ** len(self.decoder_channels)))
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def sample(self, num_samples: int, device: str = 'cuda') -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)

    def compute_loss(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        kl_weight: float = 0.0001
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='sum') / x.size(0)
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-sample reconstruction MSE for anomaly detection.
        Returns tensor of shape (B,).
        """
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x)
            mse = nn.functional.mse_loss(reconstruction, x, reduction='none')
            return mse.view(x.size(0), -1).mean(dim=1)
