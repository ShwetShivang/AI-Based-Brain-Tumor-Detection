import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    DCGAN Generator for brain MRI generation.
    Input: latent_dim noise vector
    Output: 64x64x3 image
    """
    def __init__(self, latent_dim: int = 100, features: int = 64, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # state size. (features*8) x 4 x 4
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # state size. (features*4) x 8 x 8
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # state size. (features*2) x 16 x 16
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # state size. (features) x 32 x 32
            nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (out_channels) x 64 x 64
        )
        self.apply(self.weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input from (B, latent_dim) to (B, latent_dim, 1, 1) if necessary
        if x.dim() == 2:
            x = x.view(-1, self.latent_dim, 1, 1)
        return self.main(x)
        
    @staticmethod
    def weights_init(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for brain MRI.
    Input: 64x64x3 image
    """
    def __init__(self, in_channels: int = 3, features: int = 64):
        super().__init__()
        self.main = nn.Sequential(
            # input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features) x 32 x 32
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*8) x 4 x 4
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(self.weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1).squeeze(1)

    @staticmethod
    def weights_init(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
