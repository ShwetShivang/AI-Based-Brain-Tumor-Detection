import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict, Any, Tuple

class BrainTumorUNet(nn.Module):
    """
    U-Net segmentation model for brain tumor detection.
    """
    def __init__(
        self, 
        encoder_name: str = 'resnet34', 
        encoder_pretrained: bool = True, 
        in_channels: int = 1, 
        out_channels: int = 1
    ):
        super().__init__()
        weights = 'imagenet' if encoder_pretrained else None
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits."""
        return self.model(x)

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_parameters, trainable_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'BrainTumorUNet':
        """Creates a model instance from a configuration dictionary."""
        return cls(
            encoder_name=config_dict.get('encoder_name', 'resnet34'),
            encoder_pretrained=config_dict.get('encoder_pretrained', True),
            in_channels=config_dict.get('in_channels', 1),
            out_channels=config_dict.get('out_channels', 1)
        )
