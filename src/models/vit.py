import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Tuple

class BrainTumorViT(nn.Module):
    """
    Vision Transformer classifier for brain tumor detection.
    """
    def __init__(
        self, 
        model_name: str = 'vit_base_patch16_224', 
        num_classes: int = 4, 
        pretrained: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freezes all layers except the head."""
        for param in self.model.parameters():
            param.requires_grad = False
        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'get_classifier'):
            classifier = self.model.get_classifier()
            for param in classifier.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Unfreezes all layers."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_last_conv_layer(self) -> str:
        """Returns the name of the last attention block (useful for ViT-based Grad-CAM)."""
        return 'blocks.-1' # Commonly the last block in timm ViT models

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_parameters, trainable_parameters)."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'BrainTumorViT':
        """Creates a model instance from a configuration dictionary."""
        return cls(
            model_name=config_dict.get('model_name', 'vit_base_patch16_224'),
            num_classes=config_dict.get('num_classes', 4),
            pretrained=config_dict.get('pretrained', True),
            dropout=config_dict.get('dropout', 0.0)
        )
