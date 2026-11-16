import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Optional, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FigshareDataset(Dataset):
    """
    PyTorch Dataset for Figshare PNG images (3-class).
    Classes: meningioma (0), glioma (1), pituitary (2)
    """
    class_to_idx = {
        'meningioma': 0,
        'glioma': 1,
        'pituitary': 2
    }
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Directory containing processed/figshare/...
            split: 'train', 'val', or 'test'
            transform: Albumentations transforms
            image_size: Image size
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set (remainder goes to test)
            seed: Random seed
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
            
        self.samples = self._load_samples(train_ratio, val_ratio, seed)
        
        self.transform = transform
        if self.transform is None:
            self.transform = self._get_default_transform(split)

    def _load_samples(self, train_ratio: float, val_ratio: float, seed: int) -> List[Tuple[Path, int]]:
        all_samples = []
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name / 'images'
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    all_samples.append((img_path, class_idx))
                    
        # Deterministic split
        np.random.seed(seed)
        np.random.shuffle(all_samples)
        
        n_samples = len(all_samples)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        if self.split == 'train':
            return all_samples[:train_end]
        elif self.split == 'val':
            return all_samples[train_end:val_end]
        else:
            return all_samples[val_end:]

    def _get_default_transform(self, split: str) -> Callable:
        if split == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label
