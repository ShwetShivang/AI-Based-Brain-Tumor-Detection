import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BrainTumorClassificationDataset(Dataset):
    """
    PyTorch Dataset for the Nickparvar 4-class brain tumor classification dataset.
    Classes: glioma (0), meningioma (1), notumor (2), pituitary (3).
    """

    class_to_idx = {
        'glioma': 0,
        'meningioma': 1,
        'notumor': 2,
        'pituitary': 3
    }
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 224,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory of the dataset containing 'Training' and 'Testing' folders.
            split: 'train', 'val', or 'test'.
            transform: Albumentations transforms to apply. If None, default transforms are applied.
            image_size: Target image size.
            val_ratio: Ratio of training data to use for validation.
            seed: Random seed for splitting.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        if split not in ['train', 'val', 'test']:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")

        self.samples = self._load_samples(val_ratio, seed)

        if self.transform is None:
            self.transform = self._get_default_transform(split)

    def _load_samples(self, val_ratio: float, seed: int) -> List[Tuple[Path, int]]:
        samples = []
        
        if self.split in ['train', 'val']:
            base_dir = self.root_dir / 'Training'
            all_train_samples = []
            
            for class_name, class_idx in self.class_to_idx.items():
                class_dir = base_dir / class_name
                # Some datasets might have subfolder name variations, trying lowercase/capitalize just in case
                if not class_dir.exists():
                    class_dir = base_dir / class_name.capitalize()
                    
                if class_dir.exists():
                    for img_path in class_dir.glob('*.*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            all_train_samples.append((img_path, class_idx))
            
            # Deterministic split
            np.random.seed(seed)
            np.random.shuffle(all_train_samples)
            
            val_size = int(len(all_train_samples) * val_ratio)
            
            if self.split == 'train':
                samples = all_train_samples[val_size:]
            else:
                samples = all_train_samples[:val_size]
                
        else: # test
            base_dir = self.root_dir / 'Testing'
            for class_name, class_idx in self.class_to_idx.items():
                class_dir = base_dir / class_name
                if not class_dir.exists():
                    class_dir = base_dir / class_name.capitalize()
                    
                if class_dir.exists():
                    for img_path in class_dir.glob('*.*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            samples.append((img_path, class_idx))

        return samples

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
        
        # Read image using OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

    def get_sample_count_per_class(self) -> Dict[str, int]:
        """Returns the number of samples for each class in the current split."""
        counts = {k: 0 for k in self.class_to_idx.keys()}
        for _, label_idx in self.samples:
            class_name = self.idx_to_class[label_idx]
            counts[class_name] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Computes class weights for handling class imbalance using inverse frequency."""
        counts = self.get_sample_count_per_class()
        total_samples = len(self.samples)
        
        weights = []
        for i in range(len(self.class_to_idx)):
            class_name = self.idx_to_class[i]
            count = counts[class_name]
            # Add small epsilon to prevent division by zero
            weight = total_samples / (len(self.class_to_idx) * max(count, 1))
            weights.append(weight)
            
        return torch.tensor(weights, dtype=torch.float32)
