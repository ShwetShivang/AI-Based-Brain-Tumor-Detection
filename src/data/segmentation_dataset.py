import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BraTSSegmentationDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2021 NIfTI volumes.
    Extracts 2D axial slices from 3D NIfTI volumes.
    """
    def __init__(
        self,
        root_dir: str | Path,
        split: str = 'train',
        transform: Optional[Callable] = None,
        image_size: int = 256,
        modality: str = 'flair',
        min_tumor_fraction: float = 0.001
    ):
        """
        Args:
            root_dir: Directory containing BraTS cases.
            split: 'train' or 'val'.
            transform: Albumentations transforms to apply.
            image_size: Target image size.
            modality: Which MRI modality to use ('flair', 't1', 't1ce', 't2').
            min_tumor_fraction: Minimum fraction of tumor pixels required to include a slice.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.modality = modality
        self.min_tumor_fraction = min_tumor_fraction
        
        if self.split not in ['train', 'val']:
            raise ValueError("Split must be one of 'train' or 'val'")
            
        # Get all case directories
        case_dirs = [d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('BraTS2021')]
        
        # Simple split (you could make this more robust with seed)
        np.random.seed(42)
        np.random.shuffle(case_dirs)
        val_size = int(len(case_dirs) * 0.15)
        
        if split == 'val':
            self.case_dirs = case_dirs[:val_size]
        else:
            self.case_dirs = case_dirs[val_size:]
            
        self.transform = transform
        if self.transform is None:
            self.transform = self._get_default_transform(split)
            
        # Pre-compute valid slices
        self.valid_slices = self._precompute_valid_slices()

    def _get_default_transform(self, split: str) -> Callable:
        if split == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                ToTensorV2(),
            ])

    def _precompute_valid_slices(self) -> List[Tuple[Path, Path, int]]:
        """Precomputes mapping of (image_path, mask_path, slice_index) for valid slices."""
        valid_slices = []
        for case_dir in self.case_dirs:
            case_id = case_dir.name
            img_path = case_dir / f"{case_id}_{self.modality}.nii.gz"
            seg_path = case_dir / f"{case_id}_seg.nii.gz"
            
            if not img_path.exists() or not seg_path.exists():
                continue
                
            # Load segmentation mask
            try:
                mask_nii = nib.load(str(seg_path))
                mask_data = mask_nii.get_fdata()
                
                # Check each axial slice
                for z in range(mask_data.shape[2]):
                    slice_mask = mask_data[:, :, z]
                    
                    # Convert BraTS multi-class (1,2,4) to binary
                    binary_mask = (slice_mask > 0).astype(np.float32)
                    
                    tumor_fraction = binary_mask.mean()
                    if tumor_fraction > self.min_tumor_fraction:
                        valid_slices.append((img_path, seg_path, z))
            except Exception as e:
                print(f"Error processing {case_id}: {e}")
                
        return valid_slices

    def __len__(self) -> int:
        return len(self.valid_slices)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] using percentile clipping to handle outliers."""
        non_zero = image[image > 0]
        if len(non_zero) == 0:
            return image
            
        p1, p99 = np.percentile(non_zero, [1, 99])
        
        # Clip and normalize
        image = np.clip(image, p1, p99)
        if p99 - p1 > 0:
            image = (image - p1) / (p99 - p1)
        else:
            image = np.zeros_like(image)
            
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, seg_path, slice_idx = self.valid_slices[idx]
        
        # Load NIfTI data
        img_nii = nib.load(str(img_path))
        img_data = img_nii.get_fdata()
        
        seg_nii = nib.load(str(seg_path))
        seg_data = seg_nii.get_fdata()
        
        # Extract 2D axial slice
        image_slice = img_data[:, :, slice_idx]
        mask_slice = seg_data[:, :, slice_idx]
        
        # Normalize image
        image_slice = self._normalize_image(image_slice)
        
        # Convert mask to binary
        binary_mask = (mask_slice > 0).astype(np.float32)
        
        # Add channel dimension for Albumentations
        # A wants (H, W, C) so we don't strictly need to add C for grayscale, 
        # but let's be explicit if we use float32 images
        
        if self.transform:
            augmented = self.transform(image=image_slice, mask=binary_mask)
            image_slice = augmented['image']
            mask_slice = augmented['mask']
            
            # Albumentations ToTensorV2 converts to [C, H, W]
            # Since input was 2D, output image might be [H, W], we ensure it's [1, H, W]
            if len(image_slice.shape) == 2:
                image_slice = image_slice.unsqueeze(0)
            if len(mask_slice.shape) == 2:
                mask_slice = mask_slice.unsqueeze(0)
                
        return image_slice.float(), mask_slice.float()
