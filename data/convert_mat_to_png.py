import argparse
import logging
import os
from pathlib import Path
import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class label mapping based on documentation
# 1 = meningioma, 2 = glioma, 3 = pituitary
CLASS_MAP = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary"
}

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0-255 uint8."""
    arr = arr.astype(float)
    min_val = arr.min()
    max_val = arr.max()
    
    if max_val - min_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)
        
    arr = (arr - min_val) / (max_val - min_val)
    arr = (arr * 255.0).astype(np.uint8)
    return arr

def convert_mat_file(mat_path: Path, output_dir: Path) -> int:
    """
    Convert a single .mat file to PNG image and mask.
    Returns the class label integer on success, 0 on failure.
    """
    try:
        mat_data = scipy.io.loadmat(str(mat_path))
        
        if 'cjdata' not in mat_data:
            logger.warning(f"No 'cjdata' struct found in {mat_path.name}")
            return 0
            
        cjdata = mat_data['cjdata'][0, 0]
        
        # cjdata structure:
        # label: 1, 2, or 3
        # PID: patient ID string
        # image: image data
        # tumorBorder: coordinates
        # tumorMask: binary mask
        
        label_arr = cjdata['label']
        image_arr = cjdata['image']
        mask_arr = cjdata['tumorMask']
        
        label = int(label_arr.flatten()[0])
        class_name = CLASS_MAP.get(label)
        
        if not class_name:
            logger.warning(f"Unknown label {label} in {mat_path.name}")
            return 0
            
        # Prepare directories
        class_dir = output_dir / class_name
        img_dir = class_dir / "images"
        mask_dir = class_dir / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Process image
        # Some images might be 3D or RGB, but typically they are 2D grayscale
        if image_arr.ndim == 3 and image_arr.shape[2] == 3:
            # RGB image, normalize each channel or just scale
            img_uint8 = normalize_to_uint8(image_arr)
            img = Image.fromarray(img_uint8, 'RGB')
        else:
            # Grayscale
            img_uint8 = normalize_to_uint8(image_arr)
            img = Image.fromarray(img_uint8, 'L')
            
        # Process mask
        mask_uint8 = (mask_arr > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask_uint8, 'L')
        
        # Save
        base_name = mat_path.stem
        img.save(img_dir / f"{base_name}.png")
        mask.save(mask_dir / f"{base_name}.png")
        
        return label
        
    except Exception as e:
        logger.error(f"Error processing {mat_path.name}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Convert Figshare .mat files to PNG")
    parser.add_argument("--input-dir", type=str, default="data/raw/figshare",
                        help="Input directory containing .mat files")
    parser.add_argument("--output-dir", type=str, default="data/processed/figshare",
                        help="Output directory for PNG images and masks")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    mat_files = list(input_dir.glob("*.mat"))
    
    # In some versions, the .mat files might be zipped inside the directory, 
    # but the download script saves them as downloaded. We assume they are extracted or downloaded directly.
    if not mat_files:
        logger.warning(f"No .mat files found in {input_dir}")
        return
        
    logger.info(f"Found {len(mat_files)} .mat files to process.")
    
    class_counts = {1: 0, 2: 0, 3: 0}
    
    for mat_file in tqdm(mat_files, desc="Converting files"):
        label = convert_mat_file(mat_file, output_dir)
        if label in class_counts:
            class_counts[label] += 1
            
    logger.info("Conversion complete.")
    logger.info("Final counts per class:")
    for label, name in CLASS_MAP.items():
        logger.info(f"  {name.capitalize()}: {class_counts[label]}")

if __name__ == "__main__":
    main()
