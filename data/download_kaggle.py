import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_available = True
except (OSError, ImportError) as e:
    kaggle_available = False
    logger.warning("Kaggle API credentials not found or kaggle library not installed.")
    logger.warning("Ensure KAGGLE_USERNAME and KAGGLE_KEY are set or ~/.kaggle/kaggle.json exists.")

DEFAULT_DATASETS = {
    "nickparvar": {
        "id": "masoudnickparvar/brain-tumor-mri-dataset",
        "expected_count": 7023
    },
    "brats": {
        "id": "dschettler8845/brats-2021-task1",
        "expected_count": None # Varies
    },
    "br35h": {
        "id": "ahmedhamada0/brain-tumor-detection",
        "expected_count": 3000
    }
}

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load dataset configuration from base.yaml if available."""
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get("datasets", DEFAULT_DATASETS)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
    return DEFAULT_DATASETS

def verify_dataset(dataset_name: str, target_dir: Path, expected_count: Optional[int]) -> bool:
    """Verify if the downloaded dataset has the expected number of files."""
    if not target_dir.exists():
        logger.error(f"Directory {target_dir} does not exist.")
        return False
    
    # Count image files
    image_extensions = {".jpg", ".jpeg", ".png", ".nii.gz", ".nii"}
    file_count = 0
    for root, _, files in os.walk(target_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                file_count += 1
    
    logger.info(f"[{dataset_name}] Found {file_count} image files in {target_dir}")
    
    if expected_count is not None:
        if dataset_name == "br35h" and file_count >= 3000:
            logger.info(f"[{dataset_name}] Verification passed.")
            return True
        elif dataset_name == "nickparvar" and file_count >= expected_count:
            logger.info(f"[{dataset_name}] Verification passed.")
            return True
        elif file_count != expected_count:
            logger.warning(f"[{dataset_name}] Expected {expected_count} files, but found {file_count}.")
            return False
    
    logger.info(f"[{dataset_name}] Verification passed.")
    return True

def download_and_extract(api: Any, dataset_id: str, target_dir: Path, dataset_name: str, subset_n: Optional[int] = None) -> None:
    """Download and extract a kaggle dataset. For BraTS, support subset extraction."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting download for {dataset_name} ({dataset_id})")
    try:
        api.dataset_download_files(dataset_id, path=str(target_dir), unzip=False, quiet=False)
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_id}: {e}")
        return
    
    zip_path = target_dir / f"{dataset_id.split('/')[-1]}.zip"
    if not zip_path.exists():
        logger.error(f"Zip file {zip_path} not found after download.")
        return
        
    logger.info(f"Extracting {zip_path} to {target_dir}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if dataset_name == "brats" and subset_n is not None:
                logger.info(f"Extracting subset of {subset_n} cases for BraTS dataset...")
                # Extract first subset_n subjects. Typically BraTS cases are in top level folders
                # We identify subject folders by getting top level directory names
                all_files = zip_ref.namelist()
                top_level_dirs = set()
                for f in all_files:
                    parts = f.split('/')
                    if len(parts) > 1:
                        top_level_dirs.add(parts[0])
                
                selected_dirs = list(top_level_dirs)[:subset_n]
                files_to_extract = [f for f in all_files if f.split('/')[0] in selected_dirs]
                zip_ref.extractall(path=str(target_dir), members=files_to_extract)
                logger.info(f"Extracted {len(selected_dirs)} cases.")
            else:
                zip_ref.extractall(path=str(target_dir))
    except zipfile.BadZipFile:
        logger.error(f"Bad zip file: {zip_path}")
    finally:
        # Cleanup zip
        try:
            os.remove(zip_path)
            logger.info(f"Removed zip file: {zip_path}")
        except OSError as e:
            logger.warning(f"Could not remove zip file {zip_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download brain tumor datasets from Kaggle")
    parser.add_argument("--dataset", type=str, choices=["nickparvar", "brats", "br35h", "all"], default="all",
                        help="Dataset to download")
    parser.add_argument("--subset", type=int, default=200,
                        help="Number of cases to extract for BraTS dataset (default: 200)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing datasets without downloading")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Root directory for raw data")
    args = parser.parse_args()

    # Base paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    config_path = project_root / "configs" / "base.yaml"
    
    datasets_config = load_config(config_path)

    if not args.verify_only and not kaggle_available:
        logger.error("Kaggle API is not available. Please configure credentials.")
        return

    api = None
    if not args.verify_only:
        api = KaggleApi()
        api.authenticate()

    target_datasets = ["nickparvar", "brats", "br35h"] if args.dataset == "all" else [args.dataset]
    
    for ds_name in target_datasets:
        if ds_name not in datasets_config:
            logger.error(f"Configuration for dataset '{ds_name}' not found.")
            continue
            
        ds_info = datasets_config[ds_name]
        ds_id = ds_info.get("id")
        expected_count = ds_info.get("expected_count")
        
        # BraTS goes to brats2021
        target_folder = "brats2021" if ds_name == "brats" else ds_name
        target_path = data_dir / target_folder
        
        if not args.verify_only:
            logger.info(f"--- Processing dataset: {ds_name} ---")
            download_and_extract(
                api=api,
                dataset_id=ds_id,
                target_dir=target_path,
                dataset_name=ds_name,
                subset_n=args.subset if ds_name == "brats" else None
            )
        
        logger.info(f"--- Verifying dataset: {ds_name} ---")
        verify_dataset(ds_name, target_path, expected_count)

if __name__ == "__main__":
    main()
