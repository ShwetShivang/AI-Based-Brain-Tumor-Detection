import argparse
import logging
import time
from pathlib import Path
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIGSHARE_API_URL = "https://api.figshare.com/v2/articles/1512427/files"
MAX_RETRIES = 3

def download_file(url: str, dest_path: Path, expected_size: int = None) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            if expected_size and total_size == 0:
                total_size = expected_size

            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            if expected_size and dest_path.stat().st_size != expected_size:
                logger.warning(f"Size mismatch for {dest_path.name}. Expected {expected_size}, got {dest_path.stat().st_size}")
                if attempt < MAX_RETRIES:
                    continue
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Attempt {attempt} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
            else:
                return False
    return False

def main():
    parser = argparse.ArgumentParser(description="Download Figshare Brain Tumor Dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw/figshare",
                        help="Target directory for downloaded files")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing files")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(FIGSHARE_API_URL, timeout=10)
        response.raise_for_status()
        files_info = response.json()
    except Exception as e:
        logger.error(f"Failed to fetch file list from Figshare API: {e}")
        return

    all_passed = True
    
    for file_info in files_info:
        file_name = file_info.get("name")
        download_url = file_info.get("download_url")
        expected_size = file_info.get("size")
        
        if not file_name or not download_url:
            continue
            
        dest_path = data_dir / file_name
        
        if args.verify_only or dest_path.exists():
            if dest_path.exists():
                actual_size = dest_path.stat().st_size
                if actual_size == expected_size:
                    logger.info(f"[Verified] {file_name}")
                    continue
                else:
                    logger.warning(f"[Size Mismatch] {file_name}: expected {expected_size}, got {actual_size}")
                    if args.verify_only:
                        all_passed = False
                        continue
            elif args.verify_only:
                logger.warning(f"[Missing] {file_name}")
                all_passed = False
                continue

        if not args.verify_only:
            logger.info(f"Downloading {file_name}...")
            success = download_file(download_url, dest_path, expected_size)
            if not success:
                all_passed = False
                logger.error(f"Failed to download {file_name} after {MAX_RETRIES} attempts.")

    if all_passed:
        logger.info("All files successfully verified/downloaded.")
    else:
        logger.error("Some files failed verification/download.")

if __name__ == "__main__":
    main()
