"""
Phase 0: Visualize sample images from all datasets.
Saves a grid to results/phase0_samples/ for verification.
"""
import sys
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def visualize_nickparvar(data_dir: Path, output_dir: Path, n_per_class: int = 3):
    """Visualize samples from Nickparvar 4-class dataset."""
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(4 * n_per_class, 4 * len(classes)))
    fig.suptitle('Nickparvar Brain Tumor MRI Dataset (4 Classes)', fontsize=16, fontweight='bold')

    for i, cls in enumerate(classes):
        # Try both Training and Testing directories
        class_dir = None
        for subdir in ['Training', 'testing', 'Testing']:
            candidate = data_dir / subdir / cls
            if candidate.exists():
                class_dir = candidate
                break
            candidate = data_dir / subdir / cls.capitalize()
            if candidate.exists():
                class_dir = candidate
                break

        if class_dir and class_dir.exists():
            images = list(class_dir.glob('*.*'))
            images = [p for p in images if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            random.seed(42)
            samples = random.sample(images, min(n_per_class, len(images)))

            for j, img_path in enumerate(samples):
                img = plt.imread(str(img_path))
                axes[i][j].imshow(img, cmap='gray' if img.ndim == 2 else None)
                axes[i][j].set_title(f'{cls}\n{img_path.name}', fontsize=9)
                axes[i][j].axis('off')
        else:
            for j in range(n_per_class):
                axes[i][j].text(0.5, 0.5, f'{cls}\nNot Found', ha='center', va='center')
                axes[i][j].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'nickparvar_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved nickparvar samples to {output_dir / 'nickparvar_samples.png'}")


def visualize_brats(data_dir: Path, output_dir: Path, n_cases: int = 3):
    """Visualize samples from BraTS 2021 dataset (NIfTI volumes)."""
    try:
        import nibabel as nib
    except ImportError:
        print("nibabel not installed, skipping BraTS visualization")
        return

    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('BraTS')])
    if not case_dirs:
        print(f"No BraTS cases found in {data_dir}")
        return

    random.seed(42)
    selected = random.sample(case_dirs, min(n_cases, len(case_dirs)))
    modalities = ['flair', 't1', 't1ce', 't2']

    fig, axes = plt.subplots(n_cases, len(modalities) + 1, figsize=(4 * (len(modalities) + 1), 4 * n_cases))
    fig.suptitle('BraTS 2021 Samples (Axial Slices)', fontsize=16, fontweight='bold')

    if n_cases == 1:
        axes = [axes]

    for i, case_dir in enumerate(selected):
        case_id = case_dir.name
        for j, mod in enumerate(modalities):
            nii_path = case_dir / f"{case_id}_{mod}.nii.gz"
            if nii_path.exists():
                vol = nib.load(str(nii_path)).get_fdata()
                mid_slice = vol.shape[2] // 2
                axes[i][j].imshow(vol[:, :, mid_slice].T, cmap='gray', origin='lower')
                axes[i][j].set_title(f'{mod.upper()}\n{case_id[-5:]}', fontsize=9)
            else:
                axes[i][j].text(0.5, 0.5, f'{mod}\nMissing', ha='center', va='center')
            axes[i][j].axis('off')

        seg_path = case_dir / f"{case_id}_seg.nii.gz"
        if seg_path.exists():
            seg = nib.load(str(seg_path)).get_fdata()
            mid_slice = seg.shape[2] // 2
            axes[i][-1].imshow(seg[:, :, mid_slice].T, cmap='nipy_spectral', origin='lower')
            axes[i][-1].set_title(f'Segmentation\n{case_id[-5:]}', fontsize=9)
        else:
            axes[i][-1].text(0.5, 0.5, 'Seg\nMissing', ha='center', va='center')
        axes[i][-1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'brats_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved BraTS samples to {output_dir / 'brats_samples.png'}")


def visualize_figshare(data_dir: Path, output_dir: Path, n_per_class: int = 3):
    """Visualize converted Figshare PNG images + masks."""
    classes = ['meningioma', 'glioma', 'pituitary']

    fig, axes = plt.subplots(len(classes), n_per_class * 2, figsize=(3 * n_per_class * 2, 4 * len(classes)))
    fig.suptitle('Figshare Brain Tumor Dataset (Images + Masks)', fontsize=16, fontweight='bold')

    for i, cls in enumerate(classes):
        img_dir = data_dir / cls / 'images'
        mask_dir = data_dir / cls / 'masks'

        if img_dir.exists():
            images = sorted(list(img_dir.glob('*.png')))
            random.seed(42)
            samples = random.sample(images, min(n_per_class, len(images)))

            for j, img_path in enumerate(samples):
                img = plt.imread(str(img_path))
                axes[i][j * 2].imshow(img, cmap='gray' if img.ndim == 2 else None)
                axes[i][j * 2].set_title(f'{cls} - Image', fontsize=9)
                axes[i][j * 2].axis('off')

                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    mask = plt.imread(str(mask_path))
                    axes[i][j * 2 + 1].imshow(mask, cmap='gray')
                    axes[i][j * 2 + 1].set_title(f'{cls} - Mask', fontsize=9)
                else:
                    axes[i][j * 2 + 1].text(0.5, 0.5, 'Mask Missing', ha='center', va='center')
                axes[i][j * 2 + 1].axis('off')
        else:
            for j in range(n_per_class * 2):
                axes[i][j].text(0.5, 0.5, f'{cls}\nNot Found', ha='center', va='center')
                axes[i][j].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'figshare_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figshare samples to {output_dir / 'figshare_samples.png'}")


def visualize_br35h(data_dir: Path, output_dir: Path, n_per_class: int = 3):
    """Visualize Br35H binary classification dataset."""
    fig, axes = plt.subplots(2, n_per_class, figsize=(4 * n_per_class, 8))
    fig.suptitle('Br35H Brain Tumor Detection (Binary)', fontsize=16, fontweight='bold')

    for i, label in enumerate(['yes', 'no']):
        label_dir = None
        for candidate in [data_dir / label, data_dir / label.capitalize()]:
            if candidate.exists():
                label_dir = candidate
                break
        if label_dir is None:
            for sub in data_dir.iterdir():
                if sub.is_dir():
                    for candidate in [sub / label, sub / label.capitalize()]:
                        if candidate.exists():
                            label_dir = candidate
                            break
                if label_dir:
                    break

        if label_dir and label_dir.exists():
            images = list(label_dir.glob('*.*'))
            images = [p for p in images if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            random.seed(42)
            samples = random.sample(images, min(n_per_class, len(images)))

            for j, img_path in enumerate(samples):
                img = plt.imread(str(img_path))
                axes[i][j].imshow(img, cmap='gray' if img.ndim == 2 else None)
                axes[i][j].set_title(f'Tumor: {label}\n{img_path.name}', fontsize=9)
                axes[i][j].axis('off')
        else:
            for j in range(n_per_class):
                axes[i][j].text(0.5, 0.5, f'{label}\nNot Found', ha='center', va='center')
                axes[i][j].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'br35h_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Br35H samples to {output_dir / 'br35h_samples.png'}")


def count_files(data_dir: Path, extensions=None) -> int:
    """Count files in a directory recursively."""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.nii', '.nii.gz', '.mat'}
    count = 0
    if data_dir.exists():
        for f in data_dir.rglob('*'):
            if f.is_file():
                suffix = f.suffix.lower()
                if f.name.endswith('.nii.gz'):
                    suffix = '.nii.gz'
                if suffix in extensions:
                    count += 1
    return count


def main():
    output_dir = PROJECT_ROOT / 'results' / 'phase0_samples'
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = PROJECT_ROOT / 'data' / 'raw'
    processed_dir = PROJECT_ROOT / 'data' / 'processed'

    print("=" * 60)
    print("DATASET FILE COUNT VERIFICATION")
    print("=" * 60)

    nickparvar_count = count_files(raw_dir / 'nickparvar', {'.jpg', '.jpeg', '.png'})
    print(f"Nickparvar: {nickparvar_count} images (expected: >=7023)")

    brats_count = count_files(raw_dir / 'brats2021', {'.nii.gz'})
    brats_dirs = len([d for d in (raw_dir / 'brats2021').iterdir() if d.is_dir() and d.name.startswith('BraTS')]) if (raw_dir / 'brats2021').exists() and (raw_dir / 'brats2021').is_dir() else 0
    print(f"BraTS 2021: {brats_count} NIfTI files in {brats_dirs} cases")

    figshare_raw = count_files(raw_dir / 'figshare', {'.mat'})
    figshare_processed = count_files(processed_dir / 'figshare', {'.png'})
    print(f"Figshare: {figshare_raw} .mat files (raw), {figshare_processed} PNGs (processed)")

    br35h_count = count_files(raw_dir / 'br35h', {'.jpg', '.jpeg', '.png'})
    print(f"Br35H: {br35h_count} images (expected: >=3000)")

    print("=" * 60)

    if nickparvar_count > 0:
        visualize_nickparvar(raw_dir / 'nickparvar', output_dir)
    else:
        print("Skipping Nickparvar visualization (no data)")

    if brats_dirs > 0:
        visualize_brats(raw_dir / 'brats2021', output_dir)
    else:
        print("Skipping BraTS visualization (no data)")

    if figshare_processed > 0:
        visualize_figshare(processed_dir / 'figshare', output_dir)
    elif figshare_raw > 0:
        print("Figshare raw .mat files present but not converted. Run convert_mat_to_png.py first.")
    else:
        print("Skipping Figshare visualization (no data)")

    if br35h_count > 0:
        visualize_br35h(raw_dir / 'br35h', output_dir)
    else:
        print("Skipping Br35H visualization (no data)")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
