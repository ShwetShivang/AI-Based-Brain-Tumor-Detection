<div align="center">

# 🧠 AI-Based Brain Tumor Detection

### A Comprehensive Deep Learning Pipeline for Brain Tumor Classification, Segmentation, and Explainability

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

<p align="center">
  <em>An end-to-end deep learning system for detecting, classifying, and segmenting brain tumors from MRI scans — featuring CNN & ViT classifiers, U-Net segmentation, generative augmentation (GAN, VAE, DDPM), Grad-CAM explainability, domain adaptation, and checkpoint-backed classification summaries.</em>
</p>

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Pipeline Phases](#-pipeline-phases)
- [Usage](#-usage)
- [Model Architectures](#-model-architectures)
- [Configuration](#-configuration)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🔍 Overview

This project implements a **multi-phase deep learning pipeline** for brain tumor detection from MRI scans. It goes beyond simple classification by incorporating:

- **Multi-class tumor classification** (Glioma, Meningioma, Pituitary, No Tumor)
- **Tumor region segmentation** using U-Net
- **Explainable AI** via Grad-CAM and SHAP
- **Synthetic data generation** using GANs, VAEs, and Diffusion Models
- **Cross-dataset domain adaptation**
- **Per-image classification summaries**

The pipeline is designed to be **modular**, **configurable**, and **reproducible** — making it suitable for research, experimentation, and educational purposes.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏥 **Multi-Class Classification** | EfficientNet-B0 and Vision Transformer (ViT) classifiers for 4-class tumor detection |
| 🎯 **Tumor Segmentation** | U-Net with ResNet-34 encoder for pixel-level tumor localization on BraTS 2021 |
| 🔬 **Explainability** | Grad-CAM heatmaps with sanity checks + SHAP explanations |
| 🎨 **Generative Models** | DCGAN, Convolutional VAE, and DDPM for synthetic MRI generation (synthetic images are not yet added to classifier training) |
| 🌐 **Domain Adaptation** | Zero-shot and fine-tuned transfer to Figshare, with Nickparvar/Figshare overlap reported |
| 📊 **Evaluation** | Confusion matrices, per-class metrics, model comparison and inference timing, with test images that duplicate Training scored separately |
| 📝 **Classification Reports** | Downloadable model scores tied to the input image |
| ⚙️ **YAML Configuration** | Fully configurable pipeline via structured YAML config files |
| 🚀 **Mixed Precision Training** | AMP support for faster training on CUDA GPUs |
| 📈 **TensorBoard Logging** | Real-time training visualization and monitoring |

---

## 🏗️ Architecture

The pipeline is organized into **8 sequential phases**, each building upon the previous:

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7
Dataset     CNN         U-Net       Grad-CAM    ViT         Generative  Domain      CNN
Prep &      Classifier  Segmen-     Explain-    Classifier  Models      Adaptation  Report
Verify      Training    tation      ability     Training    (GAN/VAE/   Experiment  Generation
                                                            DDPM)
```

---

## 📁 Project Structure

```
AI-Based-Brain-Tumor-Detection/
│
├── configs/                          # YAML configuration files
│   ├── base.yaml                     # Global settings, dataset paths, seeds
│   ├── cnn_classifier.yaml           # EfficientNet-B0 training config
│   ├── vit_classifier.yaml           # Vision Transformer training config
│   ├── unet_segmentation.yaml        # U-Net segmentation config
│   ├── gan.yaml                      # DCGAN configuration
│   ├── vae.yaml                      # Variational Autoencoder config
│   └── diffusion.yaml                # DDPM diffusion model config
│
├── data/                             # Dataset management
│   ├── download_kaggle.py            # Automated Kaggle dataset downloader
│   ├── download_figshare.py          # Figshare dataset downloader
│   ├── convert_mat_to_png.py         # MATLAB .mat → PNG converter
│   ├── raw/                          # Raw downloaded datasets
│   └── processed/                    # Preprocessed data
│
├── src/                              # Core source code
│   ├── models/                       # Model architectures
│   │   ├── cnn.py                    # EfficientNet-B0 classifier
│   │   ├── vit.py                    # Vision Transformer classifier
│   │   ├── unet.py                   # U-Net segmentation model
│   │   ├── gan.py                    # DCGAN (Generator + Discriminator)
│   │   ├── vae.py                    # Convolutional VAE
│   │   └── diffusion.py             # DDPM wrapper (HuggingFace Diffusers)
│   │
│   ├── data/                         # Dataset classes & loaders
│   │   ├── classification_dataset.py # Classification dataset (Nickparvar)
│   │   ├── segmentation_dataset.py   # Segmentation dataset (BraTS 2021)
│   │   ├── dedup.py                  # Exact / near-duplicate detection for leakage-free splits
│   │   ├── figshare_dataset.py       # Figshare brain tumor dataset
│   │   └── br35h_dataset.py          # Br35H binary detection dataset
│   │
│   ├── train/                        # Training scripts
│   │   ├── classifier_trainer.py     # Shared CNN/ViT training loop
│   │   ├── train_cnn.py              # CNN classifier trainer
│   │   ├── train_vit.py              # ViT classifier trainer
│   │   ├── train_unet.py             # U-Net segmentation trainer
│   │   ├── train_gan.py              # DCGAN trainer
│   │   ├── train_vae.py              # VAE trainer
│   │   └── train_diffusion.py        # DDPM trainer
│   │
│   ├── eval/                         # Evaluation & benchmarking
│   │   ├── evaluate_classifier.py    # Classification metrics & reports
│   │   ├── evaluate_segmentation.py  # Segmentation metrics (Dice, IoU)
│   │   ├── evaluate_augmentation.py  # Augmentation effectiveness analysis
│   │   └── compare_models.py         # CNN vs ViT comparison
│   │
│   ├── explain/                      # Explainability modules
│   │   ├── grad_cam.py               # Grad-CAM heatmap generation
│   │   └── shap_explain.py           # SHAP value explanations
│   │
│   ├── adapt/                        # Domain adaptation
│   │   └── domain_adaptation.py      # Cross-dataset transfer experiments
│   │
│   └── report/                       # Report generation
│       └── llm_report.py             # checkpoint-backed classification summaries
│
├── scripts/                          # Utility scripts
│   ├── visualize_samples.py          # Dataset visualization & verification
│   └── recalibrate_gate.py           # Deploy a new CNN and recalibrate the upload gate
│
├── app/                              # FastAPI server (Brain Tumor Classification and Segmentation dashboard) and Streamlit fallback
├── frontend/                         # React workstation (Vite)
├── tests/                            # pytest suite (gate, pipeline, evaluation, dedup)
│
├── checkpoints/                      # Saved model weights
├── results/                          # Output results by phase
│   ├── phase0_samples/               # Dataset sample visualizations
│   ├── phase1/                       # CNN evaluation results
│   ├── phase2/                       # U-Net segmentation results
│   ├── phase3/                       # Grad-CAM explanations
│   ├── phase4/                       # ViT evaluation results
│   ├── phase5/                       # Generative model outputs
│   ├── phase6/                       # Domain adaptation results
│   └── phase7/                       # CNN classification summaries
│
├── runs/                             # TensorBoard log directory
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🛠️ Installation

### Prerequisites

- **Python** 3.10 or higher
- **CUDA** 12.x (recommended for GPU acceleration)
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/ShwetShivang/AI-Based-Brain-Tumor-Detection.git
cd AI-Based-Brain-Tumor-Detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install PyTorch with CUDA

```bash
# For CUDA 12.8 (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

This project uses **4 publicly available brain tumor MRI datasets**:

| Dataset | Source | Samples | Task | Format |
|---------|--------|---------|------|--------|
| **Nickparvar** | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 7,200 (5,600 Training / 1,600 Testing in the current Kaggle version) | 4-class Classification | JPEG/PNG |
| **BraTS 2021** | [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) | 1,251 cases | Segmentation | NIfTI (.nii.gz) |
| **Figshare** | [Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) | 3,064 | 3-class Classification + Masks | MATLAB (.mat) |
| **Br35H** | [Kaggle](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) | ~3,000 | Binary Detection | JPEG/PNG |

### Automated Download

```bash
# Download all datasets (requires Kaggle API credentials)
python data/download_kaggle.py --dataset all

# Download specific dataset
python data/download_kaggle.py --dataset nickparvar

# Download Figshare dataset
python data/download_figshare.py

# Convert Figshare .mat files to PNG
python data/convert_mat_to_png.py

# Verify datasets
python data/download_kaggle.py --verify-only
```

> **Note:** You need to set up [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials) before downloading. Place your `kaggle.json` in `~/.kaggle/`.

---

## 🔬 Pipeline Phases

### Phase 0 — Dataset Preparation & Visualization

Verify downloaded data and generate sample visualizations.

```bash
python scripts/visualize_samples.py
```

### Phase 1 — CNN Classification (EfficientNet-B0)

Train the CNN classifier with transfer learning, linear warmup + cosine LR, label smoothing, a weight EMA and early stopping. Validation is a duplicate-grouped split of Training (near-identical slices stay on one side), and the final step scores the Testing folder overall and without Training near-duplicates.

```bash
# Train
python -m src.train.train_cnn

# Evaluate
python -m src.eval.evaluate_classifier --checkpoint checkpoints/cnn_best.pth --model-type cnn
```

### Phase 2 — U-Net Tumor Segmentation

Train U-Net with ResNet-34 encoder on BraTS 2021 FLAIR slices for binary whole-tumor segmentation (`python -m src.train.train_unet`), then adapt it to uploads (contrast-enhanced T1) on Figshare tumor masks: `python -m src.train.train_unet --config configs/unet_figshare.yaml`. Measure localization with `python -m src.eval.evaluate_localization`. The first run caches 2D slices once (`python -m src.data.segmentation_dataset`): every tumor slice plus tumor-free brain slices (0.3 per tumor slice), split by case (20% of cases held out). Validation reports Dice/IoU on tumor slices and the false-positive rate on tumor-free slices.

The deployed U-Net is the Figshare-adapted model; see Tumor localization under Results for its measured accuracy on unseen patients.

```bash
# Train
python -m src.train.train_unet

# Evaluate
python -m src.eval.evaluate_segmentation
```

### Phase 3 — Grad-CAM Explainability

Generate Grad-CAM heatmaps with automatic sanity checks for localization.

```bash
python -m src.explain.grad_cam --checkpoint checkpoints/cnn_best.pth
```

### Phase 4 — Vision Transformer (ViT) Classification

Train a ViT-Base-Patch16-224 classifier with layer-wise LR decay and Mixup augmentation. `compare_models` needs both `cnn_best.pth` and `vit_best.pth`; no ViT checkpoint is included.

```bash
# Train
python -m src.train.train_vit

# Evaluate & Compare with CNN
python -m src.eval.evaluate_classifier --checkpoint checkpoints/vit_best.pth --model-type vit
python -m src.eval.compare_models
```

### Phase 5 — Generative Data Augmentation

Train generative models for synthetic MRI generation. `python -m src.eval.evaluate_augmentation --augmented-checkpoint <path>` compares a CNN trained with synthetic images against the baseline on the same test images. No training run in this repository adds synthetic images to the classifier yet, so no augmentation result exists.

```bash
# DCGAN
python -m src.train.train_gan

# Variational Autoencoder
python -m src.train.train_vae

# DDPM (Diffusion Model)
python -m src.train.train_diffusion
```

### Phase 6 — Domain Adaptation

Score the Nickparvar CNN on the Figshare test split zero-shot, then after fine-tuning on part of the Figshare train split. Requires `python data/convert_mat_to_png.py` first. Nickparvar was assembled partly from Figshare, so the script reports how many Figshare test slices also appear in Nickparvar Training and scores the rest separately.

```bash
python -m src.adapt.domain_adaptation --checkpoint checkpoints/cnn_best.pth
```

### Phase 7 — Classification Summary

Generate a research summary from a valid JPEG or PNG slice and the trained CNN checkpoint.

```bash
python -m src.report.llm_report --image-path <path_to_mri> --output-dir results/phase7/
```

#### Clinical-style report (local LLM)

`--narrative` (CLI), the **Generate clinical-style report (PDF)** button (workstation, shown inline) and **Write detailed report** (Streamlit) produce a report laid out like a radiology report, as PDF and Markdown. Examples: `results/sample_reports/` (glioma, pituitary, no tumor).

```bash
python -m src.report.llm_report --image-path <path_to_mri> --output-dir results/phase7/ --narrative
```

Sections: study information, technique, key images (the slice and the slice with the AI-marked tumor region), **findings**, quantitative results (CNN/ViT class probabilities; tumor-region regions, position, extent, area and share of the head cross-section), **impression**, recommendation, input-quality screening, models, limitations, and a review-status block.

- Only Findings and Impression are written by the LLM; every other section is filled directly from the measurements (`src/report/narrative.py`).
- Tumor-region measurements (`src/report/llm_report.py::tumor_findings`) come from the U-Net mask. They are in pixels and image coordinates: JPEG/PNG uploads carry no pixel spacing or patient orientation, so the report gives no millimetre sizes and no patient left/right. Specks under 0.02% of the image are not measured.
- Nothing is invented: patient, indication and sequence are stated as not recorded, no hospital or radiologist is named, and the status is always "preliminary, automated, not reviewed by a radiologist".
- The LLM never sees the image. It receives a fact list built from the report (`facts`) and must not add anatomy, sequences, sizes in mm, grading, prognosis or treatment advice.
- Every number in its draft must appear in that fact list (or be one of its values rounded to fewer decimals) and both headings must be present. Otherwise the model gets one retry that names the problem; if that also fails, Findings and Impression are generated deterministically from the measurements and the report says so.
- Backend: [Ollama](https://ollama.com) with `qwen3:8b` (4-bit, ~5.2 GB, ~6 GB VRAM together with the CNN, ViT and U-Net on an RTX 4060 Laptop GPU). Measured: about 5 s per verified Findings/Impression draft on that GPU. Settings are under `report_llm` in `configs/base.yaml`.
- Using the GPU machine from another computer: `ssh -N -L 11435:127.0.0.1:11434 gpu-laptop`, then set `NEUROLENS_LLM_URL=http://127.0.0.1:11435` before starting the app. Ollama listens on localhost only.

The report is research output generated from model outputs. It inherits every limitation of those outputs and is not a clinical report.

---

## 🧬 Model Architectures

### Classification Models

| Model | Architecture | Parameters | Input Size | Pretrained |
|-------|-------------|------------|------------|-----------|
| **CNN** | EfficientNet-B0 (timm) | ~5.3M | 224×224 | ImageNet |
| **ViT** | ViT-Base-Patch16-224 (timm) | ~86M | 224×224 | ImageNet |

### Segmentation Model

| Model | Encoder | Input | Output | Loss |
|-------|---------|-------|--------|------|
| **U-Net** | ResNet-34 (SMP) | 256×256 (grayscale) | Binary mask | Dice + BCE |

### Generative Models

| Model | Type | Output Size | Latent Dim |
|-------|------|-------------|-----------|
| **DCGAN** | GAN | 64×64×3 | 100 |
| **VAE** | Variational Autoencoder | 64×64×3 | 128 |
| **DDPM** | Denoising Diffusion | 64×64×3 | — |

---

## ⚙️ Configuration

All hyperparameters are managed through YAML configuration files in the `configs/` directory.

### Base Configuration (`configs/base.yaml`)

```yaml
project:
  name: "AI-Based Brain Tumor Detection"
  seed: 42
  device: "auto"
  mixed_precision: true
  num_workers: 2
```

### Model-Specific Configs

Each model has its own config file for full control over training:

- `cnn_classifier.yaml` — Learning rate, epochs, augmentation, scheduler
- `vit_classifier.yaml` — Layer-wise decay, label smoothing, Mixup
- `unet_segmentation.yaml` — Dice/BCE loss weights, elastic transforms
- `gan.yaml` / `vae.yaml` / `diffusion.yaml` — Generative model settings

---

## 📊 Results

### Measured results (deployed checkpoints)

All numbers below were produced by the scripts named in each table on the deployed checkpoints (RTX 4060 Laptop GPU). "Without Training near-duplicates" drops Testing images whose SHA-256 matches, or whose 64-bit dHash is within 2 bits of, a Training image (`src/data/dedup.py`); it is the estimate for unseen slices.

**Classification, Nickparvar Testing (1,600 images)** — `src.eval.evaluate_classifier`, `src.eval.compare_models`

| Model | Accuracy (all) | Macro F1 (all) | Accuracy (no Training near-duplicate, 1,022) | Macro F1 (no near-duplicate) | Glioma recall (all) |
|---|---|---|---|---|---|
| CNN v1 (per-image split, archived) | 94.6% | 0.945 | 91.9% | 0.858 | 80.0% |
| **CNN v2 (deployed, `cnn_best.pth`)** | **96.3%** | **0.963** | **94.7%** | **0.910** | **86.8%** |
| ViT-Base/16 (`vit_best.pth`) | 95.6% | 0.955 | 93.4% | 0.885 | — |

CNN v2 changes: duplicate-grouped validation split, config-driven augmentation (flip, ±15° affine, brightness/contrast), linear warmup + per-step cosine LR, label smoothing 0.1, weight EMA (0.999), checkpoint selection on macro F1. CNN v1 is kept in `checkpoints/archive/`.

**Segmentation, BraTS 2021 FLAIR, 250 held-out cases** — `src.eval.evaluate_segmentation`

| Metric | Value |
|---|---|
| Mean / median Dice on 15,462 tumor slices | 0.849 / 0.922 |
| Mean IoU on tumor slices | 0.774 |
| Tumor-free slices with any predicted tumor pixel (3,752 slices) | 6.6% |

**Tumor localization on unseen patients** — `src.eval.evaluate_localization` (478 Figshare test slices from held-out patients with expert tumor masks + 400 Nickparvar no-tumor slices; images go through the upload pipeline)

| Method | Overlap with expert mask (Dice, mean / median) | Meningioma / pituitary / glioma Dice | Tumor found | Healthy scans with a false mark |
|---|---|---|---|---|
| U-Net trained on BraTS FLAIR only (archived) | 0.36 / 0.23 | 0.66 / 0.45 / 0.11 | 54% | 57% |
| **U-Net trained on Figshare masks (deployed, `unet_best.pth`)** | **0.77 / 0.88** | **0.91 / 0.80 / 0.65** | **94%** | **0.25%** |

| Grad-CAM layer (CNN) | Hottest point inside the tumor | Share of heat inside the tumor |
|---|---|---|
| conv_head (previous default) | 0.4% | 0.3% |
| bn2, post-activation (current) | 3.8% | 3.5% |
| Pointing at the image center (baseline) | 20.7% | — |

Grad-CAM does not locate tumors here: the classifier relies mostly on image context outside the tumor (Nickparvar mixes three source datasets, and class correlates with source). The app therefore draws the tumor from the U-Net (filled red-pink region with outline) and shows Grad-CAM faintly, labeled as CNN attention.

The deployed U-Net (`configs/unet_figshare.yaml`) starts from the BraTS U-Net and is fine-tuned on Figshare contrast-enhanced T1 slices with expert masks (patient-level split, same as the classifier's Figshare split) plus Nickparvar no-tumor slices with empty masks (`src/data/figshare_segmentation.py`). Both use the exact preprocessing of the upload pipeline.

**Upload gate** — `scripts/recalibrate_gate.py` → `results/robustness/gate_evaluation.md` (thresholds re-frozen for CNN v2; 718 held-out clean scans, 1,500 held-out attack fixtures)

| Metric | CNN v1 + gate | CNN v2 + gate |
|---|---|---|
| Clean accuracy without gate / among accepted | 93.2% / 95.8% | 96.1% / 97.5% |
| Clean scans flagged as suspected | 4.7% | 4.5% |
| PGD L∞ 2/4/8 detected | 100 / 99.5 / 88% | 95.5 / 97.5 / 83.5% |
| FGSM L∞ 2/4/8 attack success (no gate) | 91 / 93 / 87% | 20 / 22 / 28% |
| Square L∞ 8 detected | 52% (100 queries) | 68% (1,000 queries) |

The lower FGSM success is most likely gradient masking from label smoothing, not robustness: multi-step PGD still succeeds on 97–100% of images. All attacks are non-adaptive (they do not target the gate).

**Cross-dataset (Figshare)** — `src.adapt.domain_adaptation` (patient-level Figshare split, 478 test slices)

Zero-shot and fine-tuned accuracy are both 99.6%, also on the 235 slices without a near-duplicate in Nickparvar Training. Nickparvar was assembled partly from Figshare, so these slices are not new to the CNN and this experiment does not measure transfer to an unseen dataset.

**Not measured:** synthetic augmentation (no classifier has been trained with GAN/VAE/DDPM images; `src.eval.evaluate_augmentation` exits with "Not evaluated" until one exists).

### Output layout

Results are organized by phase in the `results/` directory:

| Phase | Output | Description |
|-------|--------|-------------|
| Phase 0 | Sample grids | Visual verification of all 4 datasets |
| Phase 1 | Metrics JSON, confusion matrix | CNN classification performance |
| Phase 2 | Dice/IoU scores, overlay masks | Segmentation quality metrics |
| Phase 3 | Grad-CAM heatmaps, sanity JSON | Explainability visualizations |
| Phase 4 | Metrics JSON, CNN vs ViT comparison | ViT classification results |
| Phase 5 | Generated image grids | Synthetic MRI samples |
| Phase 6 | Accuracy comparison charts | Domain adaptation analysis |
| Phase 7 | Text & JSON reports | CNN classification summaries |

### TensorBoard Monitoring

```bash
tensorboard --logdir runs/
```

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, timm, Segmentation Models PyTorch, HuggingFace Diffusers |
| **Data Processing** | NumPy, Pandas, OpenCV, Pillow, scikit-image, NiBabel |
| **Visualization** | Matplotlib, Seaborn, TensorBoard |
| **Explainability** | Captum (Grad-CAM), SHAP |
| **Augmentation** | Albumentations |
| **ML Utilities** | scikit-learn, SciPy |
| **App Framework** | Streamlit |
| **Configuration** | PyYAML, python-dotenv |

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Datasets:** [Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), [BraTS 2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1), [Figshare Brain Tumor](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427), [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
- **Frameworks:** [PyTorch](https://pytorch.org), [timm](https://github.com/huggingface/pytorch-image-models), [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch), [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- **Explainability:** [Captum](https://captum.ai/)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

*Built with ❤️ by [Shwet Shivang](https://github.com/ShwetShivang)*

</div>

## Upload demo and paired robustness evaluation

Run the upload demo from the project root with the trained baseline in `checkpoints/cnn_best.pth`.

```bash
python -m streamlit run app/streamlit_app.py
```

For offline evaluation, supply two directories containing the same held-out cases in the four class folders `glioma`, `meningioma`, `notumor`, and `pituitary`. Use matching filename stems. A lossless PNG can be paired with its original JPEG. Set the pixel budget to the benchmark's declared maximum absolute change in decoded 8-bit RGB values, before resizing. For example, a benchmark with a budget of two intensity levels uses this command.

```bash
python -m src.eval.evaluate_robustness --clean path/to/clean --perturbed path/to/paired_images --pixel-budget 2
```

The evaluator checks pair coverage, image dimensions, and pixel bounds. Record the source and attack method of each supplied set separately. Paired robust accuracy counts a case as correct only when the checkpoint classifies both images correctly. Repeat with `--checkpoint` to score another model on the same pairs. Results from precomputed pairs measure performance on those particular examples. For a white-box comparison, use examples evaluated against the specific checkpoint under test.

## Brain Tumor Classification and Segmentation dashboard

From the project root, install the Python dependencies and build the React interface.

```powershell
python -m pip install -r requirements.txt
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
python -m uvicorn app.api_server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/. For frontend development, run the API and `npm.cmd --prefix frontend run dev` in separate terminals. Vite proxies API calls to the local server.

The React workstation has clinical research, robustness, generative artifact, and YAML/CLI views. See `frontend/README.md` for startup and availability details.

## Upload gate (input validation and robustness screen)

Every entry point (FastAPI `/api/predict`, `/api/sample/...`, `/api/benchmark/...`, `/api/robustness-preview`, the Streamlit fallback and `python -m src.report.llm_report`) calls `src/pipeline.py::analyze`. The gate runs before anything is released:

| Outcome | HTTP | Meaning |
|---|---|---|
| `INVALID_INPUT` | 422 | A named image check failed (decode, min 128 px, grayscale contrast, RGB channel consistency, dark border). No model runs. |
| `ADVERSARIAL_SUSPECTED` | 409 | A stage-1 signal exceeded its frozen threshold: CNN logit drift under squeezing (bit depth, 3x3 median, JPEG q80), an unusually low confidence margin, or margin collapse under small Gaussian noise. CNN/ViT disagreement is available as an opt-in signal (`model_agreement_check`); its threshold is not calibrated. This is a reference-free signal, not proof of an attack. |
| `KNOWN_ADVERSARIAL_TEST_IMAGE` | 409 | The SHA-256 matches a labeled fixture in `data/benchmarks/nickparvar_adversarial/manifest.json` (documented attack, budget and target checkpoint). |
| `SCREENING_UNAVAILABLE` | 503 | A required checkpoint is missing/corrupt or the model raised a runtime error. |
| `SCREENING_PASSED` | 200 | Stage 2 runs unchanged: CNN classification, Grad-CAM, optional U-Net and the report. |

Blocked requests return the gate checks only; Grad-CAM, U-Net and report generation are never invoked, and the GUI clears earlier predictions, overlays and downloads on every new request.

Thresholds live in `configs/base.yaml` (`upload_security`). The signals were chosen on a calibration-split development attack set (`data/benchmarks/nickparvar_dev_calibration`), and their thresholds are frozen jointly from clean calibration scans by `src.eval.evaluate_gate` (see `results/robustness/frozen_thresholds.json`). The screen is stage 1 only: it decides pass or block. Classification, Grad-CAM, U-Net and the report (stage 2) run exactly as before on the original image with their own model passes, so their accuracy is unaffected by the screen.

### Measuring the gate

```powershell
python -m src.eval.adversarial_benchmark      # duplicate-grouped split + FGSM/PGD/Square fixtures with manifest
python -m src.eval.evaluate_gate              # calibrate, freeze, evaluate held-out (results/robustness/)
python -m src.eval.evaluate_robustness --clean data/benchmarks/nickparvar_adversarial/clean --perturbed data/benchmarks/nickparvar_adversarial/pgd_linf4 --pixel-budget 4 --gate
```

Measured held-out results are in `results/robustness/gate_evaluation.md`: classifier accuracy 96.1% on the held-out half of Testing (unchanged by the screen; see the duplicate caveat under Results), 4.5% of input-valid clean scans flagged, PGD detected 84-98%, FGSM 27-39%, Square (1,000 queries) 68%. All attacks are gate-unaware; adaptive attacks are not evaluated, so this is a screen, not a certified defense. Results for the previous checkpoint are kept in `results/robustness/superseded_cnn_v1_per_image_split`, earlier probability-squeeze results in `results/robustness/superseded_v*`. After retraining the CNN, run `python scripts/recalibrate_gate.py --candidate <checkpoint> --tag <old-name>`: thresholds and white-box fixtures belong to one checkpoint.
