<div align="center">

# 🧠 AI-Based Brain Tumor Detection

### A Comprehensive Deep Learning Pipeline for Brain Tumor Classification, Segmentation, and Explainability

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

<p align="center">
  <em>An end-to-end deep learning system for detecting, classifying, and segmenting brain tumors from MRI scans — featuring CNN & ViT classifiers, U-Net segmentation, generative augmentation (GAN, VAE, DDPM), Grad-CAM explainability, domain adaptation, and LLM-powered diagnostic reports.</em>
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
- **AI-powered diagnostic report generation**

The pipeline is designed to be **modular**, **configurable**, and **reproducible** — making it suitable for research, experimentation, and educational purposes.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏥 **Multi-Class Classification** | EfficientNet-B0 and Vision Transformer (ViT) classifiers for 4-class tumor detection |
| 🎯 **Tumor Segmentation** | U-Net with ResNet-34 encoder for pixel-level tumor localization on BraTS 2021 |
| 🔬 **Explainability** | Grad-CAM heatmaps with sanity checks + SHAP explanations |
| 🎨 **Generative Augmentation** | DCGAN, Convolutional VAE, and DDPM for synthetic MRI generation |
| 🌐 **Domain Adaptation** | Cross-dataset transfer learning and fine-tuning evaluation |
| 📊 **Comprehensive Evaluation** | Confusion matrices, per-class metrics, model comparison, and inference benchmarks |
| 📝 **LLM-Powered Reports** | Automated diagnostic report generation using Anthropic Claude API |
| ⚙️ **YAML Configuration** | Fully configurable pipeline via structured YAML config files |
| 🚀 **Mixed Precision Training** | AMP support for faster training on CUDA GPUs |
| 📈 **TensorBoard Logging** | Real-time training visualization and monitoring |

---

## 🏗️ Architecture

The pipeline is organized into **8 sequential phases**, each building upon the previous:

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7
Dataset     CNN         U-Net       Grad-CAM    ViT         Generative  Domain      LLM
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
│   │   ├── figshare_dataset.py       # Figshare brain tumor dataset
│   │   └── br35h_dataset.py          # Br35H binary detection dataset
│   │
│   ├── train/                        # Training scripts
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
│       └── llm_report.py             # LLM-powered diagnostic reports
│
├── scripts/                          # Utility scripts
│   └── visualize_samples.py          # Dataset visualization & verification
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
│   └── phase7/                       # LLM diagnostic reports
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
| **Nickparvar** | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 7,023 | 4-class Classification | JPEG/PNG |
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

Train the CNN classifier with transfer learning, cosine annealing LR, and early stopping.

```bash
# Train
python -m src.train.train_cnn

# Evaluate
python -m src.eval.evaluate_classifier --checkpoint checkpoints/cnn_best.pth --model-type cnn
```

### Phase 2 — U-Net Tumor Segmentation

Train U-Net with ResNet-34 encoder on BraTS 2021 for binary tumor segmentation.

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

Train a ViT-Base-Patch16-224 classifier with layer-wise LR decay and Mixup augmentation.

```bash
# Train
python -m src.train.train_vit

# Evaluate & Compare with CNN
python -m src.eval.evaluate_classifier --checkpoint checkpoints/vit_best.pth --model-type vit
python -m src.eval.compare_models
```

### Phase 5 — Generative Data Augmentation

Train generative models for synthetic MRI generation.

```bash
# DCGAN
python -m src.train.train_gan

# Variational Autoencoder
python -m src.train.train_vae

# DDPM (Diffusion Model)
python -m src.train.train_diffusion
```

### Phase 6 — Domain Adaptation

Evaluate cross-dataset transfer learning performance.

```bash
python -m src.adapt.domain_adaptation --checkpoint checkpoints/cnn_best.pth
```

### Phase 7 — LLM Diagnostic Report

Generate AI-powered diagnostic reports for MRI scans.

```bash
python -m src.report.llm_report --image-path <path_to_mri> --output-dir results/phase7/
```

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
| Phase 7 | Text & JSON reports | AI diagnostic reports |

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
| **LLM Integration** | Anthropic Claude API |
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
