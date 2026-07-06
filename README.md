<div align="center">

# 🧬 GelVision AI

### Intelligent Gel Electrophoresis Analysis Platform

*Automated lane detection, band segmentation, and molecular weight estimation powered by deep learning*

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-00D4B8)](LICENSE)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Lane_Detection-00FFCE)](https://docs.ultralytics.com)
[![U--Net](https://img.shields.io/badge/U--Net-Band_Segmentation-7C3AED)](https://github.com/qubvel/segmentation_models.pytorch)

[**🚀 Live Demo**](https://gelvision-ai.streamlit.app) · [**📄 Documentation**](#pipeline-overview) · [**🐛 Report Bug**](https://github.com/yourusername/GelVision-AI/issues)

</div>

---

## 📋 Overview

GelVision AI is a production-grade biomedical computer vision platform that automates the analysis of gel electrophoresis images. Using a dual deep learning architecture — YOLOv8 for lane detection and U-Net for band segmentation — it replaces manual, error-prone gel analysis with reproducible, quantitative results.

The platform processes a gel image through a multi-stage pipeline: preprocessing with CLAHE enhancement, lane detection at 93.1% mAP50, semantic band segmentation with Dice loss of 0.165, and log-linear calibration (R²=0.989) against DNA size standards to estimate fragment sizes in base pairs.

---

## 🔬 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      GelVision AI Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📸 Input Image                                                 │
│       │                                                         │
│       ▼                                                         │
│  🔧 Preprocessing ─── CLAHE + Gaussian Blur + Normalization     │
│       │                                                         │
│       ▼                                                         │
│  🎯 YOLOv8 Lane Detection ─── 93.1% mAP50, 2 classes           │
│       │                                                         │
│       ├──▶ Lane crops (sorted left-to-right)                    │
│       │                                                         │
│       ▼                                                         │
│  🧠 U-Net Band Segmentation ─── ResNet34, Dice=0.165           │
│       │                                                         │
│       ▼                                                         │
│  📊 Horizontal Profile Analysis ─── scipy.signal.find_peaks    │
│       │                                                         │
│       ▼                                                         │
│  📐 Log-Linear Calibration ─── R²=0.989, ladder mapping        │
│       │                                                         │
│       ▼                                                         │
│  📋 Results ─── Lane, Band, Size (bp), Intensity                │
│       │                                                         │
│       ▼                                                         │
│  💾 Export ─── CSV │ Excel │ PDF Report                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Model Architecture

### Lane Detection — YOLOv8n
| Metric | Value |
|--------|-------|
| Architecture | YOLOv8 Nano |
| Classes | 2 (Lane, Ladder) |
| Training Data | 41 annotated gel images (Roboflow) |
| mAP@50 | **93.1%** |
| mAP@50-95 | 63.5% |
| Precision | 91.8% |
| Recall | 87.5% |
| Input Size | 640×640 |

### Band Segmentation — U-Net
| Metric | Value |
|--------|-------|
| Architecture | U-Net with ResNet34 encoder |
| Encoder Weights | ImageNet pretrained |
| Framework | segmentation-models-pytorch |
| Training Data | 146 gel images (GelGenie subset) |
| Loss Function | Dice Loss |
| Best Dice Loss | **0.165** |
| Input Size | 512×512 |
| Channels | 1 (grayscale) → 1 (binary mask) |

### Size Calibration
| Metric | Value |
|--------|-------|
| Method | Log-linear regression |
| Calibration R² | **0.989** |
| Supported Ladders | 1kb Plus, 100bp, NEB 1kb |
| Size Range | 50 – 10,000 bp |

---

## 📁 Project Structure

```
GelVision-AI/
├── app.py                    # Streamlit application entry point
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies (Streamlit Cloud)
├── README.md                 # This file
├── .gitignore                # Git ignore rules
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── models/
│   ├── best.pt               # YOLOv8 lane detection model
│   └── unet_best.pth         # U-Net band segmentation model
├── src/
│   ├── __init__.py            # Package initialization
│   ├── preprocess.py          # Image preprocessing (CLAHE, blur, normalization)
│   ├── pipeline.py            # Main analysis pipeline orchestration
│   ├── calibration.py         # Ladder calibration and size estimation
│   └── report.py              # CSV, Excel, and PDF report generation
├── assets/
│   └── sample_gel.jpg         # Sample gel image for demo
└── sample_data/
    └── test_gel.jpg            # Test gel image
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git LFS (for model files)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GelVision-AI.git
cd GelVision-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare Repository
```bash
# Initialize Git LFS for large model files
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
```

### Step 2: Push to GitHub
```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

### Step 3: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your repository, branch (`main`), and main file (`app.py`)
4. Click **"Deploy"**

### Alternative: Hugging Face Hub for Models
If Git LFS quota is exceeded, host models on Hugging Face:
```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="yourusername/gelvision-models", filename="best.pt")
```

### Common Deployment Issues

| Issue | Solution |
|-------|----------|
| OpenCV import error | Use `opencv-python-headless` in requirements.txt |
| Memory exceeded | Models load once with `@st.cache_resource` |
| PyTorch too large | Use CPU-only: `--extra-index-url https://download.pytorch.org/whl/cpu` |
| System deps missing | Add `packages.txt` with `libgl1-mesa-glx` and `libglib2.0-0` |
| Model files too large | Use Git LFS or Hugging Face Hub |
| CUDA not available | Pipeline auto-detects and falls back to CPU |

---

## 📊 Dataset

### Lane Detection
- **Source**: Custom annotated gel images via Roboflow
- **Size**: 41 images with bounding box annotations
- **Classes**: Lane, Ladder
- **Augmentation**: Roboflow auto-augmentation pipeline

### Band Segmentation
- **Source**: [GelGenie Dataset](https://github.com/mattaq31/GelGenie) — 575 annotated gel electrophoresis images
- **Subset Used**: 146 images with pixel-level band masks
- **Task**: Binary segmentation (band vs. background)

---

## 🔮 Future Improvements

- [ ] Multi-channel fluorescence gel support
- [ ] Protein gel (SDS-PAGE) analysis mode
- [ ] Automatic ladder type detection
- [ ] Band molecular weight database integration
- [ ] Batch processing for multiple gel images
- [ ] REST API endpoint for programmatic access
- [ ] Western blot densitometry analysis
- [ ] Integration with laboratory information management systems (LIMS)

---

## 🙏 Acknowledgments

- **[GelGenie](https://github.com/mattaq31/GelGenie)** — Open-source gel image dataset for training band segmentation models
- **[Ultralytics YOLOv8](https://docs.ultralytics.com)** — State-of-the-art object detection framework
- **[Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)** — High-level API for neural network architectures for image segmentation
- **[Streamlit](https://streamlit.io)** — Framework for building data applications
- **[PyTorch](https://pytorch.org)** — Deep learning framework

---

## 📝 Citation

If you use GelVision AI in your research, please cite:

```bibtex
@software{gelvision_ai_2025,
  author = {Abdulrhman},
  title = {GelVision AI: Intelligent Gel Electrophoresis Analysis Platform},
  year = {2025},
  url = {https://github.com/yourusername/GelVision-AI}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for the biomedical research community**

*GelVision AI — Bringing computer vision to the wet lab*

</div>
