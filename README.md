# 🌿 Cotton Guard — AI-Powered Cotton Leaf Disease Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A deep learning-based web application for detecting cotton leaf diseases using satellite and RGB imagery, built to help Pakistani cotton farmers identify crop diseases early and get actionable treatment recommendations.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Datasets](#-datasets) • [Results](#-results) • [Project Structure](#-project-structure)

</div>

---

## 📋 Overview

Cotton Guard is an end-to-end cotton leaf disease detection system that combines state-of-the-art deep learning models with an intuitive Streamlit web interface. Farmers can upload an image of a cotton leaf and instantly receive:

- **Disease classification** with confidence scores
- **Detailed disease information** — symptoms, causes, and severity
- **Treatment recommendations** — chemical and organic solutions
- **Prevention strategies** — resistant varieties, cultural practices
- **AI Chatbot** — interactive crop assistant for real-time Q&A

The system supports **6 deep learning architectures** trained on **2 benchmark datasets** covering **8 unique cotton diseases**.

---

## ✨ Features

- **🔍 Disease Detection** — Upload a cotton leaf image and get instant classification across multiple disease classes
- **🧠 6 Deep Learning Models** — ResNet50, DenseNet121, EfficientNetB7, ViT-B/16, Swin Transformer, ConvNeXt-T
- **📁 2 Datasets** — SAR-CLD 2024 (7 classes) and Cotton Leaf Disease (4 classes)
- **📊 Confidence Visualization** — Color-coded probability bars for all classes
- **💊 Treatment Plans** — Actionable chemical and organic treatment recommendations
- **🛡️ Prevention Tips** — Resistant varieties, cultural practices, and pest management
- **🤖 AI Chatbot** — Sidebar crop assistant that answers disease, treatment, and model questions
- **🎨 Colorful UI** — Dark theme with vibrant color-coded disease cards and animations
- **⚡ Real-time Inference** — Fast predictions with inference time tracking

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/cotton-guard.git
cd cotton-guard
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 💻 Usage

### Demo Mode (No Trained Models Required)

By default, the app runs in **Demo Mode** with simulated predictions so you can explore the full UI immediately:

```python
USE_REAL_MODEL = False  # Default — simulated predictions
```

### Real Inference Mode

To use your trained models for real predictions:

**1. Create the model directory structure:**

```
saved_models/
├── SAR-CLD_2024/
│   ├── ResNet50_best.pt
│   ├── DenseNet121_best.pt
│   ├── EfficientNetB7_best.pt
│   ├── ViT_B16_best.pt
│   ├── Swin_T_best.pt
│   └── ConvNeXt_T_best.pt
└── Cotton_Leaf_Disease/
    ├── ResNet50_best.pt
    ├── DenseNet121_best.pt
    ├── EfficientNetB7_best.pt
    ├── ViT_B16_best.pt
    ├── Swin_T_best.pt
    └── ConvNeXt_T_best.pt
```

**2. Enable real inference in `app.py`:**

```python
USE_REAL_MODEL = True
```

**3. Restart the app:**

```bash
streamlit run app.py
```

> **Note:** The `.pt` files are the model weights saved during training via `torch.save(model.state_dict(), path)` in the baseline training script.

---

## 🧠 Models

Six state-of-the-art deep learning architectures were trained and evaluated:

| Model | Parameters | FLOPs | Input Size | Accuracy | MCC | Balanced Acc |
|:------|:-----------|:------|:-----------|:---------|:----|:-------------|
| **DenseNet121** | 7.0M | 2.9G | 224×224 | **98.36%** | 0.9802 | 98.39% |
| **Swin-T** | 27.5M | 4.5G | 224×224 | **98.36%** | 0.9803 | 98.78% |
| **ResNet50** | 23.5M | 4.1G | 224×224 | 97.90% | 0.9747 | 98.60% |
| **ConvNeXt-T** | 27.8M | 4.5G | 224×224 | 97.66% | 0.9720 | 98.39% |
| **EfficientNetB7** | 64.1M | 37.8G | 600×600 | 96.20% | — | — |
| **ViT-B/16** | 85.8M | 17.6G | 224×224 | 95.79% | 0.9494 | 96.56% |

### Training Configuration

- **Split:** 80/20 stratified train/validation
- **Loss Function:** Focal Loss (γ=2.0) with class-weighted α
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing
- **Sampler:** WeightedRandomSampler for class imbalance
- **Early Stopping:** Patience = 10 epochs
- **Augmentation:** Random flip, rotation (20°), color jitter

---

## 📁 Datasets

### SAR-CLD 2024 — 7 Classes

| Class | Description |
|:------|:------------|
| Bacterial Blight | Angular lesions, vein necrosis |
| Curl Virus (CLCuV) | Leaf curling, stunted growth |
| Healthy | No disease symptoms |
| Herbicide Growth Damage | Cupped/strap-shaped leaves |
| Leaf Hopper Jassids | Yellowing margins (hopper burn) |
| Leaf Reddening | Premature purpling of leaves |
| Leaf Variegation | Mosaic-like light/dark patches |

### Cotton Leaf Disease — 4 Classes

| Class | Description |
|:------|:------------|
| Bacterial Blight | Angular water-soaked lesions |
| Curl Virus (CLCuV) | Upward/downward curling |
| Fusarium Wilt | One-sided wilting, vascular browning |
| Healthy | Normal green leaves |

---

## 📊 Results

### Baseline Model Comparison

```
╔═══════════════╦══════════╦═══════════╦═══════╦════════════╗
║ Model         ║ Accuracy ║ Weight F1 ║ MCC   ║ Balanced   ║
╠═══════════════╬══════════╬═══════════╬═══════╬════════════╣
║ DenseNet121   ║ 98.36%   ║ 0.9836    ║ 0.980 ║ 98.39%     ║
║ Swin-T        ║ 98.36%   ║ 0.9837    ║ 0.980 ║ 98.78%     ║
║ ResNet50      ║ 97.90%   ║ 0.9790    ║ 0.975 ║ 98.60%     ║
║ ConvNeXt-T    ║ 97.66%   ║ 0.9768    ║ 0.972 ║ 98.39%     ║
║ ViT-B/16      ║ 95.79%   ║ 0.9582    ║ 0.949 ║ 96.56%     ║
╚═══════════════╩══════════╩═══════════╩═══════╩════════════╝
```

### Key Findings

- **DenseNet121** achieves the best accuracy-to-efficiency ratio with only 7M parameters
- **Swin-T** matches DenseNet121 accuracy but with 4× more parameters
- **CNN-based models** (DenseNet, ResNet, ConvNeXt) generally outperform **Transformer-based** models (ViT, Swin) on this dataset size
- **Focal Loss** with class weighting effectively handles class imbalance

---

## 📂 Project Structure

```
cotton-guard/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── baseline_models.py          # Training script for all 6 models
│
├── saved_models/               # Trained model weights (.pt files)
│   ├── SAR-CLD_2024/
│   │   ├── ResNet50_best.pt
│   │   ├── DenseNet121_best.pt
│   │   ├── EfficientNetB7_best.pt
│   │   ├── ViT_B16_best.pt
│   │   ├── Swin_T_best.pt
│   │   └── ConvNeXt_T_best.pt
│   └── Cotton_Leaf_Disease/
│       └── ... (same structure)
│
├── dataset/                    # Training dataset (ImageFolder format)
│   ├── Bacterial_Blight/
│   ├── Curl_Virus/
│   ├── Healthy/
│   └── ...
│
└── baseline_results/           # Training outputs (auto-generated)
    ├── comparison_table.csv
    ├── comparison_bar_chart.png
    ├── ResNet50/
    │   ├── metrics.txt
    │   ├── training_curves.png
    │   ├── cm_final.png
    │   ├── roc.png
    │   ├── prc.png
    │   ├── calibration.png
    │   └── confused_pairs.png
    ├── DenseNet121/
    └── ... (same for each model)
```

---

## 🤖 Chatbot Capabilities

The built-in Crop Assistant chatbot can answer questions about:

| Topic | Example Questions |
|:------|:-----------------|
| **Disease Info** | "What is curl virus?", "Tell me about bacterial blight" |
| **Treatment** | "How to treat curl virus?", "Blight treatment spray" |
| **Pest Control** | "Whitefly control methods", "Jassid management" |
| **Prevention** | "How to prevent fusarium wilt?" |
| **Models** | "Which model is best?", "Model accuracy comparison" |
| **Datasets** | "What classes are in SAR-CLD?", "Dataset information" |

---

## 🔧 Configuration

Key settings can be modified at the top of `app.py`:

```python
USE_REAL_MODEL = False    # True = load .pt files, False = demo mode

SAR_CLD_CLASSES = [...]   # 7 classes for SAR-CLD 2024
COTTON_LEAF_CLASSES = [...] # 4 classes for Cotton Leaf Disease
```

Training hyperparameters can be adjusted in `baseline_models.py`:

```python
CFG = {
    'img_size'     : 224,
    'batch_size'   : 16,
    'epochs'       : 50,
    'lr'           : 3e-4,
    'weight_decay' : 1e-4,
    'patience'     : 10,
    'seed'         : 42,
}
```

---

## 🙏 Acknowledgements

- **SAR-CLD 2024 Dataset** — Cotton leaf disease satellite imagery
- **Cotton Leaf Disease Dataset** — RGB cotton leaf images
- **PyTorch** & **torchvision** — Deep learning framework
- **Streamlit** — Web application framework

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for Pakistani Cotton Farmers 🇵🇰**

*Cotton Guard v1.0 — Final Year Computer Science Project*

</div>
