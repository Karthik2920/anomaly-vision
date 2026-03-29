---
title: AnomalyVision
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

<div align="center">

# 🎯 AnomalyVision
### Real-Time Surveillance Video Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=flat-square&logo=docker&logoColor=white)](Dockerfile)

*Detect anomalies in surveillance footage using a ConvLSTM Autoencoder —
frame-level scoring, spatial heatmap localisation, and interactive web UI.*

[Demo](#demo) • [Architecture](#architecture) • [Results](#results) • [Setup](#setup) • [Usage](#usage)

</div>

---

## Overview

**AnomalyVision** is a deep learning system that automatically detects anomalous events in surveillance videos. A **ConvLSTM Autoencoder** is trained exclusively on *normal* pedestrian activity from the UCSD Pedestrian Dataset. At inference time the model reconstructs every 10-frame clip — sequences with high reconstruction error deviate from learned normal patterns and are flagged as anomalies.

### Key Features

| Feature | Description |
|---|---|
| 🎥 **Video Upload** | Drag-and-drop `.mp4` / `.avi` files for instant analysis |
| 🔥 **Spatial Heatmaps** | Pixel-wise error maps show *where* in the frame the anomaly occurs |
| 📊 **Regularity Timeline** | Interactive chart with anomaly regions highlighted |
| 🔍 **Frame Inspector** | Side-by-side original vs. reconstructed vs. heatmap overlay |
| 🎛️ **Adjustable Threshold** | Sidebar slider for sensitivity control |
| 📥 **CSV Export** | Download per-frame anomaly scores and flags |
| 🗂️ **Dataset Browser** | Browse and analyse UCSD benchmark sequences in-app |
| 🔐 **Secure Auth** | Hashed credential storage (SHA-256 + salt) |
| 🐳 **Docker** | One-command containerised deployment |

---

## Demo

> **Run it yourself** — see [Setup](#setup) to get started in under 5 minutes.

### Regularity Score Plot

The model outputs a per-frame regularity score (1 = normal, 0 = highly anomalous).
Red-shaded regions are detected anomalies.

```
Regularity
  1.0 ┤ ───────────╮                  ╭────────────────
      │             ╰──╮          ╭───╯
  0.5 ┤ · · · · · · · ·│· · · · ·│· · · (threshold)
      │                ╰──────────╯
  0.0 ┤                  ANOMALY
      └──────────────────────────────────────────────▶ Frame
```

### Heatmap Overlay

For each flagged frame the system renders three panels:

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Original Frame │  Reconstruction │  Error Heatmap  │
│  (surveillance) │  (model output) │  (anomaly area) │
│                 │                 │   🔴🔴🔴        │
│   👤👤👤🚲     │   👤👤👤       │  🔴🔴🟡🟢      │
└─────────────────┴─────────────────┴─────────────────┘
                                     ↑ bicycle flagged
```

---

## Architecture

The model is a **sequence-to-sequence ConvLSTM Autoencoder** that processes
10-frame clips of grayscale surveillance video:

```
Input Sequence  (batch × 10 × 256 × 256 × 1)
        │
   ┌────▼─────────────────────────────────────────────┐
   │ ENCODER                                          │
   │  TimeDistributed Conv2D  128 filters  11×11  s=4 │
   │  LayerNormalization                              │
   │  TimeDistributed Conv2D   64 filters   5×5   s=2 │
   │  LayerNormalization                              │
   └────┬─────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────┐
   │ TEMPORAL CORE  (spatiotemporal motion modelling) │
   │  ConvLSTM2D  64 filters  3×3  return_sequences   │
   │  LayerNormalization                              │
   │  ConvLSTM2D  32 filters  3×3  return_sequences   │
   │  LayerNormalization                              │
   │  ConvLSTM2D  64 filters  3×3  return_sequences   │
   │  LayerNormalization                              │
   └────┬─────────────────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────────────────┐
   │ DECODER                                          │
   │  TimeDistributed ConvTranspose2D  64 filters 5×5 │
   │  LayerNormalization                              │
   │  TimeDistributed ConvTranspose2D 128 filters 11×11│
   │  LayerNormalization                              │
   │  TimeDistributed Conv2D    1 filter   sigmoid    │
   └────┬─────────────────────────────────────────────┘
        │
Output Sequence  (batch × 10 × 256 × 256 × 1)
```

**Anomaly detection logic:**

```
anomaly_score(clip) = ‖clip − reconstruct(clip)‖₂

regularity_score = 1 − normalise(anomaly_score)
                              ↑
                   low → anomaly detected
```

| Property | Value |
|---|---|
| Total parameters | ~1.96 M |
| Input shape | (batch, 10, 256, 256, 1) |
| Loss function | Mean Squared Error |
| Optimizer | Adam (lr=1e-4, decay=1e-5) |
| Training data | UCSD Ped1 + Ped2 Train splits |
| Inference | Sliding window, stride=1 |

---

## Results

Frame-level evaluation on the **UCSD Pedestrian Dataset** test sets:

| Dataset | AUC-ROC | EER | Avg Precision | Notes |
|---------|---------|-----|---------------|-------|
| UCSDped1 | 0.69 | 0.31 | 0.61 | 36 test sequences, 200 frames each |
| UCSDped2 | 0.82 | 0.18 | 0.74 | 12 test sequences, 200 frames each |

> Anomalies include: bicycles, skateboards, carts, and vehicles in pedestrian zones.

Run the evaluation yourself:
```bash
python scripts/evaluate.py --dataset UCSDped1 --model model_anomaly_detection.h5
```

---

## Setup

### Prerequisites

- Python 3.8+
- 8 GB RAM (for model inference)
- Optionally: UCSD Pedestrian Dataset for benchmark evaluation

### 1 — Clone the repository

```bash
git clone https://github.com/Karthik2920/anomaly-vision.git
cd anomaly-vision
```

### 2 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Add the trained model

Place `model_anomaly_detection.h5` (or `model_anomaly.keras`) in the project root.

> The model file is excluded from Git due to its size (~23 MB).
> Download it from the [Releases](https://github.com/Karthik2920/anomaly-vision/releases) page or retrain from scratch (see below).

### 4 — (Optional) Download the UCSD dataset

```
http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
```

Extract as `UCSD_Anomaly_Dataset.v1p2/` in the project root.

### 5 — Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Docker

```bash
# Build image
docker build -t anomalyvision .

# Run (mount your model file)
docker run -p 8501:8501 \
  -v $(pwd)/model_anomaly_detection.h5:/app/model_anomaly_detection.h5 \
  anomalyvision
```

---

## Usage

1. **Login / Sign Up** — create an account via the sidebar
2. **Navigate to Analyze** — choose between:
   - **Upload Video** — drag and drop any `.mp4` / `.avi` file
   - **UCSD Dataset Browser** — pick subset, split, and sequence
3. **Adjust the threshold** using the sidebar slider (default 0.5)
4. **Run detection** — view the regularity timeline and anomaly summary
5. **Inspect flagged frames** — use the slider to browse anomalous frames with heatmap overlay
6. **Export** — download the per-frame CSV report

---

## Retraining

```bash
python scripts/train.py \
  --data_path UCSD_Anomaly_Dataset.v1p2 \
  --dataset   UCSDped2 \
  --epochs    50 \
  --batch_size 4 \
  --output    model_anomaly_detection.h5
```

Training uses **EarlyStopping** (patience=7), **ReduceLROnPlateau**, and
**ModelCheckpoint** to save the best validation-loss model automatically.

---

## Project Structure

```
anomaly-vision/
├── app.py                    # Streamlit web application (main entry point)
├── src/
│   ├── model.py              # ConvLSTM Autoencoder architecture
│   ├── auth.py               # Secure authentication (SHA-256 + salt)
│   ├── config.py             # Centralised configuration
│   └── visualization.py     # Plotting & heatmap utilities
├── scripts/
│   ├── train.py              # Training pipeline with callbacks
│   └── evaluate.py          # Benchmark evaluation (AUC, EER, AP, F1)
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.x / Keras |
| Architecture | ConvLSTM2D Autoencoder |
| Web UI | Streamlit |
| Image Processing | OpenCV, Pillow |
| Visualisation | Matplotlib |
| Evaluation | scikit-learn, SciPy |
| Containerisation | Docker |
| Dataset | UCSD Pedestrian Anomaly Detection |

---

## References

- [UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) — Mahadevan et al., 2010
- [Convolutional LSTM Network](https://arxiv.org/abs/1506.04214) — Shi et al., NeurIPS 2015
- [Learning Temporal Regularity in Video Sequences](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Hasan_Learning_Temporal_Regularity_CVPR_2016_paper.pdf) — Hasan et al., CVPR 2016

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with TensorFlow & Streamlit
</div>
