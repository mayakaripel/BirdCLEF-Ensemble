# BirdCLEF+ 2026: Five-Fold SED Ensemble for Bird Species Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.15+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Overview

This repository contains an inference pipeline developed for the BirdCLEF+ 2026 Kaggle competition.

The system uses a five-fold ensemble of distilled Sound Event Detection (SED) models to classify 234 bird species from long-duration soundscape recordings.

### Key Features

* Five-fold ONNX SED ensemble
* Clip/frame fusion using weighted averaging (0.3 / 0.7)
* Exponential Moving Average (EMA) temporal smoothing (α = 0.4)
* Resource-efficient inference pipeline
* Designed for Kaggle competition environments
* Fully reproducible workflow
* Publicly available model checkpoints

---

## 📈 Status

| Item | Status |
|--------|--------|
| Competition | BirdCLEF+ 2026 |
| Models | Distilled SED Ensemble |
| Species | 234 |
| Inference | ONNX Runtime |
| Validation Results | Pending |
| Repository Status | Active Development |

## Repository Structure

```text
BirdCLEF-Ensemble/
│
├── birdclef_production.py
├── birdclef_simple.py
├── birdclef_optimized.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Model Files

The repository expects the following ONNX models:

- sed_fold0.onnx
- sed_fold1.onnx
- sed_fold2.onnx
- sed_fold3.onnx
- sed_fold4.onnx

On Kaggle, the models can be attached as a Dataset and accessed from:

```text
/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public/
```  

## 📊 Configuration Summary

| Parameter         | Value |
| ----------------- | ----- |
| Number of Folds   | 5     |
| Number of Species | 234   |
| Clip Weight       | 0.3   |
| Frame Weight      | 0.7   |
| EMA Alpha         | 0.4   |
| Window Length     | 5 s   |
| Hop Length        | 2.5 s |
| Mel Bins          | 256   |
| FFT Size          | 2048  |

> Validation metrics and runtime benchmarks will be reported after full experimental evaluation.

---

## 📄 Output

The pipeline generates:

```text
submission.csv
```

with the format:

```text
row_id,species_1,species_2,...,species_234
rec_001_5,0.12,0.04,...,0.01
rec_001_10,0.15,0.03,...,0.02
```



## 🚀 Quick Start

### Clone Repository

```bash
git clone https://github.com/mayakaripel/BirdCLEF-Ensemble.git
cd BirdCLEF-Ensemble
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Inference

```bash
python birdclef_production.py
```

---

## 🔧 Methodology

### Audio Preprocessing

* Sample Rate: 32 kHz
* Mono audio
* 5-second windows
* 2.5-second overlap

### Feature Extraction

* Log-Mel Spectrogram
* 256 Mel bins
* FFT size 2048
* Hop length 512

### Ensemble Inference

Five distilled SED models produce:

* Clip-level logits
* Frame-level logits

### Clip-Frame Fusion

```python
frame_max = frame_logits.max(axis=1)
fused = 0.3 * clip_logits + 0.7 * frame_max
```

### Ensemble Averaging

```python
probabilities = sigmoid(
    mean(fused_logits_across_folds)
)
```

### Temporal Smoothing

```python
p_smooth[t] = 0.4 * p_raw[t] + 0.6 * p_smooth[t-1]
```

### Aggregation

Predictions are aggregated per BirdCLEF output interval using maximum pooling.

---

## 📖 Reproducibility Statement

The inference pipeline is deterministic and does not contain stochastic inference components. Given identical model checkpoints and preprocessing parameters, results are reproducible across compatible environments.

---

## 📜 License

This project is licensed under the MIT License.

---


## 🙏 Acknowledgments

- Kaggle for hosting the BirdCLEF+ 2026 competition
- ImageCLEF for organizing the BirdCLEF challenge
- Tucker Arrants for releasing the distilled SED models
- PyTorch and ONNX Runtime developers

Email: [maya.karipel@gmail.com](mailto:maya.karipel@gmail.com)
