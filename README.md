# HotRecForCTR: Common Ranking Models for CTR Prediction 🚀

[中文文档](README_CN.md) | English

A repository of common ranking models for Click-Through Rate (CTR) prediction, featuring both traditional machine learning and deep learning approaches. Uses the Criteo dataset as a benchmark.

## 📋 Project Overview

This project implements various CTR prediction models with a focus on:
- 🔍 Traditional ML approaches (LR-GBDT)
- 🧠 Deep learning models (FM)
- 🧩 Modular, extensible architecture with registry mechanism
- 🛠️ Utility modules for logging, progress tracking, and model analysis

## 📁 Repository Structure

```
.
├── data/                  # Data processing and storage
├── ml-based/              # Machine learning based models
└── dl-based/              # Deep learning based models
    ├── config/            # Model configurations
    ├── datasets/          # Dataset implementations
    ├── models/            # Model implementations
    ├── utils/             # Utility modules
    └── torchprint/        # Model analysis tools
```

## ✨ Key Features

### Models
- **LR-GBDT**: Hybrid model combining GBDT for feature transformation with LR
- **FM (Factorization Machines)**: Captures pairwise feature interactions

### Utilities
- **Register Mechanism**: Component management system for models, datasets, etc.
- **ColorLogger**: Custom colored logging with multiple levels
- **ProgressBar**: Enhanced progress tracking with alive_progress
- **TorchPrint**: PyTorch model analysis (parameters, FLOPs, memory usage)

## 🔧 Environment Setup

```bash
# Create a new conda environment with Python 3.10
conda create -n reco python=3.10

# Activate the environment
conda activate reco

# Install required packages
pip install pandas numpy scikit-learn lightgbm torch jupyter colorama alive-progress pydantic
```

## 🚀 Quick Start

```bash
# Activate environment
conda activate reco

# Run ML model
cd ml-based
python lr_gbdt.py

# Run DL model with interactive component selection
cd ../dl-based
python main.py
```

## 📊 Dataset

Uses the [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) dataset with:
- 13 continuous features
- 26 categorical features
- Binary click labels (1 for click, 0 for no click)

## 📦 Requirements

- Python 3.10+
- PyTorch, pandas, numpy, scikit-learn
- LightGBM, colorama, alive-progress
- Pydantic (for configuration validation)

## 🔮 Future Work

- Additional deep learning models (DeepFM, Wide & Deep, DCN, DIN)
- Evaluation metrics and visualization tools
- Hyperparameter tuning
- Distributed training support
