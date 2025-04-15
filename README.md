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
- **TorchPrint**: Custom PyTorch model analysis tool
    - Displays model structure in a beautiful tree format with color coding
    - Calculates parameters (total and trainable) for each layer
    - Computes FLOPs and MACs for computational complexity analysis
    - Estimates memory usage for model deployment considerations
    - Supports custom input dimension specification without batch size
    - Handles multiple inputs with different data types (including torch.long for indices)

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

# Try the utility demos
python demo_progress.py  # Progress bar demo
python demo_torchprint.py  # Model analysis tool demo
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

## 📊 TorchPrint: Model Analysis Tool

The TorchPrint module provides detailed analysis of PyTorch models with a beautiful, color-coded output:

```python
from torchprint import analyze_model

# Basic usage
summary = analyze_model(model)
print(summary)

# Advanced usage with custom input specifications
summary = analyze_model(
    model,
    model_name="MyModel",
    input_dims=[(13,), (5,)],  # Two inputs with dimensions (13,) and (5,)
    long_indices=[1],  # Second input should be torch.long dtype
    batch_size=128,    # Specify batch size for analysis
)
```

Features:
- Tree-structured view of model hierarchy
- Parameter counts (total and trainable) for each layer
- Input and output shapes for each layer
- FLOPs and MACs calculations for computational complexity
- Memory usage estimation
- Support for multiple inputs with different data types

## 🔮 Future Work

- Additional deep learning models (DeepFM, Wide & Deep, DCN, DIN)
- Evaluation metrics and visualization tools
- Hyperparameter tuning
- Distributed training support
