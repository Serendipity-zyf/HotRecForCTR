# HotRecForCTR: Common Ranking Models for CTR Prediction

This repository contains implementations of common ranking models for Click-Through Rate (CTR) prediction. It focuses on both traditional machine learning approaches and deep learning methods for CTR prediction tasks, using the Criteo dataset as a benchmark.

## Repository Structure

```
.
├── data/                  # Data processing and storage
│   ├── data_process.ipynb # Data preprocessing notebook
│   └── criteo/            # Criteo dataset (ignored in git)
├── ml-based/              # Machine learning based models
│   ├── config.py          # Configuration parameters
│   └── lr_gbdt.py         # LR-GBDT model implementation
└── dl-based/              # Deep learning based models
    ├── utils/             # Utility modules
    │   ├── logger.py      # Custom colored logging utility
    │   └── progress.py    # Enhanced progress bar utilities
    ├── __init__.py        # Package initialization
    └── demo_utils.py      # Demo for progress bar utilities
```

## Models Implemented

### Machine Learning Based Models

- **LR-GBDT (Logistic Regression with Gradient Boosting Decision Trees)**
  - A hybrid model that combines the power of GBDT for feature transformation with LR for prediction
  - Implements parameter tuning with different configurations
  - Uses LightGBM as the GBDT implementation

### Deep Learning Based Models

- **Utility Modules**
  - **ColorLogger**: A custom logger with colored output using colorama
    - Provides different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) with distinct colors
    - Supports formatted log messages with arguments
    - Configurable timestamp format and output stream
  - **ProgressBar**: Enhanced progress bar utilities using alive_progress
    - Class-based implementation with context manager support
    - Supports processing items with progress tracking
    - Provides timed progress bars and multi-task progress tracking
    - Customizable spinner styles, bar styles, and colors

- **FM (Factorization Machines)**
  - A model that can capture pairwise feature interactions (planned implementation)

## Dataset

This repository uses the [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) dataset, which contains:
- 13 continuous features
- 26 categorical features
- Binary click labels (1 for click, 0 for no click)

The dataset is preprocessed in `data/data_process.ipynb`, which includes:
- Loading and sampling data
- Handling missing values
- Feature encoding
- Saving processed data in parquet format

## Environment Setup

This project uses a conda environment named 'reco'. To set up and activate the environment:

```bash
# Create the conda environment (if not already created)
conda create -n reco python=3.10

# Activate the environment
conda activate reco

# Install required packages
pip install pandas numpy scikit-learn lightgbm torch jupyter colorama alive-progress
```

## Usage

### Data Preprocessing

1. Download the Criteo dataset and place it in the `data/criteo/` directory
2. Run the `data/data_process.ipynb` notebook to preprocess the data

### Running Models

#### Machine Learning Models

```python
# Run the LR-GBDT model
cd ml-based
python lr_gbdt.py
```

#### Utility Modules

Run the demo utility to see the progress bar functionality:
```bash
# Activate the conda environment first
conda activate reco

# Run the demo utility
python -m dl-based.demo_utils
```

Deep learning model implementations are in progress. Stay tuned for upcoming implementations of FM, DeepFM, and other models.

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- LightGBM
- PyTorch (for deep learning models)
- Jupyter
- colorama (for colored logging)
- alive_progress (for progress bars)

## Future Work

- Implement more deep learning models:
  - FM (Factorization Machines)
  - DeepFM
  - Wide & Deep
  - DCN (Deep & Cross Network)
  - DIN (Deep Interest Network)
- Add evaluation metrics and visualization tools
- Implement hyperparameter tuning
- Add distributed training support

## License

This project is open source and available under the MIT License.
