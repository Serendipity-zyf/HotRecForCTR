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
    └── fm.ipynb           # Factorization Machine implementation
```

## Models Implemented

### Machine Learning Based Models

- **LR-GBDT (Logistic Regression with Gradient Boosting Decision Trees)**
  - A hybrid model that combines the power of GBDT for feature transformation with LR for prediction
  - Implements parameter tuning with different configurations
  - Uses LightGBM as the GBDT implementation

### Deep Learning Based Models

- **FM (Factorization Machines)**
  - A model that can capture pairwise feature interactions
  - Implemented in a Jupyter notebook for interactive exploration

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

#### Deep Learning Models

Open and run the Jupyter notebooks in the `dl-based` directory:
```
jupyter notebook dl-based/fm.ipynb
```

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- LightGBM
- PyTorch (for deep learning models)
- Jupyter

## Future Work

- Implement more deep learning models:
  - DeepFM
  - Wide & Deep
  - DCN (Deep & Cross Network)
  - DIN (Deep Interest Network)
- Add evaluation metrics and visualization tools
- Implement hyperparameter tuning
- Add distributed training support

## License

This project is open source and available under the MIT License.
