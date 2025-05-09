### LightGBM Parameters (`lgb_params`)
"""
- `num_leaves`: Controls tree complexity; larger values make the model more complex but prone to overfitting (default: 31, try 20-50).
- `learning_rate`: Learning rate; smaller values slow training but may improve accuracy (default: 0.05, try 0.01-0.1).
- `n_estimators`: Number of trees; can be set larger with early stopping (default: 100, try 50-500).
- `max_depth`: Maximum tree depth to control overfitting (default: 6, try 4-8).
- `subsample` and `colsample_bytree`: Sampling ratios to prevent overfitting (default: 0.8, try 0.6-1.0).
- `early_stopping_rounds`: Number of rounds for early stopping; recommended to set between 10-20.
"""

### Logistic Regression Parameters (`lr_params`)
"""
- `C`: Inverse of regularization strength; larger values mean weaker regularization (default: 1.0, try 0.01-10).
- `max_iter`: Increase this value if the model doesn’t converge (default: 1000).
- `solver`: 'lbfgs' works for most cases; try 'sag' or 'saga' for very large datasets.
"""

import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from config import lgb_params_1, lgb_params_2, lr_params_1, lr_params_2, lr_params_3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Load data
def load_df_parquet(filename_base: str = "criteo_data") -> Union[pd.DataFrame, Dict]:
    # Load DataFrame
    df = pd.read_parquet(f"../data/{filename_base}.parquet")
    # Load encoders
    with open(f"../data/{filename_base}_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    return df, label_encoders


# Load data
trans_train_data, label_encoders = load_df_parquet("criteo_data")
# Extract features and labels
X = trans_train_data.iloc[:, 1:].values  # Features (columns 1-39)
y = trans_train_data.iloc[:, 0].values  # Labels (column 0)

# Separate continuous and categorical features
continuous_features = X[:, :13]  # First 13 columns are continuous features
categorical_features = X[:, 13:]  # From column 14 onward are categorical features

# Normalize continuous features
scaler = MinMaxScaler()  # [0, 1] normalization
continuous_features_scaled = scaler.fit_transform(continuous_features)

# Prepare data for LightGBM - Use features after LabelEncoder
X_for_lgb = np.hstack([continuous_features_scaled, categorical_features])

# Prepare data for Logistic Regression - Use One-Hot encoding
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
_ = ohe.fit_transform(categorical_features)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_for_lgb, y, test_size=0.2, random_state=42
)

# Output dataset information
print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_val)}")
print(f"Number of NaNs in training set: {pd.isna(X_train).sum()}")
print(f"Number of NaNs in validation set: {pd.isna(X_val).sum()}")

# Train LightGBM and Logistic Regression models
import lightgbm as lgb
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Define candidate parameters
lgb_param_candidates = [lgb_params_1, lgb_params_2]
lr_param_candidates = [lr_params_1, lr_params_2, lr_params_3]

# Store results
results = []

# In the model training section
for i, lgb_params in enumerate(lgb_param_candidates):
    print(f"\nTesting LGB Params {i + 1}")
    # 1. Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=200),
        ],
    )

    # 2. Get leaf indices
    X_train_leaves = lgb_model.predict(X_train, pred_leaf=True)
    X_val_leaves = lgb_model.predict(X_val, pred_leaf=True)

    # 3. Create sparse One-Hot encoding for categorical features in training and validation sets
    X_train_cat = X_train[:, 13:]
    X_val_cat = X_val[:, 13:]
    X_train_cat_onehot = ohe.transform(X_train_cat)
    X_val_cat_onehot = ohe.transform(X_val_cat)

    # 4. Convert continuous features and leaf node features to sparse matrices
    X_train_cont_sparse = sparse.csr_matrix(X_train[:, :13])
    X_val_cont_sparse = sparse.csr_matrix(X_val[:, :13])
    X_train_leaves_sparse = sparse.csr_matrix(X_train_leaves)
    X_val_leaves_sparse = sparse.csr_matrix(X_val_leaves)

    # 5. Horizontally stack sparse matrices
    X_train_combined = sparse.hstack(
        [X_train_cont_sparse, X_train_cat_onehot, X_train_leaves_sparse]
    )
    X_val_combined = sparse.hstack(
        [X_val_cont_sparse, X_val_cat_onehot, X_val_leaves_sparse]
    )
    for j, lr_params in enumerate(lr_param_candidates):
        print(f"\nTesting with LR Params {j + 1}")
        # 6. Train LR (LogisticRegression supports sparse input by default)
        lr_model = LogisticRegression(**lr_params)
        lr_model.fit(X_train_combined, y_train)

        # 7. Predict and evaluate
        y_val_pred = lr_model.predict_proba(X_val_combined)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred)
        print(f"AUC: {auc:.4f}")

        # Record results
        results.append({"lgb_params": i + 1, "lr_params": j + 1, "auc": auc})

# Find the best combination
best_result = max(results, key=lambda x: x["auc"])
print("\nBest Combination:")
print(
    f"LGB Params {best_result['lgb_params']}, LR Params {best_result['lr_params']}, AUC: {best_result['auc']:.4f}"
)

# Best Combination:
# LGB Params 1, LR Params 1, AUC: 0.7808
