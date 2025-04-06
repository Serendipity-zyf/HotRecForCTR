## 参数说明与调整建议
### LightGBM 参数 (`lgb_params`)
"""
- `num_leaves`：控制树复杂度，越大模型越复杂，但容易过拟合（默认31，可尝试20-50）。
- `learning_rate`：学习率，越小训练越慢但可能更精确（默认0.05，可尝试0.01-0.1）。
- `n_estimators`：树的数量，配合早停可以设大一些（默认100，可尝试50-500）。
- `max_depth`：树的最大深度，控制过拟合（默认6，可尝试4-8）。
- `subsample` 和 `colsample_bytree`：采样比例，防止过拟合（默认0.8，可尝试0.6-1.0）。
- `early_stopping_rounds`：早停轮数，建议设为10-20。
"""

### Logistic Regression 参数 (`lr_params`)
"""
- `C`：正则化强度的倒数，值越大正则化越弱（默认1.0，可尝试0.01-10）。
- `max_iter`：增加这个值如果模型未收敛（默认1000）。
- `solver`：'lbfgs' 适用于大多数情况，若数据量很大可试 'sag' 或 'saga'。
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from config import lgb_params_1, lgb_params_2
from config import lr_params_1, lr_params_2, lr_params_3


# 加载数据
def load_df_parquet(filename_base: str = "criteo_data") -> Union[pd.DataFrame, Dict]:
    # 加载 DataFrame
    df = pd.read_parquet(f"../data/{filename_base}.parquet")
    # 加载编码器
    with open(f"../data/{filename_base}_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    return df, label_encoders


# 加载数据
trans_train_data, label_encoders = load_df_parquet("criteo_data")
# 提取特征和标签
X = trans_train_data.iloc[:, 1:].values  # 特征（1-39 列）
y = trans_train_data.iloc[:, 0].values  # 标签（0 列）

# 分离连续特征和离散特征
continuous_features = X[:, :13]  # 前13列是连续特征
categorical_features = X[:, 13:]  # 第14列开始是离散特征(已经经过LabelEncoder)

# 归一化连续特征
scaler = MinMaxScaler()  # [0, 1] 归一化
continuous_features_scaled = scaler.fit_transform(continuous_features)

# 为 LightGBM 准备数据 - 直接使用 LabelEncoder 后的特征
X_for_lgb = np.hstack([continuous_features_scaled, categorical_features])

# 为 Logistic Regression 准备数据 - 使用 One-Hot 编码
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
_ = ohe.fit_transform(categorical_features)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_for_lgb, y, test_size=0.2, random_state=42
)

# 输出数据集信息
print(f"训练集样本数：{len(X_train)}")
print(f"验证集样本数：{len(X_val)}")
print(f"训练集NaN数量：{pd.isna(X_train).sum()}")
print(f"验证集NaN数量：{pd.isna(X_val).sum()}")

# 训练 LightGBM 和 Logistic Regression 模型
import lightgbm as lgb
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 定义候选参数
lgb_param_candidates = [lgb_params_1, lgb_params_2]
lr_param_candidates = [lr_params_1, lr_params_2, lr_params_3]

# 存储结果
results = []

# 在模型训练部分
for i, lgb_params in enumerate(lgb_param_candidates):
    print(f"\nTesting LGB Params {i + 1}")
    # 1. 训练LightGBM
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

    # 2. 获取叶子索引
    X_train_leaves = lgb_model.predict(X_train, pred_leaf=True)
    X_val_leaves = lgb_model.predict(X_val, pred_leaf=True)

    # 3. 分别为训练集和验证集中的分类特征创建稀疏One-Hot编码
    X_train_cat = X_train[:, 13:]
    X_val_cat = X_val[:, 13:]
    X_train_cat_onehot = ohe.transform(X_train_cat)  # 返回稀疏矩阵
    X_val_cat_onehot = ohe.transform(X_val_cat)  # 返回稀疏矩阵

    # 4. 对连续特征和叶子节点特征转换为稀疏矩阵
    X_train_cont_sparse = sparse.csr_matrix(X_train[:, :13])
    X_val_cont_sparse = sparse.csr_matrix(X_val[:, :13])
    X_train_leaves_sparse = sparse.csr_matrix(X_train_leaves)
    X_val_leaves_sparse = sparse.csr_matrix(X_val_leaves)

    # 5. 使用稀疏矩阵水平连接方法
    X_train_combined = sparse.hstack(
        [X_train_cont_sparse, X_train_cat_onehot, X_train_leaves_sparse]
    )
    X_val_combined = sparse.hstack(
        [X_val_cont_sparse, X_val_cat_onehot, X_val_leaves_sparse]
    )
    for j, lr_params in enumerate(lr_param_candidates):
        print(f"\nTesting with LR Params {j + 1}")
        # 6. 训练 LR（ LogisticRegression 默认支持稀疏输入）
        lr_model = LogisticRegression(**lr_params)
        lr_model.fit(X_train_combined, y_train)

        # 7. 预测和评估
        y_val_pred = lr_model.predict_proba(X_val_combined)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred)
        print(f"AUC: {auc:.4f}")

        # 记录结果
        results.append({"lgb_params": i + 1, "lr_params": j + 1, "auc": auc})

# 找出最佳组合
best_result = max(results, key=lambda x: x["auc"])
print("\nBest Combination:")
print(
    f"LGB Params {best_result['lgb_params']}, LR Params {best_result['lr_params']}, AUC: {best_result['auc']:.4f}"
)

# Output:Best Combination:
# LGB Params 2, LR Params 1, AUC: 0.5491
