lgb_params_1 = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_leaves": 15,  # 较小的叶子数，减少过拟合
    "learning_rate": 0.1,  # 较大的学习率，加快收敛
    "max_depth": 5,  # 较浅的树
    "n_estimators": 100,  # 适中的树数量
    "min_child_samples": 100,  # 增加叶子最小样本数，适合大数据
    "subsample": 0.8,  # 样本采样比例
    "colsample_bytree": 0.7,  # 特征采样比例，防止过拟合
    "random_state": 42,
    "verbose": -1,
}

lgb_params_2 = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_leaves": 31,  # 中等复杂度
    "learning_rate": 0.05,  # 中等学习率
    "max_depth": 6,  # 中等深度
    "n_estimators": 200,  # 增加树数量
    "min_child_samples": 50,  # 中等叶子样本数
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

lgb_params_3 = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_leaves": 63,  # 较大叶子数，模型更复杂
    "learning_rate": 0.03,  # 较小学习率，精细调整
    "max_depth": 8,  # 较深的树
    "n_estimators": 300,  # 更多树
    "min_child_samples": 20,  # 较小叶子样本数
    "subsample": 0.7,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "verbose": -1,
}

lr_params_1 = {
    "C": 0.1,  # 较强的正则化
    "max_iter": 2000,  # 增加迭代次数，确保收敛
    "random_state": 42,
    "solver": "lbfgs",  # 默认优化器
}

lr_params_2 = {
    "C": 1.0,  # 中等正则化
    "max_iter": 2000,
    "random_state": 42,
    "solver": "lbfgs",
}

lr_params_3 = {
    "C": 10.0,  # 较弱的正则化
    "max_iter": 2000,
    "random_state": 42,
    "solver": "lbfgs",
}
