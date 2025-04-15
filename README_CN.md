# HotRecForCTR: CTR预估常用排序模型 🚀

中文 | [English](README.md)

一个包含点击率（CTR）预估常用排序模型的代码库，同时实现了传统机器学习和深度学习方法。使用Criteo数据集作为基准测试。

## 📋 项目概述

本项目实现了各种CTR预估模型，重点关注：
- 🔍 传统机器学习方法（LR-GBDT）
- 🧠 深度学习模型（FM）
- 🧩 基于注册机制的模块化、可扩展架构
- 🛠️ 日志记录、进度跟踪和模型分析的实用工具模块

## 📁 代码库结构

```
.
├── data/                  # 数据处理和存储
├── ml-based/              # 基于机器学习的模型
└── dl-based/              # 基于深度学习的模型
    ├── config/            # 模型配置
    ├── datasets/          # 数据集实现
    ├── models/            # 模型实现
    ├── utils/             # 实用工具模块
    └── torchprint/        # 模型分析工具
```

## ✨ 主要特点

### 模型
- **LR-GBDT**：结合GBDT进行特征转换与LR的混合模型
- **FM（因子分解机）**：捕获特征间的二阶交互

### 工具
- **注册机制**：用于模型、数据集等组件的管理系统
- **ColorLogger**：具有多级别的自定义彩色日志
- **ProgressBar**：基于alive_progress的增强进度跟踪
- **TorchPrint**：PyTorch模型分析（参数、FLOPs、内存使用）

## 🔧 环境设置

```bash
# 创建Python 3.10的新conda环境
conda create -n reco python=3.10

# 激活环境
conda activate reco

# 安装所需包
pip install pandas numpy scikit-learn lightgbm torch jupyter colorama alive-progress pydantic
```

## 🚀 快速开始

```bash
# 激活环境
conda activate reco

# 运行机器学习模型
cd ml-based
python lr_gbdt.py

# 运行深度学习模型（交互式组件选择）
cd ../dl-based
python main.py
```

## 📊 数据集

使用[Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)数据集，包含：
- 13个连续特征
- 26个类别特征
- 二元点击标签（1表示点击，0表示未点击）

## 📦 依赖项

- Python 3.10+
- PyTorch, pandas, numpy, scikit-learn
- LightGBM, colorama, alive-progress
- Pydantic（用于配置验证）

## 🔮 未来工作

- 更多深度学习模型（DeepFM, Wide & Deep, DCN, DIN）
- 评估指标和可视化工具
- 超参数调优
- 分布式训练支持
