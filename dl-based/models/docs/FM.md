# 使用因子分解机 (Factorization Machines, FM) 进行 CTR 预测

## 1. 引言

### 1.1 CTR 预测问题

点击率 (Click-Through Rate, CTR) 预测是计算广告、推荐系统等领域中的核心问题之一。它的目标是预测用户点击一个特定项目（如广告、商品、新闻）的概率。CTR 预测的准确性直接影响平台的收入和用户体验。CTR 预测通常被建模为一个二分类问题，输入是描述用户、项目和上下文信息的特征，输出是用户点击该项目的概率。

### 1.2 传统方法的局限性

*   **逻辑回归 (Logistic Regression, LR):** LR 是 CTR 预测中常用的基线模型。它是一个广义线性模型，易于理解和实现。但 LR 难以捕捉特征之间的交互关系，需要大量手动进行特征交叉工程。对于高维稀疏数据，二阶特征交叉的参数数量会爆炸式增长（复杂度为 $O(n^2)$，其中 $n$ 为特征数量）。
*   **带交叉项的多项式模型 (Polynomial Regression):** 虽然可以显式地引入二阶（或更高阶）特征交叉项，但面临以下问题：
    *   **参数数量过多:** 参数数量为 $O(n^2)$，训练困难且容易过拟合。
    *   **数据稀疏性:** 在高度稀疏的数据中（如 one-hot 编码后的 ID 类特征），许多特征交叉项（$x_i x_j$）的值恒为 0，导致对应的交叉项权重 $w_{ij}$ 无法通过训练得到有效学习。模型无法泛化到训练集中未出现过的特征组合。

## 2. FM 模型原理

### 2.1 模型思想
**因子分解机 (Factorization Machines, FM)** 由 Steffen Rendle 于 2010 年提出，旨在解决上述问题，尤其是在高维稀疏数据场景下有效学习特征之间的交互关系。FM 模型的核心思想是**将二阶交叉项的权重 $w_{ij}$ 分解为两个低维隐向量 $\mathbf{v}_i$ 和 $\mathbf{v}_j$ 的点积**。即：

$$
w_{ij} \approx \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{if} v_{jf}
$$

其中：
*   $\mathbf{v}_i$ 是第 $i$ 个特征对应的 $k$ 维隐向量 $\mathbf{v}_i = (v_{i1}, v_{i2}, \dots, v_{ik})$。
*   $\mathbf{v}_j$ 是第 $j$ 个特征对应的 $k$ 维隐向量 $\mathbf{v}_j = (v_{j1}, v_{j2}, \dots, v_{jk})$。
*   $k$ 是隐向量的维度，通常远小于特征数量 $n$, 即 $k \ll n$。
*   $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 表示两个隐向量的点积。

通过这种分解：
1.  **参数数量减少:** 原本需要学习 $O(n^2)$ 个二阶交叉项权重 $w_{ij}$，现在只需要学习 $O(n \cdot k)$ 个隐向量参数 ($v_{if}$)。当 $k \ll n$ 时，参数数量大大减少。
2.  **解决数据稀疏性:** 即使特征 $i$ 和特征 $j$ 在训练集中从未同时出现过（即 $x_i x_j = 0$ 对所有样本成立），模型仍然可以通过它们各自与其他特征的交互（如 $i$ 与 $m$ 的交互学习 $\mathbf{v}_i$， $j$ 与 $p$ 的交互学习 $\mathbf{v}_j$）来学习到它们各自的隐向量 $\mathbf{v}_i$ 和 $\mathbf{v}_j$。这样，即使 $(i, j)$ 组合未见过，其交互权重 $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 也能被估算出来，从而提高了模型的泛化能力。

### 2.2 FM 模型方程 (Degree=2)

二阶 FM 模型的预测方程如下：

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
$$

其中：
*   $\mathbf{x} = (x_1, x_2, \dots, x_n)$ 是 $n$ 维特征向量。
*   $w_0$ 是全局偏置项（截距）。
*   $w_i$ 是第 $i$ 个特征的线性权重（一阶项）。
*   $\mathbf{v}_i$ 是第 $i$ 个特征的 $k$ 维隐向量。
*   $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \sum_{f=1}^{k} v_{if} v_{jf}$ 是第 $i$ 个特征和第 $j$ 个特征交互的权重。

这个模型结合了线性回归的优势（一阶项）和特征交互建模的优势（二阶项），并且通过**隐向量分解**解决了参数数量和稀疏性问题。

### 2.3 计算复杂度优化 (FM 核心技巧)

直接计算二阶交叉项 $\sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$ 的复杂度是 $O(k n^2)$，在 $n$ 很大时仍然非常耗时。FM 的一个关键贡献在于推导出该项的线性时间复杂度 **$O(k n)$** 的计算方法。

推导过程如下：

$$
\begin{aligned}\begin{aligned}\sum_{i=1}^{n-1}\sum_{j=i+1}^n&<v_i,v_j>x_ix_j\end{aligned}&=\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n<v_i,v_j>x_ix_j-\frac{1}{2}\sum_{i=1}^n<v_i,v_i>x_ix_i\\&=\frac{1}{2}\left(\sum_{i=1}^n\sum_{j=1}^n\sum_{f=1}^kv_{i,f}v_{j,f}x_ix_j-\sum_{i=1}^n\sum_{f=1}^kv_{i,f}v_{i,f}x_ix_i\right)\\&=\frac{1}{2}\sum_{f=1}^k\left[\left(\sum_{i=1}^nv_{i,f}x_i\right)\cdot\left(\sum_{j=1}^nv_{j,f}x_j\right)-\sum_{i=1}^nv_{i,f}^2x_i^2\right]\\&=\frac{1}{2}\sum_{f=1}^k\left[\left(\sum_{i=1}^nv_{i,f}x_i\right)^2-\sum_{i=1}^nv_{i,f}^2x_i^2\right]\end{aligned}
$$


**这个公式的意义:**

计算 $(\sum_{i=1}^{n} v_{if} x_i)^2$ 的复杂度是 $O(n)$，计算 $\sum_{i=1}^{n} (v_{if} x_i)^2$ 的复杂度也是 $O(n)$。由于需要对 $k$ 个维度进行计算，总的二阶项计算复杂度降为 **$O(k n)$**。

因此，整个 FM 模型的预测计算复杂度也是 $O(k n)$，非常高效。

## 3. 模型训练 (优化求解)

### 3.1 目标函数 (以 CTR 预测为例)

CTR 预测通常是二分类问题，可以使用**对数损失函数 (LogLoss)**，也称为二元交叉熵损失。FM 的输出 $\hat{y}(\mathbf{x})$ 需要通过 Sigmoid 函数 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 转换为概率 $p(y=1|\mathbf{x}) = \sigma(\hat{y}(\mathbf{x}))$。

对于单个样本 $(\mathbf{x}, y)$（其中 $y$ 是真实标签，0 或 1），损失为：
$$
\text{Loss}(\hat{y}(\mathbf{x}), y) = -[ y \log(\sigma(\hat{y}(\mathbf{x}))) + (1-y) \log(1 - \sigma(\hat{y}(\mathbf{x}))) ]
$$

对于整个训练集 $D = \{(\mathbf{x}^{(m)}, y^{(m)})\}_{m=1}^{M}$，总损失为：
$$
L(W, V) = \sum_{m=1}^{M} \text{Loss}(\hat{y}(\mathbf{x}^{(m)}), y^{(m)})
$$

为了防止过拟合，通常会加入**正则化项** (L2 正则化比较常用)：
$$
L_{reg}(W, V) = L(W, V) + \frac{\lambda_w}{2} ||\mathbf{w}||^2 + \sum_{f=1}^{k} \frac{\lambda_v}{2} ||\mathbf{v}_f||^2
$$
或者更具体地写为：
$$
L_{reg}(W, V) = \sum_{m=1}^{M} \text{Loss}(\hat{y}(\mathbf{x}^{(m)}), y^{(m)}) + \frac{\lambda_w}{2} \sum_{i=1}^{n} w_i^2 + \sum_{i=1}^{n} \sum_{f=1}^{k} \frac{\lambda_v}{2} v_{if}^2
$$
(注：通常不对偏置项 $w_0$ 进行正则化)

其中 $\lambda_w$ 和 $\lambda_v$ 是 L2 正则化系数。

### 3.2 优化算法 (SGD)

由于 CTR 预测的数据集通常非常大，**随机梯度下降 (Stochastic Gradient Descent, SGD)** 及其变种 (如 AdaGrad, Adam) 是常用的优化算法。SGD 每次使用单个样本或一小批 (mini-batch) 样本来更新模型参数。

更新规则：
$$ \theta \leftarrow \theta - \eta \nabla_{\theta} L_{reg} $$
其中 $\theta$ 代表模型中任意一个参数 ($w_0$, $w_i$, 或 $v_{if}$), $\eta$ 是学习率, $\nabla_{\theta} L_{reg}$ 是带正则化的损失函数对参数 $\theta$ 的梯度。

### 3.3 梯度计算

计算损失函数 $L_{reg}$ 对每个参数的梯度是 SGD 的关键。我们首先计算损失函数对模型预测值 $\hat{y} = \hat{y}(\mathbf{x})$ 的导数（对于单个样本，省略上标 $(m)$ 和参数 $\mathbf{x}$）：

$$
\frac{\partial L}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} [-y \log(\sigma(\hat{y})) - (1-y) \log(1-\sigma(\hat{y}))] = \sigma(\hat{y}) - y
$$

然后，使用链式法则计算对各参数的梯度：
$$
\frac{\partial L_{reg}}{\partial \theta} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial \theta} + \frac{\partial (\text{Regularization Term})}{\partial \theta}
$$

我们需要计算 $\hat{y}(\mathbf{x})$ 对各参数的导数：

1.  **对 $w_0$ 的导数:**
    $$ \frac{\partial \hat{y}}{\partial w_0} = 1 $$

2.  **对 $w_i$ 的导数:**
    $$ \frac{\partial \hat{y}}{\partial w_i} = x_i $$

3.  **对 $v_{if}$ 的导数:** (利用 $O(kn)$ 的计算公式形式)
    $$ \frac{\partial \hat{y}}{\partial v_{if}} = \frac{\partial}{\partial v_{if}} \left( \frac{1}{2} \sum_{f'=1}^{k} \left[ \left( \sum_{j=1}^{n} v_{jf'} x_j \right)^2 - \sum_{j=1}^{n} (v_{jf'} x_j)^2 \right] \right) $$
    当 $f'=f$ 时，内部项对 $v_{if}$ 有关，其他项无关。
    $$ \frac{\partial \hat{y}}{\partial v_{if}} = \frac{1}{2} \left[ 2 \left( \sum_{j=1}^{n} v_{jf} x_j \right) x_i - 2 v_{if} x_i^2 \right] = x_i \sum_{j=1}^{n} v_{jf} x_j - v_{if} x_i^2 $$
    (这里用 $j$ 作为内层求和的索引)

**结合链式法则，得到 SGD 更新所需的梯度 (包含 L2 正则化项)：**

*   **更新 $w_0$:** (通常不加正则化)
    $$ \nabla_{w_0} L_{reg} = (\sigma(\hat{y}) - y) \cdot 1 + 0 = \sigma(\hat{y}) - y $$
    $$ w_0 \leftarrow w_0 - \eta (\sigma(\hat{y}) - y) $$

*   **更新 $w_i$:**
    $$ \nabla_{w_i} L_{reg} = (\sigma(\hat{y}) - y) \cdot x_i + \lambda_w w_i $$
    $$ w_i \leftarrow w_i - \eta [(\sigma(\hat{y}) - y) x_i + \lambda_w w_i] $$

*   **更新 $v_{if}$:**
    $$ \nabla_{v_{if}} L_{reg} = (\sigma(\hat{y}) - y) \left( x_i \sum_{j=1}^{n} v_{jf} x_j - v_{if} x_i^2 \right) + \lambda_v v_{if} $$
    $$ v_{if} \leftarrow v_{if} - \eta \left[ (\sigma(\hat{y}) - y) \left( x_i \sum_{j=1}^{n} v_{jf} x_j - v_{if} x_i^2 \right) + \lambda_v v_{if} \right] $$

**SGD 实现注意:**
对于稀疏输入 $\mathbf{x}$，只有当 $x_i \neq 0$ 时，对应的参数 $w_i$ 和 $\mathbf{v}_i$ (即所有 $v_{if}, f=1..k$) 才需要计算梯度和更新。这使得每次更新的计算量远小于 $O(k n)$。在计算 $\sum_{j=1}^{n} v_{jf} x_j$ 时也只需要对非零特征 $x_j$ 进行求和。

## 4. 工程实现注意事项

### 4.1 数据预处理与特征工程

*   **特征类型:** FM 可以处理数值型 (Numerical) 和类别型 (Categorical) 特征。
*   **类别型特征:** 通常需要进行 **One-Hot 编码** 转换为稀疏的 0/1 特征。例如，`UserID=A` 变为 `feature_UserID_A=1`，其他 `UserID` 特征为 0。
*   **高基数类别特征:** 当类别数量非常大时（如 User ID），One-Hot 编码会导致维度爆炸。可以考虑：
    *   **Hashing Trick:** 将原始特征哈希到固定数量的桶中，可能引起冲突但能控制维度。
    *   **特征筛选/组合:** 基于频率、业务意义等进行筛选或组合。
*   **数值型特征:**
    *   **归一化/标准化:** 对于某些优化算法（尤其是对尺度敏感的）可能有助于收敛。可以尝试 Min-Max Scaling 或 Z-Score Standardization。
    *   **离散化/分桶:** 将连续特征转化为类别型特征（如年龄分段），有时能提升模型效果，捕捉非线性关系。
*   **特征交叉:** FM 自动处理二阶特征交叉。但显式地加入一些有意义的高阶交叉特征或领域知识特征有时也能提升效果。

### 4.2 模型训练细节

*   **参数初始化:**
    *   $w_0$: 可以初始化为 0 或训练数据的平均目标值的对数几率 (log-odds)。
    *   $w_i$: 初始化为 0。
    *   $v_{if}$: 初始化为小的随机值（例如，从均值为 0，标准差较小的高斯分布 $N(0, \sigma^2)$ 中采样），避免对称性导致所有隐向量更新一致。$\sigma$ 通常是一个小的数，比如 0.01。
*   **学习率 $\eta$:** 非常关键的超参数。需要仔细调整（如通过网格搜索、随机搜索或贝叶斯优化）。可以使用学习率衰减策略（如随迭代次数减小：$\eta_t = \eta_0 / (1 + \text{decay} \cdot t)$）或自适应学习率算法 (AdaGrad, Adam)。
*   **正则化系数 $\lambda_w, \lambda_v$:** 控制模型复杂度，防止过拟合。也需要调优，通常通过交叉验证选择。
*   **隐向量维度 $k$:** 影响模型的表达能力和计算复杂度。通常需要通过交叉验证来选择。一般从较小的值（如 8, 16, 32, 64）开始尝试。
*   **优化器选择:** **Adam** 或 **AdaGrad** 通常比朴素 SGD 收敛更快、效果更好，尤其是在稀疏数据上，因为它们能为不同参数调整学习率。
*   **训练停止策略:**
    *   **固定迭代次数:** 简单但不一定最优。
    *   **早停 (Early Stopping):** 在每次迭代（或每 N 次迭代）后，在独立的验证集上评估模型性能（如 **LogLoss** 或 **AUC**）。当验证集性能不再提升或开始下降时停止训练，并选用验证集上性能最好的模型。这是防止过拟合的常用有效手段。
*   **Mini-Batch:** 使用小批量数据进行梯度更新，平衡了 SGD 的随机性和 Batch GD 的稳定性，并且能利用硬件并行性。Batch Size 也是一个需要调整的超参数（如 32, 64, 128, 256, ...）。

### 4.3 预测与评估

*   **预测:** 使用训练好的参数 $w_0, w_i, v_{if}$ 和 $O(kn)$ 的计算公式计算 $\hat{y}(\mathbf{x})$，然后通过 Sigmoid 函数 $p = \sigma(\hat{y}(\mathbf{x}))$ 得到 CTR 预测值 $p \in [0, 1]$。
*   **评估指标:**
    *   **AUC (Area Under the ROC Curve):** 衡量模型的排序能力，对类别不平衡不敏感，是 CTR 预测中最常用的指标之一。
    *   **LogLoss (Logarithmic Loss / Cross-Entropy):** 衡量预测概率与真实标签的差异，对预测概率的准确性敏感。值越小越好。
    *   **Calibration Plot:** 检查预测概率与实际观测到的点击率是否一致（例如，将预测概率分桶，看每个桶内的平均预测概率是否接近该桶内样本的实际点击率）。

## 5. FM 的优缺点

### 5.1 优点

1.  **有效处理稀疏数据:** 能够学习稀疏特征之间的交互，具有良好的泛化能力。
2.  **自动学习特征交叉:** 避免了手动组合大量交叉特征的繁琐工作，自动学习所有特征对的交互。
3.  **计算效率高:** 模型预测和训练（使用 $O(kn)$ 公式和 SGD）的复杂度是线性的 ($O(kn)$ 或 $O(k \cdot |\mathbf{x}|_{nz})$，其中 $|\mathbf{x}|_{nz}$ 是非零特征数)。
4.  **表达能力强于线性模型:** 能捕捉二阶特征交互信息。

### 5.2 缺点

1.  **仅限于二阶交叉:** 标准 FM 主要捕捉二阶特征交互，对更高阶的复杂交互模式可能捕捉不足。
2.  **所有特征交互共享隐向量维度 $k$:** 对所有特征对使用相同维度的隐向量可能不是最优的（FFM 对此进行了改进）。
3.  **效果可能被深度学习模型超越:** 对于极其复杂的数据模式和需要捕获高阶交互的场景，深度学习模型（如 DeepFM, NFM, xDeepFM 等）可能表现更好。
