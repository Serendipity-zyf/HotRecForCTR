# Product-based Neural Network (PNN) 模型详解

本文基于文章《Product-based Neural Networks for User Response Prediction》和 Datawhale 的相关资料，详细介绍了 PNN 模型的架构设计、数学原理、公式推导以及相关实现细节。

## 模型简介

Product-based Neural Network（PNN）是一种专为处理多字段（multi-field）分类数据设计的深度学习模型，适用于点击率（CTR）预测等任务。PNN 通过引入 **embedding 层**和 **product 层**来捕获不同字段之间的交互模式，并通过全连接层（MLP）进一步提取高阶特征交互信息。

PNN 的主要特点：

1. **Embedding 层**：将高维稀疏的 one-hot 编码特征转化为低维稠密向量表示。
2. **Product 层**：通过内积（inner product）或外积（outer product）捕获多字段特征之间的交互关系。
3. **全连接层**：进一步挖掘高阶特征交互模式。

PNN 的模型输出是一个位于 $[0, 1]$ 的概率值，表示用户点击广告的概率。

## 模型架构

PNN 模型包含以下几个主要模块（如图所示）：

### 1. 输入层

输入数据由多字段的分类特征组成。每个字段通过 one-hot 编码表示为高维稀疏二值向量，随后输入到嵌入层。

### 2. Embedding 层

将每个字段的 one-hot 编码特征映射到一个低维稠密向量空间。假设输入特征 $x \in \mathbb{R}^d$，Embedding 层的输出为：
$$
f_i = W_i x_i
$$
其中，$x_i$ 是字段 $i$ 的 one-hot 编码向量。$W_i \in \mathbb{R}^{M \times d_i}$ 是字段 $i$ 的嵌入矩阵，$M$ 是嵌入向量的维度。Embedding 层的输出是一个包含多个字段嵌入向量的集合：
$$
f = [f_1, f_2, \dots, f_N], \quad f_i \in \mathbb{R}^M
$$

### 3. Product 层

Product 层是 PNN 的核心模块，用于捕获字段之间的特征交互模式。PNN 提供了两种交互方式：

- **内积（Inner Product, IPNN）**：计算两个字段嵌入向量的内积，捕获局部依赖关系。
- **外积（Outer Product, OPNN）**：计算两个字段嵌入向量的外积，生成更丰富的交互信息。

首先，定义矩阵点积运算 $A\bigodot B\triangleq\sum_{i,j}A_{i,j}B_{i,j}$，记 $z=(z_1,z_2,\ldots,z_N)\triangleq(f_1,f_2,\ldots,f_N)$ ，其中 $f_i\in\mathbb{R}^M$ 表示经过 embedding 之后的特征向量。那么 IPNN 的计算如下：
$$
l_{z}=(l_{z}^{1},l_{z}^{2},\ldots,l_{z}^{n},\ldots,l_{z}^{D_{1}}),\quad l_{z}^{n}=W_{z}^{n}\bigodot z
$$
其中，$l_z^n =W_z^n \bigodot z =\sum_{i=1}^N\sum_{j=1}^M(W_z^n)_{i,j}z_{i,j}$。可以看出，这个部分的计算主要是为了保留低阶特征。$D_1$ 是一个超参数，表示 Product 层的输出维度。

#### 内积公式 (IPNN)

定义 $p_{i,j} = g(f_i, f_j) = \langle f_i, f_j \rangle$，将公式 (3) 进行改写，得:
$$
l_p^n = W_p^n \odot p = \sum_{i=1}^{N} \sum_{j=1}^{N} (W_p^n)_{i,j}p_{i,j} = \sum_{i=1}^{N} \sum_{j=1}^{N} (W_p^n)_{i,j}\langle f_i, f_j \rangle
$$

##### 时间空间复杂度分析

**空间复杂度** ：结合公式 (3) 可知，$l_z$ 计算空间复杂度为 $O(D_1 NM)$。结合公式 (4) 可知，计算 $p$ 需要 $O(N^2)$ 空间开销，$l_p^n$ 需要 $ O(N^2)$ 空间开销，所以 $l_p$ 计算空间复杂度为 $O(D_1 NN)$。所以，inner product 层整体计算空间复杂度为 $O(D_1 N(M + N))$。

**时间复杂度**：结合公式 (3) 可知，$l_z$ 计算时间复杂度为 $O(D_1 NM)$。结合公式 (4) 可知，计算 $p_{i,j}$ 需要 $O(M)$ 时间开销，计算 $p$ 需要 $O(N^2 M)$ 时间开销，又因为 $l_p^n$ 需要 $O(N^2)$ 时间开销，所以 $l_p$ 计算空间复杂度为 $O(N^2 (M + D_1))$。所以，inner product 层整体计算时间复杂度为 $O(N^2 (M + D_1))$。

#### IPNN 优化

受 FM 的参数矩阵分解启发，由于 $p_{i,j}$, $W_p^n$ 都是对称方阵，所以使用一阶矩阵分解，假设 $W_p^n = \theta^n \theta^{nT}$，此时有 $\theta^n \in \mathbb{R}^N$。将原本参数量为 $N * N$ 的矩阵 $W_p^n$，分解为了参数量为 $N$ 的向量 $\theta^n$。同时，将公式 (6) 改写为:

$$
\begin{aligned}
l_p^n &= W_p^n \odot p = \sum_{i=1}^{N} \sum_{j=1}^{N} (W_p^n)_{i,j}\langle f_i, f_j \rangle \\
&= \sum_{i=1}^{N} \sum_{j=1}^{N} \theta_i^n \theta_j^n \langle f_i, f_j \rangle \\
&= \sum_{i=1}^{N} \sum_{j=1}^{N} \langle \theta_i^n f_i, \theta_j^n f_j \rangle \\
&= \langle \sum_{i=1}^{N} \theta_i^n f_i, \sum_{j=1}^{N} \theta_j^n f_j \rangle \\
&= \langle \sum_{i=1}^{N} \delta_i^n, \sum_{j=1}^{N} \delta_j^n \rangle \\
&= \| \sum_{i=1}^{N} \delta_i^n \|^2
\end{aligned}
$$

其中: $\delta_i^n = \theta_i^n f_i$，$\delta_i^n \in \mathbb{R}^M$。结合公式 (3) (7)，得:

$$
l_p = (\| \sum_{i=1}^{N} \delta_i^1 \|^2, \cdots, \| \sum_{i=1}^{N} \delta_i^n \|^2, \cdots, \| \sum_{i=1}^{N} \delta_i^{D_1} \|^2)
$$
经过上述优化，IPNN 空间复杂度由 $O(D_1 N(M+N))$ 降为 $O(D_1 NM)$ 。时间复杂度由 $O(N^2 (M + D_1))$ 降为 $O(D_1 N M))$ 。虽然通过参数矩阵分解可以对计算开销进行优化，但由于采用一阶矩阵分解来近似矩阵结果，所以会丢失一定的精确性。

#### 外积公式 (OPNN)

OPNN 的计算如下：
$$
l_{p}=(l_{p}^{1},l_{p}^{2},\ldots,l_{p}^{n},\ldots,l_{p}^{D_{1}}),\quad l_{p}^{n}=W_{p}^{n}\bigodot p
$$
定义 $p_{i,j} = h(f_i, f_j) = f_i f_j^T$，将公式 (5) 进行改写：
$$
l_p^n=W_p^n\bigodot p=\sum_{i=1}^N\sum_{j=1}^N(W_p^n)_{i,j}p_{i,j}=\sum_{i=1}^N\sum_{j=1}^N(W_p^n)_{i,j}f_if_j^T
$$

##### 时间空间复杂度分析

类似于IPNN的分析，OPNN的时空复杂度均为 $O(D_1M^2N^2)$。

#### OPNN 优化

为了进行计算优化，引入叠加的概念（sum pooling）。将 $l_p$ 的计算公式重新定义为：
$$
p = \sum_{i=1}^{N} \sum_{j=1}^{N} f_i f_j^T = f_{\sum} f_{\sum}^T, \quad f_{\sum} = \sum_{i=1}^{N} f_i
$$
那么公式 (9) 重新定义为:（注意，此时 $p \in \mathbb{R}^{M \times M}$）
$$
l_p^n = W_p^n \odot p = \sum_{i=1}^{M} \sum_{j=1}^{M} (W_p^n)_{i,j}p_{i,j}
$$
通过公式 (10) 可知，$f_{\sum}$ 的时间复杂度为 $O(MN)$，$p$ 的时空复杂度均为 $O(MM)$，$l_p^n$ 的时空复杂度均为 $O(MM)$，那么计算 $l_p$ 的时空复杂度均为 $O(D_1 MM)$，从上一小节可知，计算 $l_z$ 的时空复杂度均为 $O(D_1 MN)$。所以最终OPNN的时空复杂度为$O(D_1 M(M + N))$。那么OPNN的时空复杂度由 $O(D_1 M^2 N^2)$ 降低到 $O(D_1 M(M + N))$。同样的，虽然叠加概念的引入可以降低计算开销，但是中间的精度损失也是很大的，性能与精度之间的tradeoff。

### 4. 全连接层

全连接层接收 Product 层的输出 是线性部分 $z$ 和交互部分 $p$ 的拼接，即 $l^{(0)}=[l_z;l_z]$，进一步提取高阶特征交互模式。每一层的输出为：
$$
l^{(k)} = \text{ReLU}(W^{(k)} l^{(k-1)} + b^{(k)})
$$
其中，$l^{(k)}$ 是第 $k$ 层的输出 ($l^{(0)}$ 是 Product 层的输出)。$W^{(k)}$ 和 $b^{(k)}$ 分别是第 $k$ 层的权重矩阵和偏置向量。ReLU 激活函数定义为：$\text{ReLU}(x) = \max(0, x)$。

### 5. 输出层

最终输出为一个预测概率值：
$$
\hat{y} = \sigma(W l^{(L)} + b)
$$
其中，$\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 $Sigmoid$ 激活函数。$l^{(L)}$ 是最后一层全连接层的输出。

### 优势

1. **高效捕获特征交互**：通过内积和外积显式地建模字段之间的二阶关系。
2. **易于扩展**：可通过设计更复杂的 Product 层提升模型能力。
3. **优异的性能**：在多个基准数据集上表现优于主流模型（如 LR、FM）。

### 缺点

1. **计算复杂度高**：原始的特征交互（尤其是外积操作）导致计算和存储成本显著增加。虽然有优化方法，但仍可能比简单模型复杂。
2. **容易过拟合**：参数量大，尤其在数据稀疏场景下，需要强正则化以防止过拟合。
3. **特征工程依赖**：仍需字段选择和预处理，难以完全摆脱人工干预。
4. **缺乏可解释性**：特征交互和嵌入表示难以直观解释，限制了模型的透明性。
5. **扩展性有限**：对动态字段和非结构化数据支持较弱，难以适应复杂场景。
6. **对超参数敏感**：模型性能依赖于嵌入维度、网络深度等超参数的调优。
7. **适用场景有限**：在简单线性关系或极高维特征场景下，可能不如传统模型（如 LR、FM）。

## 参考文件

[1] Qu, Yanru, et al. "Product-based neural networks for user response prediction." *2016 IEEE 16th International Conference on Data Mining (ICDM)*. IEEE, 2016.

[2] Zhang, Weinan, Tianming Du, and Jun Wang. "Deep learning over multi-field categorical data." *European conference on information retrieval*. Springer, Cham, 2016.

[3] https://zhuanlan.zhihu.com/p/56651241

[4] https://zhuanlan.zhihu.com/p/89850560