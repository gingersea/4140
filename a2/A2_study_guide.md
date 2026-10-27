# AIST4010 Assignment 2 — 学习指导 / Study Guide

**Name: JIANG Huanyun | SID: 1155233165**

本文件是作业解答的配套"学习指导"：中文为主、关键术语附英文。
目标不是让你"背答案"，而是让你：
1. 知道每道题在考察什么背景知识；
2. 理解解答中每一句话为什么成立；
3. 拿到类似的题能自己推导出来。

This guide accompanies the solutions. The answers themselves are in
`A2_report.tex`/`A2_report.pdf` (in English). Here we explain, in Chinese
with English key terms, the background knowledge behind every question and
how to think about it so that you truly understand the assignment.

---

## 第 1 题：梯度消失与梯度爆炸（Gradient Vanishing / Explosion）

### 考察什么
反向传播（backpropagation）的基本原理、链式法则（chain rule）、
激活函数的导数特性、以及深层网络训练的稳定性问题。
这是深度学习中"为什么网络训不动"的最经典问题。

### 需要的背景知识

1. **链式法则**：损失 $L$ 对第 $l$ 层参数 $W^{(l)}$ 的梯度是
   $\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial h^{(L)}} \cdot \prod_{k=l+1}^{L} \frac{\partial h^{(k)}}{\partial h^{(k-1)}} \cdot \frac{\partial h^{(l)}}{\partial W^{(l)}}$。
   注意中间是**一连串 Jacobian 的连乘**。连乘是理解一切的关键：
   如果每项都小于 1，乘积指数级变小；如果每项都大于 1，乘积指数级变大。

2. **激活函数的导数**：
   - sigmoid：$\sigma'(x) = \sigma(x)(1-\sigma(x))$，最大值 0.25；
   - tanh：$\tanh'(x) = 1-\tanh^2(x)$，最大值 1（在 $x=0$ 处），饱和区趋近 0；
   - ReLU：$x>0$ 时导数为 1，$x<0$ 时为 0。
   在深层网络里，每经过一层 sigmoid，梯度最多乘 0.25，十层后几乎归零 → 梯度消失。
   若权重矩阵范数很大，每层乘一个大于 1 的因子 → 梯度爆炸。

3. **为什么"消失/爆炸"是问题**：梯度是参数更新的依据。
   梯度≈0 → 前面的层几乎不学习，训练停滞；
   梯度→∞ → 参数更新步子过大，权重变成 NaN/Inf，训练发散。

### 怎么理解解答里的解法（记忆逻辑：从"激活函数、初始化、归一化、结构、优化器"五个角度）

| 解法 | 原理（为什么有效） |
|------|------------------|
| 换 ReLU 等激活函数 | 正区导数恒为 1，连乘不再指数衰减 |
| 好的初始化（Xavier/He） | 让每层输出的方差保持不变，连乘因子 ≈1 |
| BatchNorm / LayerNorm | 把每层输入强行拉回非饱和区，信号不会衰减或放大 |
| 梯度裁剪（gradient clipping） | 直接给梯度范数设上限，专治爆炸 |
| 残差连接（ResNet）/ LSTM 门控 | 提供一条梯度为 1 的"捷径"，深层也能收到信号 |
| Adam / RMSProp | 每一步按梯度历史自动调整步长，对梯度大小不敏感 |

### 自测
- 为什么 ReLU 不能完全解决梯度消失？（负数区导数为 0 → "死神经元"，所以有 Leaky ReLU 等变体）
- 如果训练时 loss 变成 NaN，最先怀疑谁？（学习率过大 → 梯度爆炸）
- 我们 Q5 实验里 sigmoid 版 LeNet 准确率 10%（瞎猜）——就是第 1 题活生生的例子。

---

## 第 2 题：2D 卷积 vs 3D 卷积

### 考察什么
卷积神经网络（CNN）的基本概念：卷积核、滑窗、感受野（receptive field）、
权值共享（weight sharing）、平移等变性（translation equivariance），
以及对不同数据维度（图像 vs 视频/体数据）的建模方式。

### 需要的背景知识

1. **卷积的本质**：一个小核（kernel）在输入上滑动，每个位置做局部加权求和。
   2D 核是 $k_h \times k_w$（在高、宽两个方向滑）；
   3D 核是 $k_d \times k_h \times k_w$（多一个深度/时间方向）。

2. **共同点**（similarities）：局部连接 + 权值共享 + 平移等变性 + 参数少 + 多层堆叠学层次特征
   （浅层学边缘，深层学部件和整体）。输出尺寸都按 $(n+2p-k)/s+1$ 算。

3. **不同点**（differences）：
   - 输入数据：2D → 静态图像；3D → 视频（帧序列）或 CT/MRI 体数据；
   - 学到的东西：2D 只学空间模式；3D 额外学时间上的运动/变化；
   - 代价：同样核尺寸，3D 参数量和计算量约为 2D 的 $k$ 倍；
   - 工程替代：视频常用 "2D 卷积 + 时间融合"（如每帧 2D 卷积 + RNN，
     或 (2+1)D 分解）来近似 3D 卷积、降低成本。

### 自测
- 一个 3D 卷积核 (3,3,3) 有多少参数？2D (3,3) 呢？（差 3 倍）
- 为什么视频不一定直接用 3D 卷积？（参数多、数据少、容易过拟合）

---

## 第 3 题：训练 loss 头几个 epoch 不下降的可能原因

### 考察什么
训练神经网络的**调试能力（debugging）**：能否系统地排查
"学习率、初始化、梯度流、数据、代码 bug"这条线索链。
这题没有唯一答案，答得越有条理越全面越好。

### 需要的背景知识（一条排查链）

1. **学习率**：太大 → 震荡/发散（loss 甚至会变 NaN）；
   太小 → 下降太慢，几个 epoch 内看不出变化。
   （我们的 Q6 就是例子：SGD 用 $10^{-4}$ 学习率前几个 epoch 几乎不动，Adam 用 $10^{-3}$ 飞快下降。）

2. **初始化**：全 0 初始化 → 对称性（symmetry）：同一层所有神经元收到相同梯度，
   永远一样，学不到东西；过大初始化 + sigmoid → 饱和，梯度≈0。

3. **梯度问题**：网络太深 + sigmoid/tanh → 梯度消失（第 1 题）。

4. **损失函数与输出层不匹配**：多分类却用 sigmoid 输出 + CE loss，或忘了 softmax；
   输出层激活函数与 loss 不配套 → 梯度很小或方向不对。

5. **数据**：没做归一化（特征尺度差太大，优化病态）；
   标签和输入对错位；类别标签从 1 开始而 CE 期望 0 开始。

6. **模型/代码 bug**：忘了 `optimizer.zero_grad()`、参数没接进优化器、
   `model.eval()`/`train()` 混用、梯度被 detach 等。
   （建议：先用一个小数据/玩具问题梯度检查 gradient check。）

### 自测
- 给你一个训练曲线：loss 震荡剧烈，先怀疑什么？loss 几乎水平线呢？
- "loss 不下降"和"loss 下降但准确率不涨"有什么区别？（后者可能是过拟合/数据问题）

---

## 第 4 题：类别不平衡（Imbalanced Datasets）

### 考察什么
数据分布对深度学习训练的影响，以及"偏斜数据"下的工程补救手段。
是实际业务（欺诈检测、医疗诊断、罕见事件预测）中最常见的问题。

### 需要的背景知识

1. **为什么会出问题**：
   - 优化目标（平均 loss / 准确率）被多数类主导；
   - 少数类样本的梯度在批量里占比小，边界被多数类"推走"；
   - 高准确率可能完全是假象（99:1 的数据全猜多数类就有 99% 准确率）。

2. **正确的评估方式**：不要只看 accuracy，要看 per-class precision / recall / F1、
   AUC、balanced accuracy、混淆矩阵（confusion matrix）。

3. **三类解决思路**（记忆口诀：改数据、改损失、改评估）：

   | 类别 | 方法 |
   |------|------|
   | 改数据 | 过采样（oversampling / SMOTE）、欠采样（undersampling）、对少数类做数据增强（augmentation）、平衡批次采样器（balanced sampler） |
   | 改损失/训练 | 类别加权损失（class-weighted CE）、Focal Loss（$-\alpha_t(1-p_t)^\gamma\log p_t$，让模型更关注难样本/少数类） |
   | 改评估/后处理 | 用 F1/AUC 选模型、调整决策阈值、少数类特别稀少时用异常检测思路 |

### 自测
- Focal Loss 里的 $\gamma$ 起什么作用？（越大，越抑制"容易样本"的梯度贡献）
- 过采样有什么风险？（重复样本导致过拟合少数类 → 可用 SMOTE 合成新样本缓解）

---

## 第 5 题：LeNet-5 + Fashion-MNIST 实验（35 分，重头戏）

### 考察什么
1. 经典 CNN 架构 LeNet-5 的结构（务必能默写：两层卷积+池化、三层全连接）；
2. 用 PyTorch 搭网络、训练、评估的完整流程；
3. **消融实验（ablation study）的思维**：一次只改一个变量，观察测试准确率变化；
4. 每个组件（层数、宽度、池化、核大小、激活、Dropout）的直觉理解。

### 需要的背景知识

1. **LeNet-5 原版结构**（LeCun et al., 1998，用于手写数字识别）：
   `Conv(6, 5×5) → Tanh → AvgPool(2) → Conv(16, 5×5) → Tanh → AvgPool(2) → FC(120) → FC(84) → FC(10)`。
   注意原版输入是 32×32，我们这里 Fashion-MNIST 是 28×28，所以第一层卷积要 padding=2 保持尺寸。

2. **为什么每个实验的直觉如下**（这是答题的关键，不能只贴数字）：
   - **a. 加/减卷积层**：卷积层越多，特征层次越深、非线性越强，但参数和过拟合风险也增加。
   - **b. 加/减全连接层**：全连接层参数巨大（占 LeNet 参数大头），多了容易过拟合。
   - **c. Dropout**：训练时随机丢弃神经元，防止过拟合。但 10 个 epoch 还没过拟合时，
     它反而拖慢收敛——所以本题里小幅度下降是正常现象，要能解释。
   - **d. 池化**：MaxPool 取最强响应（对纹理/边缘有用），AvgPool 平滑。
     Fashion-MNIST 是衣物图像，纹理和轮廓重要 → MaxPool 更好（我们的最佳单项改进 +2.53%）。
   - **e. 卷积核大小**：核越大感受野越大，但参数也越多。28×28 的小图上 3×3 太"近视"，
     7×7 略好。
   - **f. 输出通道数**：通道越多，能提取的特征种类越多，但计算量和过拟合风险上升。
   - **g. 激活函数**：ReLU 收敛快、不易饱和；sigmoid 深层堆叠 → 梯度消失（第 1 题），
     我们的实验里 sigmoid 版直接训不动（10% 准确率）。

3. **实验纪律（作业要求"至少两种设置"）**：
   - 固定种子（seed）、同样的 epoch / 学习率 / 批量 / 数据顺序，
     保证唯一变量是你要研究的那个组件；
   - 每个实验做 ≥2 个设置（如 Dropout 0.25 和 0.5）；
   - 报告**最佳结果和对应设置**（我们最终用 MaxPool+双通道+ReLU 的组合，84.48%）。

4. **读表技巧**：以 baseline 80.75% 为基准，每个实验报告"变化量"，
   最后挑出三项最有用的单因素改动组合成 best model。

### 自测
- LeNet-5 第二层卷积 (6→16, 5×5) 之后特征图为什么是 10×10？（28→pad2→28→pool→14→conv5→10）
- 为什么 Dropout 只在训练时开、测试时关？（测试时用完整网络，期望输出无随机性）
- 你能不看代码画出 LeNet-5 的数据流和每层尺寸吗？

---

## 第 6 题：NumPy 实现 Adam 并与 SGD 对比

### 考察什么
1. 优化器（optimizer）的数学原理——Adam 的三个核心机制；
2. 用 NumPy 手写优化器的能力（不依赖 torch.optim）；
3. 读图分析收敛速度的能力。

### 需要的背景知识

1. **回顾 A1 的问题**：100 维特征、2000 训练样本的二分类逻辑回归
   （sigmoid + 交叉熵），我们在 A1 里已经用 NumPy 实现了 GD/SGD 和 PyTorch 版本。
   Q6 就是在这套代码上加一个 Adam。

2. **Adam 的三个机制**（课本/课堂默认超参：lr=$10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$）：
   - **动量（momentum，一阶矩）**：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
     ——用历史梯度的指数滑动平均平滑当前梯度，减少震荡；
   - **自适应步长（二阶矩）**：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
     ——记录梯度平方的平均，用它给每个参数**单独**调步长（梯度大的方向步子小，反之步子大）；
   - **偏差校正（bias correction）**：$\hat{m}_t = m_t/(1-\beta_1^t)$，$\hat{v}_t = v_t/(1-\beta_2^t)$
     ——因为 $m_0=v_0=0$，训练初期估计偏小，除以 $(1-\beta^t)$ 修正。
     更新式：$\theta_t = \theta_{t-1} - \alpha\,\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$。

3. **对比协议**（作业指定）：SGD 学习率固定为 $10^{-4}$；Adam 用课堂默认超参。
   两者同样的数据、同样的 $W=0,b=0$ 初始化、同样的种子，只换优化器，比训练 loss 曲线。

4. **怎么读结果**：
   - Adam 在 epoch 3 时 loss=0.536，SGD 还在 0.614——Adam 明显快；
   - 原因有两层：a) Adam 默认 lr=$10^{-3}$ 本来就比 $10^{-4}$ 大 10 倍；
      b) 动量+自适应步长让每一步更"聪明"；
   - 100 epoch 后两者最终 loss 几乎一样（0.4219 vs 0.4224）——说明 Adam 赢在**速度**而非最终精度。

### 自测
- 为什么 $v_t$ 要加 $\epsilon$？（防止除以 0）
- 如果把 $\beta_1$ 设为 0，Adam 退化成什么？（RMSProp）
- bias correction 在 $t$ 很大时还重要吗？（不重要，$\beta^t \to 0$）

---

## 第 7 题：论文总结（4 篇 NLP/生物信息经典）

格式固定：Motivation（动机）/ Novelty（创新点）/ Intuition（直觉）/
Implementation（代码链接）。

### 需要的背景知识（每篇一句话定位）

1. **Word2Vec（Mikolov et al., 2013）**：最早的"词向量"学习法。
   - 背景：用神经网络语言模型学词向量太慢；
   - 核心：去掉隐层、用 CBOW/Skip-gram 两个简单结构 + 层次 softmax/负采样提速；
   - 直觉：**分布假设**——"词的意义由它的上下文决定"，上下文相似的词向量自然相似；
     副产品是向量可以做类比运算（King - Man + Woman ≈ Queen）。
   - 记住关键词：CBOW, Skip-gram, hierarchical softmax, negative sampling, word analogy。

2. **Transformer（Vaswani et al., 2017）**："Attention Is All You Need"。
   - 背景：RNN/LSTM 只能一步步算，无法并行、长序列信息难传递；
   - 核心：完全抛弃循环，只用**自注意力（multi-head scaled dot-product self-attention）**
     + 前馈网络 + 残差 + LayerNorm；
   - 直觉：每个词直接"看"所有其他词并动态加权，任意两个位置的信息路径都是 O(1)，
     所以既快又能捕捉长距离依赖。
   - 记住关键词：self-attention, multi-head, positional encoding, encoder-decoder, Q/K/V。

3. **BERT（Devlin et al., 2019）**：预训练 + 微调范式的代表作。
   - 背景：以前的语言模型是单向的（只能看左边或右边）；
   - 核心：**Masked Language Model（掩码语言模型）** + Next Sentence Prediction
     两个自监督任务，在 Transformer 编码器上预训练，双向建模；
   - 直觉：看到完整上下文（双向）能学到更丰富的语义表示，
     预训练学"通用语言知识"，微调时只需加一个小输出层就能迁移到 11 个下游任务。
   - 记住关键词：pre-training, fine-tuning, MLM, NSP, bidirectional, GLUE/SQuAD。

4. **MSA Transformer（Rao et al., 2021）**：把 Transformer 用到蛋白质多序列比对。
   - 背景：从蛋白质多序列比对（MSA）里提取共进化信息传统上靠昂贵迭代算法；
   - 核心：在 MSA 矩阵上做**行注意力（沿序列）+ 列注意力（沿位点）**，
     用掩码语言建模在 2600 万个 MSA 上预训练；
   - 直觉：列是"对齐的氨基酸位点"、行是"同源序列"；
     列注意力能学到哪些位点共变（物理上接触的残基会共同进化），
     行注意力能学到哪些序列信息量大——模型无监督地学到了蛋白质结构信息。
   - 记住关键词：protein MSA, row/column attention, contact prediction, ESM。

### 自测
- 四篇论文之间的传承关系是什么？
  （word2vec 把词变成向量 → Transformer 给序列建模提供新骨架 → BERT 把 Transformer 做成预训练语言模型 → MSA Transformer 把同样的思想迁移到蛋白质数据。）
- 为什么 BERT 不能用普通自回归语言模型的方式做双向？（因为"看到自己"就泄漏答案 → 所以要 MLM 掩码。）

---

## 总结：本作业考察的能力地图

| 题号 | 考察能力 |
|------|---------|
| Q1 | 反向传播 + 训练稳定性的理论理解 |
| Q2 | CNN 基础概念（卷积、维度、代价） |
| Q3 | 训练调试（troubleshooting）的系统思维 |
| Q4 | 数据分布问题 + 工程补救手段 |
| Q5 | PyTorch 实战 + 消融实验设计 + 结果分析 |
| Q6 | 优化器数学原理 + NumPy 手写实现 |
| Q7 | 论文阅读与结构化总结（动机/创新/直觉） |

Q5+Q6 是动手题，建议自己再跑一遍代码（`A2_Q5_lenet.py`、`A2_Q6_adam.py`），
改几个参数观察变化，这样对概念的理解会从"记住"变成"会用"。
