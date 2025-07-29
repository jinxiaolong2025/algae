# 数据预处理详解

文档详细阐述了 `src/data_processing/data_preprocessing.py` 脚本中实现的数据预处理流程。该流程旨在为后续的机器学习模型训练准备高质量、干净的数据。整个管道遵循了标准的机器学习最佳实践，包括数据质量分析、缺失值处理、异常值处理、数据标准化和防止数据泄露等关键步骤。

---

## 流程总览

数据预处理的完整流程在脚本的 `if __name__ == "__main__":` 主执行块中被精心编排。其核心步骤如下：

1.  **加载原始数据**：从Excel文件中读取原始数据。
2.  **数据质量分析**：对数据进行全面的健康检查，包括缺失值、零值、偏度和数值范围。
3.  **处理缺失值**：采用针对性的策略（线性回归和KNN插补）填充缺失数据。
4.  **处理异常值**：基于数据偏度，采用差异化的策略（分位数限幅）来处理异常值。
5.  **分割数据集**：在标准化之前将数据分为训练集和测试集，以防止数据泄露。
6.  **Robust标准化**：对数据进行稳健的标准化处理，使其对异常值不敏感。
7.  **保存处理后的数据**：将处理完成的训练集、测试集和完整数据集保存为CSV文件。

下面，我们将按顺序详细拆解每一步的实现细节。

---

## 1. 加载原始数据

**目标**：从外部文件加载数据到Pandas DataFrame中，并进行初步的清洗。

**实现函数**： `load_data()`

```python
def load_data():
    data = pd.read_excel("../../data/raw/数据.xlsx")
    # 删除S(%)列
    data = data.drop('S(%)', axis=1)
    return data
```

**详细说明**：
*   该函数使用 `pandas.read_excel` 读取位于 `data/raw/` 目录下的原始数据文件 `数据.xlsx`。
*   在加载后，立即删除了 `S(%)`（硫含量）列。这通常是基于领域知识或初步分析，认为该特征与目标变量（如脂质含量）无关或数据质量过差，因此提前剔除。

---

## 2. 数据质量分析

**目标**：系统地评估数据集的质量，为后续的数据清洗和预处理提供决策依据。

**实现函数**： `analyze_data_quality(data)`

```python
def analyze_data_quality(data):
    """分析数据质量：缺失率、零值占比、偏度"""
    # ... (代码实现细节)
    quality_analysis.to_csv("../../results/data_preprocess/data_quality_analysis.csv", float_format='%.3f')
    return quality_analysis
```

**详细说明**：
这是一个非常全面的数据审查函数，它从多个维度进行分析：
*   **缺失值分析**：计算每个特征的缺失值数量和比例，识别存在缺失的特征。
*   **零值分析**：计算数值型特征中零值的数量和比例。高比例的零值可能表示数据缺失或特征本身特性，需要关注。
*   **偏度分析**：计算数值型特征的偏度（skewness），以衡量其分布的对称性。偏度是后续异常值处理策略的关键依据。
*   **数值范围分析**：展示每个数值特征的最小值、最大值、均值和标准差，帮助理解数据的尺度和分布。

**输出**：
*   在控制台打印详细的分析报告。
*   将完整的质量分析结果保存到 `results/data_preprocess/data_quality_analysis.csv` 文件中，方便后续查阅。

---

## 3. 处理缺失值

**目标**：根据特征的相关性和数据特性，采用最合适的策略填充缺失值。

**实现函数**： `handle_missing_values(data)`

```python
def handle_missing_values(data):
    """处理缺失值 - 针对性处理策略"""
    # ... (代码实现细节)
    return data_filled
```

**详细说明**：
此函数没有采用单一的填充方法，而是实现了“具体问题具体分析”的策略：

1.  **`phosphate` (磷酸盐) 缺失值处理**：
    *   首先，计算 `phosphate` 和 `TP` (总磷) 之间的相关性。
    *   如果存在强相关性，则利用 `TP` 作为自变量，`phosphate` 作为因变量，训练一个简单的**线性回归模型**。
    *   使用这个训练好的模型来预测并填充 `phosphate` 的缺失值。这种方法比简单的均值或中位数填充更精确，因为它利用了特征间的内在关系。

2.  **`N(%)` 和 `C(%)` 缺失值处理**：
    *   对于剩余的缺失值（主要是氮和碳的含量），采用 **K-最近邻（KNN）插补** (`KNNImputer`)。
    *   KNN会寻找与缺失样本最相似的K个样本，并用这K个样本的加权平均值来填充缺失值。它同样利用了多维度特征间的关系，是一种比单变量填充更稳健的方法。

---

## 4. 处理异常值

**目标**：识别并处理数据中的异常值，以增强模型的稳定性和性能。

**实现函数**： `handle_outliers(data)`

```python
def handle_outliers(data):
    """基于偏度的异常值处理"""
    # ... (代码实现细节)
    return data_processed
```

**详细说明**：
该项目采用了一种非常智能的、基于数据分布的异常值处理策略，而不是“一刀切”：

*   **诊断**：首先计算每个数值特征的**偏度**。
*   **分类处理**：
    *   **轻度偏态 (`|偏度| < 1`)**：数据分布接近对称，不进行异常值处理，以保留原始信息。
    *   **中度偏态 (`1 ≤ |偏度| < 2`)**：采用 **5%-95%分位数限幅（Winsorize）**。即将低于5%分位数的值替换为5%分位数，高于95%分位数的值替换为95%分位数。
    *   **重度偏态 (`|偏度| ≥ 2`)**：采用更强的 **10%-90%分位数限幅**，对极端值进行更大力度的平滑处理。

**闭环验证**：
处理完成后，脚本通过调用 `visualize_outlier_treatment` 和 `visualize_skewness_improvement` 函数，生成处理前后的对比图，包括：
*   **箱线图和直方图对比**：直观展示异常值被处理后的数据分布变化。
*   **偏度改善分析图**：量化展示每个特征偏度的改善情况。

这些可视化结果保存在 `results/data_preprocess/` 目录下，形成了一个“处理-验证”的闭环，确保了处理的有效性。

---

## 5. 分割数据集

**目标**：将数据集分为训练集和测试集，为后续的模型训练和评估做准备。

**实现函数**： `split_dataset(data, test_size=0.2, random_state=42)`

```python
def split_dataset(data, test_size=0.2, random_state=42):
    """分割数据集为训练集和测试集"""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data
```

**详细说明**：
*   使用 `sklearn.model_selection.train_test_split` 函数进行分割。
*   `test_size=0.2` 表示将80%的数据用作训练，20%用作测试。
*   `random_state=42` 确保了每次分割的结果都是一致的，这对于实验的可复现性至关重要。

**关键点**：这一步在**数据标准化之前**执行，这是为了**防止数据泄露**。如果先标准化再分割，测试集的信息（如均值、方差）会“泄露”到训练集中，导致模型评估结果过于乐观，是不正确的做法。

---

## 6. Robust标准化

**目标**：对特征进行缩放，使其具有相似的尺度，同时对异常值不敏感。

**实现函数**： `robust_scaling(data, scaler=None, fit=True)`

```python
def robust_scaling(data, scaler=None, fit=True):
    """Robust标准化处理"""
    # ... (代码实现细节)
    return data_scaled, scaler
```

**详细说明**：
*   该函数使用 `sklearn.preprocessing.RobustScaler`。
*   与 `StandardScaler`（使用均值和标准差）不同，`RobustScaler` 使用**中位数（median）**和**四分位距（IQR）**进行缩放。由于中位数和IQR对异常值不敏感，因此 `RobustScaler` 对于包含异常值的数据集是更稳健的选择。
*   **目标变量 `lipid(%)`** 在标准化过程中被排除，因为目标变量通常不应该进行标准化。
*   **防止数据泄露的实现**：
    *   在训练集上调用时，`fit=True`，`scaler` 会学习**训练集**的中位数和IQR，并转换训练集。
    *   在测试集上调用时，`fit=False`，`scaler` 会使用**之前从训练集学到的参数**来转换测试集，保证了处理的一致性。

---

## 7. 保存处理后的数据

**目标**：将最终处理好的数据持久化，供后续的模型训练和测试模块使用。

**执行逻辑**：

```python
# ...
train_data_scaled.to_csv("../../data/processed/train_data.csv", index=False, float_format='%.6f')
test_data_scaled.to_csv("../../data/processed/test_data.csv", index=False, float_format='%.6f')
processed_data_scaled.to_csv("../../data/processed/processed_data.csv", index=False, float_format='%.6f')
```

**详细说明**：
脚本最终会将三个关键的DataFrame保存为CSV文件：
*   `train_data.csv`：经过完整预处理的训练数据。
*   `test_data.csv`：经过完整预处理的测试数据。
*   `processed_data.csv`：合并后的完整预处理数据集，可用于交叉验证或最终模型训练。

所有文件都保存在 `data/processed/` 目录下，结构清晰，便于后续模块调用。
