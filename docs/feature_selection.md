# 特征选择详解

本文档详细阐述了 `src/feature_selection/feature_selector.py` 脚本中实现的高级特征选择系统。该系统针对微藻脂质含量预测任务，结合多种特征选择方法，提供全面的特征筛选和分析。整个系统专门针对小样本高维数据进行了优化，采用集成策略确保特征选择的稳健性。

---

## 系统总览

特征选择系统在 `AdvancedFeatureSelector` 类中实现，采用多方法集成的策略。核心流程如下：

1. **数据加载与质量验证**：加载预处理后的数据并进行质量检查
2. **特征工程**：创建衍生特征以增强数据表达能力
3. **基础过滤**：移除低方差特征和高相关性特征
4. **多方法特征选择**：应用6种不同的特征选择方法
5. **集成选择**：通过投票机制综合多种方法的结果
6. **性能评估**：评估不同特征集的预测性能
7. **结果可视化与保存**：生成图表并保存选择结果

---

## 1. 数据加载与验证

**目标**：加载预处理后的数据并确保数据质量满足特征选择要求。

**实现函数**：`load_data()` 和 `validate_data_quality()`

```python
def load_data(self):
    """加载预处理后的数据"""
    processed_data = pd.read_csv("../../data/processed/processed_data.csv")
    self.y = processed_data['lipid(%)']
    self.X = processed_data.drop('lipid(%)', axis=1)
    # 创建衍生特征
    self.X = self.create_derived_features(self.X)
    return self.X, self.y
```

**详细说明**：
- 从预处理模块的输出文件中加载标准化后的数据
- 自动分离特征矩阵X和目标变量y
- 调用特征工程函数创建衍生特征，增强数据的表达能力
- 进行数据质量验证，处理无穷大值和NaN值
- 检查目标变量的分布特性和特征方差

**质量验证包括**：
- **无穷大值处理**：将inf值替换为极大值(±1e6)
- **NaN值处理**：使用中位数填充缺失值
- **低方差特征检测**：识别方差小于1e-6的特征
- **目标变量分析**：计算变异系数评估预测难度

---

## 2. 特征工程

**目标**：基于原始特征创建有意义的衍生特征，提升模型表达能力。

**实现函数**：`create_derived_features(X)`

**衍生特征类型**：

1. **元素比例特征**：
   - `C_N_ratio`: 碳氮比，反映营养平衡
   - `N_P_ratio`: 氮磷比，影响藻类生长
   - `C_P_ratio`: 碳磷比，营养限制指标

2. **环境指标特征**：
   - `DO_COD_ratio`: 溶解氧与化学需氧量比值
   - `nitrate_ammonia_ratio`: 硝态氮与氨氮比值
   - `TN_TP_ratio`: 总氮总磷比

3. **生物特征**：
   - `pigment_per_cell`: 单细胞色素含量
   - `cell_weight`: 细胞重量指标
   - `CHON_balance`: 碳氢氧氮平衡指数

**详细说明**：
这些衍生特征基于微藻生物学原理设计，能够捕获原始特征间的非线性关系，为后续的特征选择提供更丰富的候选特征集。

---

## 3. 基础过滤

**目标**：移除信息量低和冗余的特征，减少特征空间维度。

### 3.1 低方差特征过滤

**实现函数**：`remove_low_variance_features(threshold=0.001)`

```python
def remove_low_variance_features(self, threshold=0.001):
    """移除低方差特征"""
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(self.X)
    selected_features = self.X.columns[selector.get_support()]
    return selected_features
```

**详细说明**：
- 使用极小的阈值(0.001)以适应小样本数据
- 移除方差接近零的特征，这些特征对预测贡献很小
- 保留具有一定变异性的特征

### 3.2 相关性分析

**实现函数**：`correlation_analysis(threshold=0.90)`

**详细说明**：
- 计算特征间的Pearson相关系数
- 识别高度相关的特征对(|r| > 0.90)
- 在相关特征对中保留与目标变量相关性更强的特征
- 生成相关性热力图用于可视化分析

---

## 4. 多方法特征选择

系统实现了6种不同的特征选择方法，每种方法都有其独特的优势：

### 4.1 单变量选择

**实现函数**：`univariate_selection(k=15)`

**方法包括**：
1. **F-regression**：基于F统计量的特征选择
2. **Mutual Information**：基于互信息的特征选择

```python
def univariate_selection(self, k=15):
    """单变量特征选择 - 基于统计显著性"""
    # 计算相关性和p值
    correlations = {}
    p_values = {}
    for col in self.X.columns:
        corr, p_val = pearsonr(self.X[col], self.y)
        correlations[col] = abs(corr)
        p_values[col] = p_val
    
    # 选择统计显著的特征 (p < 0.1)
    significant_features = [col for col, p in p_values.items() if p < 0.1]
```

**详细说明**：
- 首先进行统计显著性检验，只考虑p<0.1的特征
- F-regression适用于线性关系的检测
- 互信息能够捕获非线性关系
- 针对小样本调整了选择策略，更加保守

### 4.2 基于树的选择

**实现函数**：`tree_based_selection(n_features=15)`

**方法包括**：
1. **Random Forest**：随机森林特征重要性
2. **Extra Trees**：极端随机树特征重要性

```python
def tree_based_selection(self, n_features=15):
    """基于树模型的特征选择"""
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
    rf.fit(self.X, self.y)
    rf_importances = rf.feature_importances_
    
    # Extra Trees  
    et = ExtraTreesRegressor(n_estimators=100, random_state=self.random_state)
    et.fit(self.X, self.y)
    et_importances = et.feature_importances_
```

**详细说明**：
- 树模型能够自动处理特征交互和非线性关系
- Random Forest通过bootstrap采样增加稳定性
- Extra Trees引入更多随机性，减少过拟合风险
- 基于Gini不纯度计算特征重要性

### 4.3 正则化选择

**实现函数**：`lasso_selection(alpha_range=None)`

```python
def lasso_selection(self, alpha_range=None):
    """基于Lasso回归的特征选择"""
    if alpha_range is None:
        alpha_range = np.logspace(-6, -1, 50)  # 针对小样本调整
    
    # 针对小样本调整CV折数
    n_samples = len(self.y)
    cv_folds = min(5, max(3, n_samples // 5))
    
    lasso_cv = LassoCV(alphas=alpha_range, cv=cv_folds, random_state=self.random_state)
    lasso_cv.fit(X_scaled, self.y)
```

**详细说明**：
- 使用L1正则化自动进行特征选择
- 针对小样本调整了alpha范围和交叉验证折数
- 通过交叉验证自动选择最优正则化参数
- 系数为零的特征被自动剔除

### 4.4 递归特征消除

**实现函数**：`rfe_selection(n_features=15)` 和 `rfecv_selection(cv=5)`

**方法包括**：
1. **RFE**：固定特征数的递归消除
2. **RFECV**：交叉验证确定最优特征数

```python
def rfe_selection(self, n_features=15):
    """递归特征消除"""
    estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(self.X, self.y)
    selected_features = self.X.columns[rfe.support_]
```

**详细说明**：
- 递归地移除最不重要的特征
- 使用Random Forest作为基础估计器
- RFE指定固定的特征数量
- RFECV通过交叉验证自动确定最优特征数

---

## 5. 集成选择策略

**目标**：综合多种方法的结果，提高特征选择的稳健性。

**实现函数**：`ensemble_selection(methods=None, min_votes=3)`

```python
def ensemble_selection(self, methods=None, min_votes=3):
    """集成多种方法的特征选择结果"""
    if methods is None:
        methods = ['f_regression', 'mutual_info', 'random_forest', 
                  'extra_trees', 'lasso', 'rfe']
    
    # 统计每个特征被选择的次数
    feature_votes = {}
    for feature in self.X.columns:
        feature_votes[feature] = 0
    
    # 计算投票
    for method in methods:
        if method in self.results:
            selected = self.results[method]['selected_features']
            for feature in selected:
                feature_votes[feature] += 1
    
    # 选择得票数 >= min_votes 的特征
    ensemble_features = [f for f, votes in feature_votes.items() if votes >= min_votes]
```

**详细说明**：
- 采用投票机制，统计每个特征被不同方法选中的次数
- 只保留得票数达到最小阈值的特征
- 针对小样本数据降低了投票阈值(min_votes=2)
- 按得票数对特征进行排序，便于分析

---

## 6. 性能评估

**目标**：评估不同特征集在预测任务上的性能表现。

**实现函数**：`evaluate_feature_sets(feature_sets=None)`

**评估策略**：
- **小样本处理**：样本数≤50时使用Leave-One-Out交叉验证
- **模型选择**：使用Ridge回归作为评估模型，适合小样本
- **异常处理**：当Ridge回归产生NaN时，自动切换到线性回归
- **备用策略**：交叉验证失败时使用简单的训练测试分割

**评估指标**：
- **R²得分**：解释方差比例
- **交叉验证均值和标准差**：评估稳定性
- **特征数量**：评估模型复杂度

---

## 7. 结果分析

### 7.1 最终选择结果

根据集成选择策略，系统最终选择了以下特征：

| 特征名称 | 描述 | 重要性排名 |
|---------|------|-----------|
| protein(%) | 蛋白质含量百分比 | 1 |
| H(%) | 氢元素含量百分比 | 2 |
| O(%) | 氧元素含量百分比 | 3 |
| pigment_per_cell | 单细胞色素含量 | 4 |

### 7.2 方法对比分析

| 方法 | 选择特征数 | 主要特点 |
|------|-----------|----------|
| F-regression | 4 | 基于线性相关性，统计显著 |
| Mutual Info | 4 | 捕获非线性关系 |
| Random Forest | 4 | 基于树模型重要性 |
| Extra Trees | 4 | 更强的随机性，减少过拟合 |
| Lasso | 24 | L1正则化，自动特征选择 |
| RFE | 4 | 递归消除，逐步优化 |
| RFECV | 23 | 交叉验证确定最优数量 |
| **集成结果** | **4** | **多方法投票，稳健性强** |

### 7.3 特征选择效果

- **维度降低**：从原始的36个特征降至4个核心特征
- **生物学意义**：选择的特征都具有明确的生物学解释
- **稳健性**：通过多方法集成确保选择的稳定性
- **适应性**：针对小样本数据进行了专门优化

---

## 8. 可视化输出

系统生成以下可视化图表：

1. **特征重要性对比图**：展示不同方法的特征重要性排序
2. **方法对比图**：比较各种特征选择方法的性能
3. **特征投票图**：显示每个特征的得票情况
4. **性能评估图**：不同特征集的交叉验证性能对比

所有图表保存在 `results/feature_selection/` 目录下。

---

## 9. 输出文件

系统保存以下结果文件：

- `selection_summary.csv`：各方法选择结果汇总
- `best_features_list.csv`：最终选择的特征列表
- `best_features_data.csv`：最终特征的数据
- `ensemble_features_list.csv`：集成方法选择的特征
- `performance_comparison.png`：性能对比图表

---

## 总结

本特征选择系统通过集成多种方法，成功地从高维特征空间中识别出了最具预测价值的特征子集。系统特别针对微藻脂质含量预测的小样本特点进行了优化，在保证选择稳健性的同时，确保了所选特征的生物学合理性。最终选择的4个特征为后续的模型训练提供了高质量的输入。
