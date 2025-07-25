#  API 文档

##  目录

1. [核心类](#核心类)
2. [数据处理模块](#数据处理模块)
3. [特征选择模块](#特征选择模块)
4. [数据增强模块](#数据增强模块)
5. [模型训练模块](#模型训练模块)
6. [评估模块](#评估模块)
7. [工具函数](#工具函数)
8. [配置管理](#配置管理)

---

##  核心类

### UltimateEnsemblePipeline

终极集成学习管道的主类，整合了所有功能模块。

```python
class UltimateEnsemblePipeline:
    """
    集成学习管道
    
    主要功能：
    - 数据预处理
    - 特征选择
    - 数据增强
    - 模型训练
    - 性能评估
    """
```

#### 构造函数

```python
def __init__(self, target_features=3, random_state=42):
    """
    初始化管道
    
    参数:
        target_features (int): 目标特征数量，默认3
        random_state (int): 随机种子，默认42
    
    示例:
        >>> pipeline = UltimateEnsemblePipeline(target_features=5)
        >>> pipeline = UltimateEnsemblePipeline(target_features=3, random_state=123)
    """
```

#### 主要方法

##### run_ultimate_pipeline

```python
def run_ultimate_pipeline(self, file_path):
    """
    运行完整的机器学习管道
    
    参数:
        file_path (str): 数据文件路径
    
    返回:
        tuple: (all_results, best_combination)
            - all_results (dict): 所有模型组合的详细结果
            - best_combination (dict): 最佳模型组合信息
    
    示例:
        >>> pipeline = UltimateEnsemblePipeline()
        >>> results, best = pipeline.run_ultimate_pipeline('data/raw/数据.xlsx')
        >>> print(f"最佳模型: {best['model']}")
        >>> print(f"最佳数据集: {best['dataset']}")
        >>> print(f"训练集R²: {best['train_r2']:.4f}")
    
    异常:
        FileNotFoundError: 数据文件不存在
        ValueError: 数据格式错误
        RuntimeError: 模型训练失败
    """
```

---

##  数据处理模块

### load_and_preprocess_data

```python
def load_and_preprocess_data(self, file_path):
    """
    加载和预处理数据
    
    功能:
    1. 读取Excel文件
    2. 智能识别目标变量
    3. 数值化处理
    4. 缺失值处理
    5. 常数特征移除
    
    参数:
        file_path (str): Excel文件路径
    
    返回:
        tuple: (X, y)
            - X (pd.DataFrame): 特征矩阵
            - y (pd.Series): 目标向量
    
    示例:
        >>> X, y = pipeline.load_and_preprocess_data('data.xlsx')
        >>> print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
        >>> print(f"目标变量范围: {y.min():.2f} - {y.max():.2f}")
    
    注意:
        - 支持的目标变量关键词: ['目标', 'target', 'y', '标签', 'label']
        - 自动处理数值型和分类型特征
        - 移除缺失值比例>50%的特征
    """
```

### identify_target_column

```python
def identify_target_column(self, df, keywords):
    """
    智能识别目标变量列
    
    参数:
        df (pd.DataFrame): 数据框
        keywords (list): 目标变量关键词列表
    
    返回:
        str: 目标变量列名
    
    示例:
        >>> target_col = pipeline.identify_target_column(df, ['目标', 'target'])
        >>> print(f"识别的目标列: {target_col}")
    
    异常:
        ValueError: 无法识别目标变量
    """
```

### convert_to_numeric

```python
def convert_to_numeric(self, data):
    """
    智能数值化转换
    
    功能:
    - 自动识别数值型数据
    - 处理字符串型数值
    - 保留原始数据类型信息
    
    参数:
        data (pd.DataFrame or pd.Series): 待转换数据
    
    返回:
        pd.DataFrame or pd.Series: 数值化后的数据
    
    示例:
        >>> numeric_data = pipeline.convert_to_numeric(raw_data)
        >>> print(f"数值化后数据类型: {numeric_data.dtypes}")
    """
```

---

##  特征选择模块

### advanced_feature_selection

```python
def advanced_feature_selection(self, X, y):
    """
    高级特征选择算法
    
    策略:
    1. F统计量特征选择
    2. 互信息特征选择
    3. 方差排序补充
    
    参数:
        X (pd.DataFrame): 特征矩阵
        y (pd.Series): 目标向量
    
    返回:
        pd.DataFrame: 选择后的特征矩阵
    
    示例:
        >>> X_selected = pipeline.advanced_feature_selection(X, y)
        >>> print(f"选择的特征: {list(X_selected.columns)}")
        >>> print(f"特征数量: {len(X_selected.columns)}")
    
    算法原理:
        F统计量: 衡量特征与目标的线性关系
        互信息: 衡量特征与目标的非线性关系
        方差排序: 补充高方差特征
    """
```

### feature_importance_analysis

```python
def feature_importance_analysis(self, X, y, method='all'):
    """
    特征重要性分析
    
    参数:
        X (pd.DataFrame): 特征矩阵
        y (pd.Series): 目标向量
        method (str): 分析方法 ['f_test', 'mutual_info', 'variance', 'all']
    
    返回:
        dict: 特征重要性结果
    
    示例:
        >>> importance = pipeline.feature_importance_analysis(X, y)
        >>> for feature, score in importance['f_test'].items():
        ...     print(f"{feature}: {score:.4f}")
    """
```

---

##  数据增强模块

### SMOGNRegressor

专门用于回归问题的SMOGN数据增强器。

```python
class SMOGNRegressor:
    """
    SMOGN回归数据增强器
    
    原理:
    - 基于K近邻的合成样本生成
    - 特征空间线性插值
    - 目标空间噪声添加
    """
```

#### 构造函数

```python
def __init__(self, k_neighbors=3, noise_factor=0.01, random_state=42):
    """
    初始化SMOGN增强器
    
    参数:
        k_neighbors (int): K近邻数量，默认3
        noise_factor (float): 噪声强度，默认0.01
        random_state (int): 随机种子，默认42
    
    示例:
        >>> smogn = SMOGNRegressor(k_neighbors=5, noise_factor=0.02)
    """
```

#### 主要方法

```python
def fit_resample(self, X, y, target_samples=None):
    """
    生成合成样本
    
    参数:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标向量
        target_samples (int): 目标样本数，默认为原样本数的2倍
    
    返回:
        tuple: (X_resampled, y_resampled)
            - X_resampled (np.ndarray): 增强后的特征矩阵
            - y_resampled (np.ndarray): 增强后的目标向量
    
    示例:
        >>> smogn = SMOGNRegressor()
        >>> X_aug, y_aug = smogn.fit_resample(X, y, target_samples=100)
        >>> print(f"原始样本数: {len(X)}, 增强后: {len(X_aug)}")
    
    算法步骤:
        1. 为每个样本寻找K个最近邻
        2. 随机选择一个邻居进行插值
        3. 在特征空间进行线性插值
        4. 在目标空间添加高斯噪声
    """
```

### noise_augmentation

```python
def noise_augmentation(self, X, y, noise_factor=0.05, target_samples=None):
    """
    噪声增强算法
    
    原理:
    - 向原始数据添加小幅高斯噪声
    - 保持数据分布的本质特征
    - 增加数据的多样性
    
    参数:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标向量
        noise_factor (float): 噪声强度，默认0.05
        target_samples (int): 目标样本数
    
    返回:
        tuple: (X_noisy, y_noisy)
    
    示例:
        >>> X_noisy, y_noisy = pipeline.noise_augmentation(X, y, noise_factor=0.03)
        >>> print(f"噪声强度: {noise_factor}")
        >>> print(f"增强后样本数: {len(X_noisy)}")
    
    数学原理:
        X_noisy = X + ε_X, ε_X ~ N(0, noise_factor * σ_X)
        y_noisy = y + ε_y, ε_y ~ N(0, noise_factor * σ_y)
    """
```

---

##  模型训练模块

### BayesianEnsemble

贝叶斯集成学习器，智能融合多个基础模型。

```python
class BayesianEnsemble(BaseEstimator, RegressorMixin):
    """
    贝叶斯集成回归器
    
    特点:
    - 自动学习模型权重
    - 基于贝叶斯推理
    - 动态权重调整
    """
```

#### 构造函数

```python
def __init__(self, base_models=None, random_state=42):
    """
    初始化贝叶斯集成器
    
    参数:
        base_models (list): 基础模型列表
        random_state (int): 随机种子
    
    示例:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.linear_model import Ridge
        >>> base_models = [RandomForestRegressor(), Ridge()]
        >>> ensemble = BayesianEnsemble(base_models=base_models)
    """
```

#### 主要方法

```python
def fit(self, X, y):
    """
    训练贝叶斯集成模型
    
    参数:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标向量
    
    返回:
        self: 训练后的模型实例
    
    示例:
        >>> ensemble.fit(X_train, y_train)
        >>> print(f"模型权重: {ensemble.weights_}")
    
    算法原理:
        1. 训练每个基础模型
        2. 计算每个模型的似然函数
        3. 使用贝叶斯公式计算权重
        4. 归一化权重向量
    """

def predict(self, X):
    """
    贝叶斯加权预测
    
    参数:
        X (np.ndarray): 特征矩阵
    
    返回:
        np.ndarray: 预测结果
    
    示例:
        >>> y_pred = ensemble.predict(X_test)
        >>> print(f"预测结果: {y_pred[:5]}")
    
    数学原理:
        ŷ = Σ(w_i * ŷ_i), 其中w_i为第i个模型的贝叶斯权重
    """
```

### create_advanced_models

```python
def create_advanced_models(self):
    """
    创建高级模型集合
    
    返回:
        dict: 模型字典，键为模型名称，值为模型实例
    
    包含模型:
        - BayesianEnsemble: 贝叶斯集成
        - RandomForest_Aggressive: 激进随机森林
        - GradientBoosting_Tuned: 调优梯度提升
        - ElasticNet_Optimized: 优化弹性网络
        - SVR_Polynomial: 多项式支持向量机
        - MLP_Deep: 深度多层感知机
    
    示例:
        >>> models = pipeline.create_advanced_models()
        >>> for name, model in models.items():
        ...     print(f"模型: {name}, 类型: {type(model).__name__}")
    """
```

---

##  评估模块

### evaluate_model

```python
def evaluate_model(self, model, X_train, X_test, y_train, y_test):
    """
    全面评估单个模型
    
    参数:
        model: 待评估的模型
        X_train (np.ndarray): 训练特征
        X_test (np.ndarray): 测试特征
        y_train (np.ndarray): 训练目标
        y_test (np.ndarray): 测试目标
    
    返回:
        dict: 评估结果字典
    
    示例:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor()
        >>> results = pipeline.evaluate_model(model, X_train, X_test, y_train, y_test)
        >>> print(f"训练R²: {results['train_r2']:.4f}")
        >>> print(f"测试R²: {results['test_r2']:.4f}")
        >>> print(f"交叉验证R²: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
    
    评估指标:
        - train_r2: 训练集R²
        - test_r2: 测试集R²
        - cv_r2_mean: 交叉验证R²均值
        - cv_r2_std: 交叉验证R²标准差
        - train_mse: 训练集均方误差
        - test_mse: 测试集均方误差
    """
```

### dynamic_cross_validation

```python
def dynamic_cross_validation(self, X, y, model):
    """
    动态交叉验证
    
    功能:
    - 根据样本量自适应调整CV折数
    - 小样本时使用留一法
    - 确保每折有足够样本
    
    参数:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标向量
        model: 待验证的模型
    
    返回:
        tuple: (cv_mean, cv_std)
            - cv_mean (float): 交叉验证均值
            - cv_std (float): 交叉验证标准差
    
    示例:
        >>> cv_mean, cv_std = pipeline.dynamic_cross_validation(X, y, model)
        >>> print(f"CV结果: {cv_mean:.4f} ± {cv_std:.4f}")
    
    算法逻辑:
        if 样本数 >= 15: 使用5折交叉验证
        elif 样本数 >= 10: 使用3折交叉验证
        else: 使用留一法交叉验证
    """
```

### comprehensive_scoring

```python
def comprehensive_scoring(self, result):
    """
    综合评分机制（防过拟合）
    
    参数:
        result (dict): 模型评估结果
    
    返回:
        float: 综合评分
    
    示例:
        >>> score = pipeline.comprehensive_scoring(result)
        >>> print(f"综合评分: {score:.4f}")
    
    评分公式:
        score = train_r2 + cv_r2 - overfitting_penalty
        overfitting_penalty = max(0, train_r2 - test_r2 - 0.1)
    
    设计理念:
        - 奖励高训练性能
        - 奖励高交叉验证性能
        - 惩罚过拟合行为
    """
```

---

##  工具函数

### robust_scaling

```python
def robust_scaling(self, X):
    """
    鲁棒标准化
    
    特点:
    - 使用中位数和四分位距
    - 对异常值更鲁棒
    - 保持数据分布形状
    
    参数:
        X (np.ndarray): 待标准化的数据
    
    返回:
        tuple: (X_scaled, scaler)
            - X_scaled (np.ndarray): 标准化后的数据
            - scaler: 标准化器对象
    
    示例:
        >>> X_scaled, scaler = pipeline.robust_scaling(X)
        >>> print(f"标准化后范围: {X_scaled.min():.2f} - {X_scaled.max():.2f}")
    
    数学原理:
        X_scaled = (X - median(X)) / IQR(X)
        其中 IQR = Q3 - Q1（四分位距）
    """
```

### save_results

```python
def save_results(self, results, best_combination, file_path):
    """
    保存详细结果到文件
    
    参数:
        results (dict): 所有结果
        best_combination (dict): 最佳组合
        file_path (str): 保存路径
    
    示例:
        >>> pipeline.save_results(results, best, 'results/analysis.txt')
    
    保存内容:
        - 数据基本信息
        - 特征选择结果
        - 数据增强效果
        - 所有模型性能
        - 最佳组合详情
        - 技术说明
    """
```

### print_results_summary

```python
def print_results_summary(self, results, best_combination):
    """
    打印结果摘要
    
    参数:
        results (dict): 所有结果
        best_combination (dict): 最佳组合
    
    示例:
        >>> pipeline.print_results_summary(results, best)
    
    输出内容:
        - 数据概览
        - 最佳模型信息
        - 性能指标
        - 目标达成情况
    """
```

---

## ⚙ 配置管理

### project_config.py

项目配置模块，集中管理所有参数。

```python
# 导入配置
from config.project_config import (
    TARGET_FEATURES,
    RANDOM_STATE,
    MODEL_CONFIGS,
    PERFORMANCE_TARGETS
)

# 使用配置
pipeline = UltimateEnsemblePipeline(
    target_features=TARGET_FEATURES,
    random_state=RANDOM_STATE
)
```

#### 主要配置项

```python
# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# 数据处理配置
TARGET_FEATURES = 3
TARGET_KEYWORDS = ['目标', 'target', 'y', '标签', 'label']

# 模型配置
RANDOM_STATE = 42
MODEL_CONFIGS = {
    'RandomForest_Aggressive': {
        'n_estimators': 500,
        'max_depth': None,
        # ...
    }
}

# 性能目标
PERFORMANCE_TARGETS = {
    'train_r2_min': 0.9,
    'cv_r2_min': 0.85,
    # ...
}
```

#### 配置函数

```python
def create_directories():
    """创建必要的目录结构"""

def get_config_summary():
    """获取配置摘要"""

def validate_config():
    """验证配置的有效性"""
```

---

##  使用示例

### 基本使用

```python
# 导入必要模块
from src.modeling.ultimate_ensemble_pipeline import UltimateEnsemblePipeline

# 创建管道实例
pipeline = UltimateEnsemblePipeline(target_features=3, random_state=42)

# 运行完整管道
results, best = pipeline.run_ultimate_pipeline('data/raw/数据.xlsx')

# 查看最佳结果
print(f"最佳模型: {best['model']}")
print(f"训练集R²: {best['train_r2']:.4f}")
print(f"交叉验证R²: {best['cv_r2_mean']:.4f}")
```

### 高级使用

```python
# 自定义配置
from config.project_config import MODEL_CONFIGS

# 修改模型参数
MODEL_CONFIGS['MLP_Deep']['hidden_layer_sizes'] = (300, 150, 75)

# 创建自定义管道
pipeline = UltimateEnsemblePipeline(target_features=5)

# 分步执行
X, y = pipeline.load_and_preprocess_data('data.xlsx')
X_selected = pipeline.advanced_feature_selection(X, y)
datasets = pipeline.create_augmented_datasets(X_selected, y)
models = pipeline.create_advanced_models()

# 评估特定组合
for model_name, model in models.items():
    for dataset_name, (X_data, y_data) in datasets.items():
        result = pipeline.evaluate_single_combination(
            model, X_data, y_data, model_name, dataset_name
        )
        print(f"{model_name} + {dataset_name}: R² = {result['test_r2']:.4f}")
```

### 批量实验

```python
# 批量测试不同参数
results_summary = []

for target_features in [3, 5, 7]:
    for random_state in [42, 123, 456]:
        pipeline = UltimateEnsemblePipeline(
            target_features=target_features,
            random_state=random_state
        )
        
        results, best = pipeline.run_ultimate_pipeline('data.xlsx')
        
        results_summary.append({
            'target_features': target_features,
            'random_state': random_state,
            'best_model': best['model'],
            'best_r2': best['cv_r2_mean']
        })

# 分析最佳参数组合
import pandas as pd
summary_df = pd.DataFrame(results_summary)
print(summary_df.groupby('target_features')['best_r2'].mean())
```

---

##  注意事项

### 性能优化

1. **内存管理**
   - 大数据集时使用分块处理
   - 及时释放不需要的变量
   - 监控内存使用情况

2. **计算优化**
   - 使用并行处理（n_jobs=-1）
   - 缓存中间结果
   - 选择合适的算法复杂度

3. **参数调优**
   - 根据数据规模调整参数
   - 使用交叉验证选择最佳参数
   - 平衡性能和计算成本

### 错误处理

1. **数据问题**
   - 检查数据格式和完整性
   - 处理缺失值和异常值
   - 验证特征和目标的数据类型

2. **模型问题**
   - 捕获训练失败的异常
   - 提供备选模型方案
   - 记录详细的错误信息

3. **资源问题**
   - 监控内存和CPU使用
   - 设置合理的超时时间
   - 提供降级处理方案

### 最佳实践

1. **实验管理**
   - 记录所有实验参数
   - 保存模型和结果
   - 使用版本控制

2. **代码质量**
   - 编写清晰的文档
   - 添加类型提示
   - 进行单元测试

3. **可重现性**
   - 固定随机种子
   - 记录环境信息
   - 提供完整的依赖列表

---

*本API文档提供了项目中所有重要函数和类的详细说明。如需更多信息，请参考源代码中的详细注释。*