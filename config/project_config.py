# -*- coding: utf-8 -*-
"""
项目配置文件

包含所有可调参数和路径配置

"""

import os
from pathlib import Path

# ==================== 路径配置 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 结果路径
RESULTS_DIR = PROJECT_ROOT / "results"

# 文档路径
DOCS_DIR = PROJECT_ROOT / "docs"

# 源代码路径
SRC_DIR = PROJECT_ROOT / "src"
MODELING_DIR = SRC_DIR / "modeling"
VISUALIZATION_DIR = SRC_DIR / "visualization"
UTILS_DIR = SRC_DIR / "utils"

# 默认数据文件
DEFAULT_DATA_FILE = RAW_DATA_DIR / "数据.xlsx"

# ==================== 数据处理配置 ====================

# 特征选择
TARGET_FEATURES = 3  # 目标特征数量
FEATURE_SELECTION_METHODS = ['f_regression', 'mutual_info_regression', 'variance']

# 目标变量识别关键词
TARGET_KEYWORDS = ['目标', 'target', 'y', '标签', 'label', 'Target', 'Y', 'LABEL']

# 数据预处理
MISSING_VALUE_THRESHOLD = 0.5  # 缺失值比例阈值
CONSTANT_FEATURE_THRESHOLD = 0.95  # 常数特征阈值

# ==================== 数据增强配置 ====================

# SMOGN参数
SMOGN_CONFIG = {
    'k_neighbors': 3,  # K近邻数量
    'noise_factor': 0.01,  # 噪声强度
    'augmentation_factor': 2,  # 增强倍数
    'random_state': 42
}

# 噪声增强参数
NOISE_AUGMENTATION_CONFIG = {
    'noise_factor': 0.05,  # 噪声比例
    'augmentation_factor': 2,  # 增强倍数
    'random_state': 42
}

# ==================== 模型配置 ====================

# 随机种子
RANDOM_STATE = 42

# 交叉验证
CV_CONFIG = {
    'default_folds': 5,  # 默认折数
    'min_folds': 2,  # 最小折数
    'max_folds': 10,  # 最大折数
    'min_samples_per_fold': 3  # 每折最小样本数
}

# 模型参数
MODEL_CONFIGS = {
    'RandomForest_Aggressive': {
        'n_estimators': 500,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'oob_score': True,
        'random_state': RANDOM_STATE
    },
    
    'GradientBoosting_Tuned': {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'random_state': RANDOM_STATE
    },
    
    'ElasticNet_Optimized': {
        'alpha': 0.01,
        'l1_ratio': 0.7,
        'max_iter': 2000,
        'random_state': RANDOM_STATE
    },
    
    'SVR_Polynomial': {
        'kernel': 'poly',
        'degree': 3,
        'C': 1000,
        'gamma': 'scale'
    },
    
    'MLP_Deep': {
        'hidden_layer_sizes': (200, 100, 50),
        'max_iter': 2000,
        'learning_rate': 'adaptive',
        'random_state': RANDOM_STATE
    },
    
    'BayesianEnsemble': {
        'base_models': ['RandomForest', 'GradientBoosting', 'Ridge', 'BayesianRidge'],
        'random_state': RANDOM_STATE
    }
}

# ==================== 评估配置 ====================

# 性能目标
PERFORMANCE_TARGETS = {
    'train_r2_min': 0.9,  # 训练集R²最小值
    'cv_r2_min': 0.85,  # 交叉验证R²最小值
    'test_r2_min': 0.8,  # 测试集R²最小值（参考）
    'overfitting_threshold': 0.1  # 过拟合阈值
}

# 评估指标
EVALUATION_METRICS = {
    'primary': 'r2_score',  # 主要指标
    'secondary': ['mean_squared_error', 'mean_absolute_error'],  # 次要指标
    'custom': ['comprehensive_score']  # 自定义指标
}

# ==================== 可视化配置 ====================

# 图表设置
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'font_size': 12,
    'title_size': 14,
    'label_size': 10
}

# 保存格式
FIGURE_FORMATS = ['png', 'pdf', 'svg']

# ==================== 输出配置 ====================

# 结果文件
RESULT_FILES = {
    'main_results': 'ultimate_ensemble_results.txt',
    'detailed_results': 'detailed_analysis.json',
    'model_comparison': 'model_comparison.csv',
    'feature_importance': 'feature_importance.csv'
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'pipeline.log'
}

# ==================== 高级配置 ====================

# 并行处理
PARALLEL_CONFIG = {
    'n_jobs': -1,  # 使用所有可用CPU核心
    'backend': 'threading',  # 并行后端
    'verbose': 1  # 详细程度
}

# 内存管理
MEMORY_CONFIG = {
    'max_memory_usage': '4GB',  # 最大内存使用
    'chunk_size': 1000,  # 数据块大小
    'cache_size': 100  # 缓存大小
}

# 实验跟踪
EXPERIMENT_CONFIG = {
    'track_experiments': True,  # 是否跟踪实验
    'save_models': True,  # 是否保存模型
    'save_predictions': True,  # 是否保存预测结果
    'save_intermediate': False  # 是否保存中间结果
}

# ==================== 辅助函数 ====================

def create_directories():
    """创建必要的目录结构"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        RESULTS_DIR, DOCS_DIR, SRC_DIR,
        MODELING_DIR, VISUALIZATION_DIR, UTILS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ 目录结构创建完成")

def get_config_summary():
    """获取配置摘要"""
    summary = {
        '项目根目录': str(PROJECT_ROOT),
        '目标特征数': TARGET_FEATURES,
        '随机种子': RANDOM_STATE,
        '交叉验证折数': CV_CONFIG['default_folds'],
        '训练集R²目标': PERFORMANCE_TARGETS['train_r2_min'],
        '交叉验证R²目标': PERFORMANCE_TARGETS['cv_r2_min'],
        'SMOGN增强倍数': SMOGN_CONFIG['augmentation_factor'],
        '噪声增强强度': NOISE_AUGMENTATION_CONFIG['noise_factor']
    }
    
    return summary

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查路径
    if not PROJECT_ROOT.exists():
        errors.append(f"项目根目录不存在: {PROJECT_ROOT}")
    
    # 检查参数范围
    if TARGET_FEATURES < 1 or TARGET_FEATURES > 50:
        errors.append(f"目标特征数超出合理范围: {TARGET_FEATURES}")
    
    if CV_CONFIG['default_folds'] < 2:
        errors.append(f"交叉验证折数过小: {CV_CONFIG['default_folds']}")
    
    # 检查性能目标
    if PERFORMANCE_TARGETS['train_r2_min'] > 1.0:
        errors.append(f"训练集R²目标超出范围: {PERFORMANCE_TARGETS['train_r2_min']}")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    print("✅ 配置验证通过")

if __name__ == "__main__":
    # 创建目录结构
    create_directories()
    
    # 验证配置
    validate_config()
    
    # 显示配置摘要
    print("\n📋 配置摘要:")
    for key, value in get_config_summary().items():
        print(f"  {key}: {value}")