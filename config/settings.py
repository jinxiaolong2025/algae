# -*- coding: utf-8 -*-
"""
项目配置文件
Project Configuration Settings
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径配置
DATA_PATHS = {
    'raw_data': PROJECT_ROOT / 'data' / 'raw' / '数据.xlsx',
    'clean_data': PROJECT_ROOT / 'data' / 'processed' / 'clean_data.csv',
    'enhanced_data': PROJECT_ROOT / 'data' / 'processed' / 'enhanced_clean_data.csv',
    'selected_features': PROJECT_ROOT / 'data' / 'features' / 'selected_features_data.csv',
    'feature_list': PROJECT_ROOT / 'data' / 'features' / 'rfecv_selected_features.txt'
}

# 结果输出路径配置
RESULT_PATHS = {
    'preprocessing': PROJECT_ROOT / 'results' / 'preprocessing',
    'feature_selection': PROJECT_ROOT / 'results' / 'feature_selection',
    'modeling': PROJECT_ROOT / 'results' / 'modeling',
    'evaluation': PROJECT_ROOT / 'results' / 'evaluation'
}

# 模型保存路径
MODEL_SAVE_PATH = PROJECT_ROOT / 'models'

# 专利支持文件路径
PATENT_PATHS = {
    'documents': PROJECT_ROOT / 'patent' / 'documents',
    'figures': PROJECT_ROOT / 'patent' / 'figures'
}

# 模型训练参数
MODEL_CONFIG = {
    'random_state': 42,
    'cv_folds': 5,
    'test_size': 0.2,
    'n_jobs': -1
}

# 特征选择参数
FEATURE_CONFIG = {
    'correlation_threshold': 0.1,
    'mutual_info_k_best': 20,
    'univariate_k_best': 15,
    'tree_based_n_features': 15
}

# 可视化配置
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'font_family': ['Arial Unicode MS', 'DejaVu Sans'],
    'style': 'seaborn-v0_8'
}

# 目标变量名
TARGET_COLUMN = 'lipid(%)'

# 确保目录存在
for path_dict in [DATA_PATHS, RESULT_PATHS, PATENT_PATHS]:
    for path in path_dict.values():
        if isinstance(path, Path) and not path.name.endswith(('.csv', '.xlsx', '.txt', '.md', '.png')):
            path.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)