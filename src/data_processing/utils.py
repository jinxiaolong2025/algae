"""数据预处理通用工具函数

包含数据预处理过程中使用的通用工具函数和辅助方法。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Any
try:
    from .config import ProcessingConfig
except ImportError:
    from config import ProcessingConfig


def setup_matplotlib_chinese():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def calculate_skewness_kurtosis(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Index]:
    """统一计算偏度和峰度的函数
    
    Args:
        data: 输入数据框
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Index]: 偏度值、峰度值、数值列名
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    skewness_values = data[numeric_cols].skew()
    kurtosis_values = data[numeric_cols].kurtosis()
    return skewness_values, kurtosis_values, numeric_cols


def setup_subplot_layout(n_items: int, n_cols: int = 4) -> Tuple[int, int]:
    """通用的子图布局设置函数
    
    Args:
        n_items: 子图数量
        n_cols: 每行列数
        
    Returns:
        Tuple[int, int]: 行数、列数
    """
    n_rows = (n_items + n_cols - 1) // n_cols
    return n_rows, n_cols


def hide_extra_subplots(axes: Any, n_items: int, n_rows: int, n_cols: int) -> None:
    """隐藏多余的子图
    
    Args:
        axes: matplotlib轴对象
        n_items: 实际子图数量
        n_rows: 总行数
        n_cols: 总列数
    """
    for idx in range(n_items, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if row < n_rows and col < n_cols:
            if n_rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)


def save_plot(config: ProcessingConfig, filename: str, dpi: int = None) -> str:
    """统一的图片保存函数
    
    Args:
        config: 配置对象
        filename: 文件名（相对于results_dir）
        dpi: 图片分辨率
        
    Returns:
        str: 完整的保存路径
    """
    if dpi is None:
        dpi = config.figure_dpi
    
    full_path = os.path.join(config.results_dir, filename)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    return full_path


def interpret_skewness(skew_val: float) -> str:
    """偏度解释函数
    
    Args:
        skew_val: 偏度值
        
    Returns:
        str: 偏度解释
    """
    if skew_val > 2:
        return '严重右偏'
    elif skew_val > 1:
        return '中度右偏'
    elif skew_val > 0.5:
        return '轻度右偏'
    elif skew_val > -0.5:
        return '近似对称'
    elif skew_val > -1:
        return '轻度左偏'
    elif skew_val > -2:
        return '中度左偏'
    else:
        return '严重左偏'


def interpret_kurtosis(kurt_val: float) -> str:
    """峰度解释函数
    
    Args:
        kurt_val: 峰度值
        
    Returns:
        str: 峰度解释
    """
    if kurt_val > 3:
        return '高峰态'
    elif kurt_val > 0:
        return '中峰态'
    else:
        return '低峰态'


def categorize_skewness(skewness_values: pd.Series, config: ProcessingConfig) -> dict:
    """根据偏度值对特征进行分类
    
    Args:
        skewness_values: 偏度值序列
        config: 配置对象
        
    Returns:
        dict: 分类结果
    """
    categories = {
        'light_skew': [],  # 轻度偏斜 (|偏度| < 1)
        'moderate_skew': [],  # 中度偏斜 (1 <= |偏度| < 2)
        'heavy_skew': []  # 重度偏斜 (|偏度| >= 2)
    }
    
    for feature, skew_val in skewness_values.items():
        abs_skew = abs(skew_val)
        if abs_skew < config.light_skew_threshold:
            categories['light_skew'].append(feature)
        elif abs_skew < config.moderate_skew_threshold:
            categories['moderate_skew'].append(feature)
        else:
            categories['heavy_skew'].append(feature)
    
    return categories


def calculate_correlation_matrix(data: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, List[Tuple]]:
    """计算相关性矩阵并找出高相关性特征对
    
    Args:
        data: 输入数据框
        threshold: 相关性阈值
        
    Returns:
        Tuple[pd.DataFrame, List[Tuple]]: 相关性矩阵、高相关性特征对列表
    """
    # 获取数值特征
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    # 找出高相关性特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_val))
    
    # 按相关性绝对值排序
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return corr_matrix, high_corr_pairs


def print_section_header(title: str, width: int = 80) -> None:
    """打印章节标题
    
    Args:
        title: 标题文本
        width: 标题宽度
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection_header(title: str) -> None:
    """打印子章节标题
    
    Args:
        title: 标题文本
    """
    print(f"\n{title}")


def format_percentage(value: float, total: int) -> str:
    """格式化百分比显示
    
    Args:
        value: 数值
        total: 总数
        
    Returns:
        str: 格式化的百分比字符串
    """
    percentage = (value / total) * 100 if total > 0 else 0
    return f"{percentage:.2f}% ({int(value)}个)"


def validate_data_integrity(data: pd.DataFrame, stage: str = "") -> None:
    """验证数据完整性
    
    Args:
        data: 数据框
        stage: 处理阶段描述
    """
    if data.empty:
        raise ValueError(f"数据为空 - {stage}")
    
    if data.isnull().all().all():
        raise ValueError(f"数据全为缺失值 - {stage}")
    
    print(f"   数据完整性验证通过 - {stage}: 形状{data.shape}")


def log_processing_step(step_name: str, details: str = "") -> None:
    """记录处理步骤
    
    Args:
        step_name: 步骤名称
        details: 详细信息
    """
    print(f"\n{step_name}...")
    if details:
        print(f"   {details}")
