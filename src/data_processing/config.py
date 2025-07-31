"""数据预处理配置模块

定义数据预处理过程中使用的所有配置参数和常量。
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any


class DataProcessingError(Exception):
    """数据预处理异常基类"""
    pass


class DataLoadError(DataProcessingError):
    """数据加载异常"""
    pass


class MissingValueError(DataProcessingError):
    """缺失值处理异常"""
    pass


class OutlierHandlingError(DataProcessingError):
    """异常值处理异常"""
    pass


@dataclass
class ProcessingConfig:
    """数据预处理配置类"""
    
    # 文件路径配置
    raw_data_path: str = "data/raw/数据.xlsx"
    results_dir: str = "results/data_preprocess/"
    raw_analysis_dir: str = "results/data_preprocess/raw_analysis/"
    after_filling_dir: str = "results/data_preprocess/after_filling/"
    processed_data_dir: str = "data/processed/"
    
    # 数据分割配置
    test_size: float = 0.2
    random_state: int = 42
    
    # 缺失值处理配置
    correlation_threshold: float = 0.5  # 用于选择回归填充的相关性阈值
    knn_neighbors: int = 5  # KNN填充的邻居数量
    
    # 异常值处理配置
    light_skew_threshold: float = 1.0  # 轻度偏斜阈值
    moderate_skew_threshold: float = 2.0  # 中度偏斜阈值
    light_percentile_range: tuple = (5, 95)  # 轻度偏斜分位数范围
    moderate_percentile_range: tuple = (10, 90)  # 中度偏斜分位数范围
    heavy_percentile_range: tuple = (5, 95)  # 重度偏斜分位数范围
    
    # 可视化配置
    figure_dpi: int = 300
    subplot_cols: int = 4
    correlation_threshold_viz: float = 0.5  # 可视化相关性阈值
    max_scatter_pairs: int = 24  # 散点图最大显示对数
    
    # 目标变量配置
    target_column: str = "lipid(%)"
    exclude_columns: List[str] = None  # 需要排除的列
    
    # 偏度和峰度解释阈值
    skewness_thresholds: Dict[str, float] = None
    kurtosis_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.exclude_columns is None:
            self.exclude_columns = ['S(%)']
        
        if self.skewness_thresholds is None:
            self.skewness_thresholds = {
                'severe_right': 2.0,
                'moderate_right': 1.0,
                'light_right': 0.5,
                'symmetric': -0.5,
                'light_left': -1.0,
                'moderate_left': -2.0
            }
        
        if self.kurtosis_thresholds is None:
            self.kurtosis_thresholds = {
                'high_peak': 3.0,
                'normal_peak': 0.0
            }
        
        # 不在初始化时创建目录，延迟到实际使用时创建
    
    def ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            self.results_dir,
            self.raw_analysis_dir,
            self.after_filling_dir,
            self.processed_data_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_output_paths(self) -> Dict[str, str]:
        """获取输出文件路径"""
        return {
            'train_data': os.path.join(self.processed_data_dir, 'train_data.csv'),
            'test_data': os.path.join(self.processed_data_dir, 'test_data.csv'),
            'processed_data': os.path.join(self.processed_data_dir, 'processed_data.csv'),
            'quality_analysis': os.path.join(self.raw_analysis_dir, 'data_quality_analysis.csv'),
            'skewness_analysis': os.path.join(self.after_filling_dir, 'skewness_kurtosis_analysis.csv')
        }
    
    def get_visualization_paths(self) -> Dict[str, str]:
        """获取可视化文件路径"""
        return {
            'boxplots': os.path.join(self.after_filling_dir, 'boxplots_after_filling.png'),
            'scatter_plots': os.path.join(self.after_filling_dir, 'scatter_plots_after_filling.png'),
            'histograms': os.path.join(self.after_filling_dir, 'histograms_after_filling.png'),
            'skewness_improvement': os.path.join(self.results_dir, 'skewness_improvement_analysis.png')
        }


# 全局默认配置实例
DEFAULT_CONFIG = ProcessingConfig()
