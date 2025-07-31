"""配置和异常定义模块

包含训练配置类和自定义异常类
"""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练配置类，管理所有训练参数"""
    # 交叉验证配置
    cv_folds: int = 5
    cv_random_state: int = 42
    
    # 数据增强配置
    use_augmentation: bool = True
    augmentation_factor: int = 3
    base_noise_intensity: float = 0.05
    target_noise_intensity: float = 0.015
    
    # 模型配置
    model_random_state: int = 42
    
    # 文件路径配置
    train_data_path: str = "../../data/processed/train_data.csv"
    ga_features_path: str = "../../results/feature_selection/04_feature_selection/ga_selected_features.csv"
    results_dir: str = "../../results/model_training"
    
    # 性能阈值配置
    excellent_r2_threshold: float = 0.9
    good_r2_threshold: float = 0.8
    fair_r2_threshold: float = 0.6


class ModelTrainingError(Exception):
    """模型训练相关异常"""
    pass


class DataLoadError(Exception):
    """数据加载相关异常"""
    pass
