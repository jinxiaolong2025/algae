"""
特征选择模块配置管理

该模块包含特征选择过程中的所有配置参数、异常定义和路径管理。
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# 自定义异常类
class FeatureSelectionError(Exception):
    """特征选择基础异常"""
    pass


class DataLoadError(FeatureSelectionError):
    """数据加载异常"""
    pass


class GeneticAlgorithmError(FeatureSelectionError):
    """遗传算法异常"""
    pass


class ModelEvaluationError(FeatureSelectionError):
    """模型评估异常"""
    pass


class VisualizationError(FeatureSelectionError):
    """可视化异常"""
    pass


@dataclass
class FeatureSelectionConfig:
    """特征选择配置类"""
    
    # 数据路径配置
    train_data_path: str = "../../data/processed/train_data.csv"
    results_dir: str = "../../results/feature_selection/"
    
    # 遗传算法参数
    population_size: int = 30
    generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.85
    elite_size: int = 6
    target_features: int = 6
    cv_folds: int = 5
    random_state: int = 42
    
    # 模型评估参数
    tournament_size: int = 3
    n_estimators: int = 30
    max_depth: int = 3
    learning_rate: float = 0.1
    
    # 可视化参数
    figure_dpi: int = 300
    figure_size_large: tuple = (20, 8)
    figure_size_medium: tuple = (15, 10)
    figure_size_small: tuple = (12, 8)
    
    # 目标变量
    target_column: str = "lipid(%)"
    
    # 子目录配置
    data_analysis_dir: str = "01_data_analysis"
    evolution_process_dir: str = "02_evolution_process"
    convergence_analysis_dir: str = "03_convergence_analysis"
    feature_selection_dir: str = "04_feature_selection"
    model_validation_dir: str = "05_model_validation"
    correlation_analysis_dir: str = "06_correlation_analysis"
    final_reports_dir: str = "07_final_reports"

    # 输出文件名配置（带子目录路径）
    selected_features_file: str = "04_feature_selection/ga_selected_features.csv"
    selected_data_file: str = "04_feature_selection/ga_selected_data.csv"
    best_features_list_file: str = "04_feature_selection/best_features_list.csv"
    best_features_data_file: str = "04_feature_selection/best_features_data.csv"
    validation_results_file: str = "05_model_validation/ga_validation_results.csv"
    evolution_process_file: str = "02_evolution_process/ga_evolution_process.png"
    feature_importance_file: str = "05_model_validation/ga_feature_importance_comparison.png"
    model_comparison_file: str = "05_model_validation/ga_model_comparison.png"
    convergence_analysis_file: str = "03_convergence_analysis/ga_convergence_analysis.png"
    feature_correlation_file: str = "06_correlation_analysis/feature_correlation_heatmap.png"
    population_diversity_file: str = "03_convergence_analysis/population_diversity_analysis.png"
    feature_distribution_file: str = "04_feature_selection/optimal_feature_distribution.png"
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保结果目录存在
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保所有必要的目录存在"""
        # 创建主目录
        os.makedirs(self.results_dir, exist_ok=True)

        # 创建所有子目录
        subdirs = [
            self.data_analysis_dir,
            self.evolution_process_dir,
            self.convergence_analysis_dir,
            self.feature_selection_dir,
            self.model_validation_dir,
            self.correlation_analysis_dir,
            self.final_reports_dir
        ]

        for subdir in subdirs:
            full_path = os.path.join(self.results_dir, subdir)
            os.makedirs(full_path, exist_ok=True)
    
    def get_file_path(self, filename: str) -> str:
        """获取完整的文件路径"""
        return os.path.join(self.results_dir, filename)
    
    def get_output_paths(self) -> Dict[str, str]:
        """获取所有输出文件的完整路径"""
        return {
            'selected_features': self.get_file_path(self.selected_features_file),
            'selected_data': self.get_file_path(self.selected_data_file),
            'best_features_list': self.get_file_path(self.best_features_list_file),
            'best_features_data': self.get_file_path(self.best_features_data_file),
            'validation_results': self.get_file_path(self.validation_results_file),
            'evolution_process': self.get_file_path(self.evolution_process_file),
            'feature_importance': self.get_file_path(self.feature_importance_file),
            'model_comparison': self.get_file_path(self.model_comparison_file),
            'convergence_analysis': self.get_file_path(self.convergence_analysis_file),
            'feature_correlation': self.get_file_path(self.feature_correlation_file),
            'population_diversity': self.get_file_path(self.population_diversity_file),
            'feature_distribution': self.get_file_path(self.feature_distribution_file)
        }
    
    def get_genetic_algorithm_params(self) -> Dict[str, Any]:
        """获取遗传算法参数字典"""
        return {
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elite_size': self.elite_size,
            'target_features': self.target_features,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'tournament_size': self.tournament_size
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取模型参数字典"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }
    
    def validate_config(self) -> bool:
        """验证配置参数的有效性"""
        try:
            # 检查基本参数范围
            assert 0 < self.population_size <= 1000, "种群大小必须在1-1000之间"
            assert 0 < self.generations <= 10000, "进化代数必须在1-10000之间"
            assert 0 <= self.mutation_rate <= 1, "变异率必须在0-1之间"
            assert 0 <= self.crossover_rate <= 1, "交叉率必须在0-1之间"
            assert 0 <= self.elite_size < self.population_size, "精英个体数量必须小于种群大小"
            assert 0 < self.target_features <= 50, "目标特征数量必须在1-50之间"
            assert 2 <= self.cv_folds <= 10, "交叉验证折数必须在2-10之间"
            
            # 检查文件路径
            train_dir = os.path.dirname(self.train_data_path)
            if train_dir and not os.path.exists(train_dir):
                raise ValueError(f"训练数据目录不存在: {train_dir}")
            
            return True
            
        except AssertionError as e:
            raise ValueError(f"配置参数验证失败: {e}")
        except Exception as e:
            raise ValueError(f"配置验证过程中出现错误: {e}")
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("特征选择配置摘要:")
        print("=" * 50)
        print(f"数据路径: {self.train_data_path}")
        print(f"结果目录: {self.results_dir}")
        print(f"目标变量: {self.target_column}")
        print()
        print("遗传算法参数:")
        print(f"  种群大小: {self.population_size}")
        print(f"  进化代数: {self.generations}")
        print(f"  变异率: {self.mutation_rate}")
        print(f"  交叉率: {self.crossover_rate}")
        print(f"  精英个体数: {self.elite_size}")
        print(f"  目标特征数: {self.target_features}")
        print(f"  交叉验证折数: {self.cv_folds}")
        print(f"  随机种子: {self.random_state}")
        print("=" * 50)


# 全局配置实例
DEFAULT_CONFIG = FeatureSelectionConfig()


def get_default_config() -> FeatureSelectionConfig:
    """获取默认配置"""
    return DEFAULT_CONFIG


def create_custom_config(**kwargs) -> FeatureSelectionConfig:
    """创建自定义配置"""
    return FeatureSelectionConfig(**kwargs)


# 常用配置预设
QUICK_CONFIG = FeatureSelectionConfig(
    population_size=20,
    generations=50,
    target_features=5
)

THOROUGH_CONFIG = FeatureSelectionConfig(
    population_size=50,
    generations=200,
    target_features=8,
    cv_folds=10
)

EXPERIMENTAL_CONFIG = FeatureSelectionConfig(
    population_size=100,
    generations=500,
    mutation_rate=0.2,
    crossover_rate=0.9,
    target_features=10
)


def get_preset_config(preset_name: str) -> FeatureSelectionConfig:
    """获取预设配置"""
    presets = {
        'quick': QUICK_CONFIG,
        'thorough': THOROUGH_CONFIG,
        'experimental': EXPERIMENTAL_CONFIG,
        'default': DEFAULT_CONFIG
    }
    
    if preset_name not in presets:
        raise ValueError(f"未知的预设配置: {preset_name}. 可用预设: {list(presets.keys())}")
    
    return presets[preset_name]
