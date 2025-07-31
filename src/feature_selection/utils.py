"""
特征选择工具函数模块

该模块包含特征选择过程中使用的通用工具函数。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold

# 设置中文字体和警告过滤
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def setup_matplotlib_chinese():
    """设置matplotlib中文字体"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def print_section_header(title: str, width: int = 60):
    """打印章节标题"""
    print("=" * width)
    print(title)
    print("=" * width)


def print_subsection_header(title: str, width: int = 40):
    """打印子章节标题"""
    print("-" * width)
    print(title)
    print("-" * width)


def log_processing_step(step_number: int, description: str):
    """记录处理步骤"""
    print(f"\n{step_number}. {description}")


def validate_data_shape(data: np.ndarray, expected_dims: int = 2) -> bool:
    """验证数据形状"""
    if data.ndim != expected_dims:
        raise ValueError(f"数据维度错误: 期望{expected_dims}维，实际{data.ndim}维")
    return True


def validate_feature_indices(indices: np.ndarray, max_features: int) -> bool:
    """验证特征索引的有效性"""
    if len(indices) == 0:
        raise ValueError("特征索引不能为空")
    
    if np.any(indices < 0) or np.any(indices >= max_features):
        raise ValueError(f"特征索引超出范围: [0, {max_features-1}]")
    
    if len(np.unique(indices)) != len(indices):
        raise ValueError("特征索引包含重复值")
    
    return True


def calculate_hamming_distance(individual1: np.ndarray, individual2: np.ndarray) -> int:
    """计算两个个体之间的汉明距离"""
    return np.sum(individual1 != individual2)


def calculate_population_diversity(population: np.ndarray) -> float:
    """计算种群多样性"""
    if len(population) <= 1:
        return 0.0
    
    diversity_sum = 0
    count = 0
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            diversity_sum += calculate_hamming_distance(population[i], population[j])
            count += 1
    
    return diversity_sum / count if count > 0 else 0.0


def evaluate_model_performance(model, X: np.ndarray, y: np.ndarray, 
                             cv_folds: int = 5, random_state: int = 42) -> Dict[str, float]:
    """评估模型性能"""
    try:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        
        return {
            'mean_r2': np.mean(scores),
            'std_r2': np.std(scores),
            'min_r2': np.min(scores),
            'max_r2': np.max(scores),
            'scores': scores.tolist()
        }
    except Exception as e:
        return {
            'mean_r2': np.nan,
            'std_r2': np.nan,
            'min_r2': np.nan,
            'max_r2': np.nan,
            'scores': [np.nan] * cv_folds,
            'error': str(e)
        }


def calculate_feature_importance_ranking(X: np.ndarray, y: np.ndarray, 
                                       feature_names: np.ndarray) -> pd.DataFrame:
    """计算特征重要性排序"""
    from sklearn.ensemble import RandomForestRegressor
    
    # 使用随机森林计算特征重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance': rf.feature_importances_,
        'rank': range(1, len(feature_names) + 1)
    })
    
    # 按重要性降序排列
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return importance_df


def calculate_feature_correlation_matrix(X: np.ndarray, feature_names: np.ndarray) -> pd.DataFrame:
    """计算特征相关性矩阵"""
    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # 计算相关性矩阵
    correlation_matrix = df.corr()
    
    return correlation_matrix


def find_highly_correlated_features(correlation_matrix: pd.DataFrame, 
                                  threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """找出高度相关的特征对"""
    highly_correlated = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                highly_correlated.append((feature1, feature2, corr_value))
    
    return highly_correlated


def calculate_convergence_metrics(fitness_history: List[float]) -> Dict[str, float]:
    """计算收敛指标"""
    if len(fitness_history) < 2:
        return {'convergence_rate': 0.0, 'stability_index': 0.0, 'improvement_ratio': 0.0}
    
    # 收敛速度：最后10%代数的平均改进
    last_10_percent = max(1, len(fitness_history) // 10)
    recent_improvements = np.diff(fitness_history[-last_10_percent:])
    convergence_rate = np.mean(recent_improvements) if len(recent_improvements) > 0 else 0.0
    
    # 稳定性指标：最后20%代数的标准差
    last_20_percent = max(1, len(fitness_history) // 5)
    stability_index = np.std(fitness_history[-last_20_percent:])
    
    # 改进比例：有改进的代数占总代数的比例
    improvements = np.diff(fitness_history)
    improvement_ratio = np.sum(improvements > 0) / len(improvements) if len(improvements) > 0 else 0.0
    
    return {
        'convergence_rate': convergence_rate,
        'stability_index': stability_index,
        'improvement_ratio': improvement_ratio
    }


def format_feature_list(features: List[str], max_per_line: int = 3) -> str:
    """格式化特征列表为多行显示"""
    formatted_lines = []
    for i in range(0, len(features), max_per_line):
        line_features = features[i:i + max_per_line]
        formatted_lines.append(", ".join(line_features))
    
    return "\n     ".join(formatted_lines)


def save_results_summary(results: Dict[str, Any], filepath: str):
    """保存结果摘要到文本文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("特征选择结果摘要\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            if isinstance(value, (list, np.ndarray)):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  - {item}\n")
            elif isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")
            f.write("\n")


def check_xgboost_availability() -> bool:
    """检查XGBoost是否可用"""
    try:
        import xgboost as xgb
        return True
    except ImportError:
        return False


def get_color_palette(n_colors: int) -> List[str]:
    """获取颜色调色板"""
    if n_colors <= 10:
        # 使用预定义的颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return colors[:n_colors]
    else:
        # 使用matplotlib的颜色映射
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        return [cmap(i / n_colors) for i in range(n_colors)]


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """创建进度条字符串"""
    progress = current / total
    filled_width = int(width * progress)
    bar = '█' * filled_width + '░' * (width - filled_width)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"


def validate_input_data(X: np.ndarray, y: np.ndarray, feature_names: Optional[np.ndarray] = None):
    """验证输入数据的有效性"""
    # 检查数据类型
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X和y必须是numpy数组")
    
    # 检查数据形状
    if X.ndim != 2:
        raise ValueError("X必须是二维数组")
    
    if y.ndim != 1:
        raise ValueError("y必须是一维数组")
    
    # 检查样本数量一致性
    if X.shape[0] != len(y):
        raise ValueError("X和y的样本数量不一致")
    
    # 检查特征名称
    if feature_names is not None:
        if len(feature_names) != X.shape[1]:
            raise ValueError("特征名称数量与特征数量不一致")
    
    # 检查数据中是否包含NaN或无穷大
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X包含NaN或无穷大值")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y包含NaN或无穷大值")
    
    return True


def calculate_selection_pressure(fitness_scores: np.ndarray) -> float:
    """计算选择压力"""
    if len(fitness_scores) <= 1:
        return 0.0
    
    mean_fitness = np.mean(fitness_scores)
    max_fitness = np.max(fitness_scores)
    
    if mean_fitness == 0:
        return float('inf') if max_fitness > 0 else 0.0
    
    return max_fitness / mean_fitness


def detect_premature_convergence(fitness_history: List[float], 
                                window_size: int = 20, 
                                threshold: float = 1e-6) -> bool:
    """检测是否过早收敛"""
    if len(fitness_history) < window_size:
        return False
    
    recent_fitness = fitness_history[-window_size:]
    fitness_variance = np.var(recent_fitness)
    
    return fitness_variance < threshold
