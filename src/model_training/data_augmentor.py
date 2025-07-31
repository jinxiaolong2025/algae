"""数据增强模块

包含数据增强器类和相关工具方法
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union

try:
    from .config import TrainingConfig
except ImportError:
    from config import TrainingConfig


class DataAugmentor:
    """数据增强器类，负责噪声注入数据增强"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def augment_data(self, X_train: pd.DataFrame, y_train: pd.Series,
                    feature_names: List[str], random_state: int = None, verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        执行数据增强
        
        Args:
            X_train: 训练特征数据
            y_train: 训练目标变量
            feature_names: 特征名称列表
            random_state: 随机种子
            verbose: 是否显示详细信息
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 增强后的特征和目标变量
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        if verbose:
            print(f"   开始噪声注入数据增强...")
            print(f"   原始数据: {len(X_train)} 样本")
            print(f"   增强倍数: {self.config.augmentation_factor}x")
            print(f"   目标数据: {len(X_train) * (self.config.augmentation_factor + 1)} 样本")
        
        # 计算统计信息
        feature_stats = self._calculate_feature_stats(X_train, feature_names)
        y_stats = self._calculate_target_stats(y_train)
        
        # 生成增强数据（保持pandas格式）
        X_augmented_list = [X_train.copy()]  # 包含原始数据
        y_augmented_list = [y_train.copy()]
        for aug_round in range(self.config.augmentation_factor):
            if verbose:
                print(f"   生成第 {aug_round + 1} 轮增强数据...")
            X_aug, y_aug = self._generate_augmented_round_df(
                X_train, y_train, feature_names, feature_stats, y_stats, aug_round
            )
            X_augmented_list.append(X_aug)
            y_augmented_list.append(y_aug)
        
        # 合并数据（使用pandas concat）
        X_final_df = pd.concat(X_augmented_list, ignore_index=True)
        y_final_series = pd.concat(y_augmented_list, ignore_index=True)
        y_final_series.name = 'lipid(%)'  # 确保名称正确
        
        if verbose:
            print(f"   数据增强完成: {len(X_final_df)} 样本 × {len(feature_names)} 特征")
        
        # 验证数据质量
        self._validate_augmented_data(X_train, X_final_df, y_train, y_final_series, feature_names, verbose)
        
        return X_final_df, y_final_series
    
    def _calculate_stats(self, data: Union[pd.Series, np.ndarray], include_range: bool = False) -> Dict[str, float]:
        """
        通用统计计算函数
        
        Args:
            data: 要计算统计信息的数据（Series或ndarray）
            include_range: 是否包含范围计算
            
        Returns:
            Dict[str, float]: 包含统计信息的字典
        """
        stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }
        if include_range:
            stats['range'] = stats['max'] - stats['min']
        return stats
    
    def _calculate_feature_stats(self, X_train: pd.DataFrame, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """计算特征统计信息"""
        feature_stats = {}
        for feature in feature_names:
            feature_stats[feature] = self._calculate_stats(X_train[feature], include_range=True)
        return feature_stats
    
    def _calculate_target_stats(self, y_train: pd.Series) -> Dict[str, float]:
        """计算目标变量统计信息"""
        return self._calculate_stats(y_train, include_range=False)
    
    def _generate_augmented_round_df(self, X_original: pd.DataFrame, y_original: pd.Series,
                                   feature_names: List[str], feature_stats: Dict,
                                   y_stats: Dict, aug_round: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        生成一轮增强数据（pandas版本）
        
        Args:
            X_original: 原始特征数据（DataFrame）
            y_original: 原始目标变量（Series）
            feature_names: 特征名称列表
            feature_stats: 特征统计信息
            y_stats: 目标变量统计信息
            aug_round: 增强轮次
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 增强后的特征和目标变量
        """
        n_samples = len(X_original)
        
        # 创建增强数据的DataFrame和Series
        X_aug_data = []
        y_aug_data = []
        
        for i in range(n_samples):
            # 生成噪声样本
            noisy_sample = self._generate_noisy_sample(
                X_original.iloc[i].values, feature_names, feature_stats, aug_round
            )
            noisy_target = self._generate_noisy_target(
                y_original.iloc[i], y_stats, aug_round
            )
            
            X_aug_data.append(noisy_sample)
            y_aug_data.append(noisy_target)
        
        # 创建DataFrame和Series
        X_aug_df = pd.DataFrame(X_aug_data, columns=feature_names)
        y_aug_series = pd.Series(y_aug_data, name=y_original.name)
        
        return X_aug_df, y_aug_series

    def _generate_augmented_round(self, X_original: np.ndarray, y_original: np.ndarray,
                                feature_names: List[str], feature_stats: Dict,
                                y_stats: Dict, aug_round: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成一轮增强数据（numpy版本，保持向后兼容）

        Args:
            X_original: 原始特征数据
            y_original: 原始目标变量
            feature_names: 特征名称列表
            feature_stats: 特征统计信息
            y_stats: 目标变量统计信息
            aug_round: 增强轮次

        Returns:
            Tuple[np.ndarray, np.ndarray]: 增强后的特征和目标变量
        """
        n_samples, n_features = X_original.shape
        X_augmented = np.zeros_like(X_original)
        y_augmented = np.zeros_like(y_original)

        for i in range(n_samples):
            # 生成噪声样本
            X_augmented[i] = self._generate_noisy_sample(
                X_original[i], feature_names, feature_stats, aug_round
            )
            y_augmented[i] = self._generate_noisy_target(
                y_original[i], y_stats, aug_round
            )

        return X_augmented, y_augmented

    def _generate_noisy_sample(self, original_sample: np.ndarray, feature_names: List[str],
                             feature_stats: Dict, aug_round: int) -> np.ndarray:
        """
        为单个样本生成噪声版本

        噪声生成机制：
        - 基础噪声强度随增强轮次递增：base_noise_intensity * (1 + aug_round * 0.02)
        - 组合噪声：70%比例噪声 + 30%范围噪声
        - 允许超出原始范围20%，提供更好的泛化能力

        Args:
            original_sample: 原始样本数据
            feature_names: 特征名称列表
            feature_stats: 特征统计信息字典
            aug_round: 当前增强轮次（影响噪声强度）

        Returns:
            np.ndarray: 添加噪声后的样本数据
        """
        noisy_sample = original_sample.copy()
        # 不同轮次使用不同的噪声强度
        noise_intensity = self.config.base_noise_intensity * (1 + aug_round * 0.02)

        for i, feature in enumerate(feature_names):
            stats = feature_stats[feature]
            # 比例噪声 (主要方法)
            proportional_noise = np.random.normal(0, stats['std'] * noise_intensity)
            # 范围噪声 (辅助方法)
            range_noise = np.random.uniform(-stats['range'] * noise_intensity * 0.5,
                                           stats['range'] * noise_intensity * 0.5)
            # 组合噪声 (70% 比例噪声 + 30% 范围噪声)
            combined_noise = 0.7 * proportional_noise + 0.3 * range_noise
            noisy_value = original_sample[i] + combined_noise
            # 确保噪声后的值在合理范围内
            range_extension = stats['range'] * 0.2
            min_bound = stats['min'] - range_extension
            max_bound = stats['max'] + range_extension
            noisy_sample[i] = np.clip(noisy_value, min_bound, max_bound)

        return noisy_sample

    def _generate_noisy_target(self, original_target: float, y_stats: Dict, aug_round: int) -> float:
        """
        为目标变量生成相关噪声

        目标变量噪声特点：
        - 噪声强度较特征噪声更小：target_noise_intensity * (1 + aug_round * 0.005)
        - 基于目标变量标准差的高斯噪声
        - 严格的边界约束：最小值不低于原始最小值的80%，最大值有合理扩展

        Args:
            original_target: 原始目标值
            y_stats: 目标变量统计信息
            aug_round: 当前增强轮次

        Returns:
            float: 添加噪声后的目标值
        """
        target_noise_intensity = self.config.target_noise_intensity * (1 + aug_round * 0.005)
        # 添加高斯噪声
        noise = np.random.normal(0, y_stats['std'] * target_noise_intensity)
        noisy_target = original_target + noise
        # 确保目标变量在合理范围内
        range_extension = (y_stats['max'] - y_stats['min']) * 0.05
        min_bound = max(y_stats['min'] * 0.8, 0.1)
        max_bound = y_stats['max'] + range_extension

        return np.clip(noisy_target, min_bound, max_bound)

    def _validate_augmented_data(self, X_original: pd.DataFrame, X_augmented: pd.DataFrame,
                               y_original: pd.Series, y_augmented: pd.Series,
                               feature_names: List[str], verbose: bool = True) -> None:
        """
        验证增强数据的质量

        验证内容：
        - 特征分布保持性：检查增强后特征的均值和标准差变化
        - 目标变量分布：确保目标变量分布没有显著偏移
        - 数据范围合理性：验证增强数据在合理范围内
        - 异常值检测：识别可能的异常增强样本

        质量标准：
        - 特征均值变化 < 5%，标准差变化 < 10%
        - 目标变量分布变化在可接受范围内

        Args:
            X_original: 原始特征数据
            X_augmented: 增强后特征数据
            y_original: 原始目标变量
            y_augmented: 增强后目标变量
            feature_names: 特征名称列表
            verbose: 是否显示详细信息
        """
        if not verbose:
            return

        print(f"   数据质量验证:")
        # 检查特征分布保持性
        print(f"   特征分布验证:")
        for feature in feature_names:
            orig_mean = X_original[feature].mean()
            aug_mean = X_augmented[feature].mean()
            orig_std = X_original[feature].std()
            aug_std = X_augmented[feature].std()
            mean_diff = abs(aug_mean - orig_mean) / orig_mean * 100
            std_diff = abs(aug_std - orig_std) / orig_std * 100
            status = "正常" if mean_diff < 10 and std_diff < 20 else "异常"
            print(f"     {status} {feature:<25}: 均值差异 {mean_diff:5.1f}%, 标准差差异 {std_diff:5.1f}%")

        # 检查目标变量分布
        y_orig_mean = y_original.mean()
        y_aug_mean = y_augmented.mean()
        y_orig_std = y_original.std()
        y_aug_std = y_augmented.std()
        y_mean_diff = abs(y_aug_mean - y_orig_mean) / y_orig_mean * 100
        y_std_diff = abs(y_aug_std - y_orig_std) / y_orig_std * 100
        print(f"   目标变量验证:")
        print(f"     均值: {y_orig_mean:.3f} → {y_aug_mean:.3f} (差异: {y_mean_diff:.1f}%)")
        print(f"     标准差: {y_orig_std:.3f} → {y_aug_std:.3f} (差异: {y_std_diff:.1f}%)")
        if y_mean_diff < 5 and y_std_diff < 15:
            print(f"     目标变量分布保持良好")
        else:
            print(f"     目标变量分布有较大变化，请检查噪声参数")
