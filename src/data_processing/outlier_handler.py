"""异常值处理模块

负责检测和处理数据中的异常值，使用多种策略包括分位数截断、Z-score等。
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Tuple
try:
    from .config import ProcessingConfig, OutlierHandlingError
    from .utils import (
        calculate_skewness_kurtosis,
        categorize_skewness,
        log_processing_step,
        validate_data_integrity
    )
except ImportError:
    from config import ProcessingConfig, OutlierHandlingError
    from utils import (
        calculate_skewness_kurtosis,
        categorize_skewness,
        log_processing_step,
        validate_data_integrity
    )


class OutlierHandler:
    """异常值处理器类"""
    
    def __init__(self, config: ProcessingConfig):
        """初始化异常值处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值 - 基于偏度的分位数截断
        
        Args:
            data: 输入数据
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        log_processing_step("4. 异常值处理")
        
        data_processed = data.copy()
        
        # 计算偏度
        skewness_values, _, numeric_cols = calculate_skewness_kurtosis(data_processed)
        
        # 根据偏度对特征分类
        skew_categories = categorize_skewness(skewness_values, self.config)
        
        # 统计信息
        total_outliers_removed = 0
        processing_summary = {}
        
        print("   异常值处理详情:")
        print("    注意：目标变量 'lipid(%)' 不进行异常值处理")

        # 按原始版本的格式输出处理策略
        light_features = [f for f in skew_categories['light_skew'] if f != self.config.target_column]
        moderate_features = [f for f in skew_categories['moderate_skew'] if f != self.config.target_column]
        heavy_features = [f for f in skew_categories['heavy_skew'] if f != self.config.target_column]

        if light_features:
            print(f"    近似对称与轻度偏斜(|偏度|<1)，5%-95%分位数替换: {len(light_features)}个")
            for feature in light_features:
                if feature in numeric_cols:
                    outliers_removed = self._process_feature_outliers(
                        data_processed, feature, 'light_skew', skewness_values[feature]
                    )
                    total_outliers_removed += outliers_removed
                    processing_summary[feature] = {
                        'category': 'light_skew',
                        'skewness': skewness_values[feature],
                        'outliers_removed': outliers_removed
                    }

        if moderate_features:
            print(f"    中度偏斜(1≤|偏度|<2)，10%-90%分位数替换: {len(moderate_features)}个")
            for feature in moderate_features:
                if feature in numeric_cols:
                    outliers_removed = self._process_feature_outliers(
                        data_processed, feature, 'moderate_skew', skewness_values[feature]
                    )
                    total_outliers_removed += outliers_removed
                    processing_summary[feature] = {
                        'category': 'moderate_skew',
                        'skewness': skewness_values[feature],
                        'outliers_removed': outliers_removed
                    }

        if heavy_features:
            print(f"    严重偏斜(|偏度|≥2)，先对数变换再5%-95%分位数替换: {len(heavy_features)}个")
            for feature in heavy_features:
                if feature in numeric_cols:
                    outliers_removed = self._process_feature_outliers(
                        data_processed, feature, 'heavy_skew', skewness_values[feature]
                    )
                    total_outliers_removed += outliers_removed
                    processing_summary[feature] = {
                        'category': 'heavy_skew',
                        'skewness': skewness_values[feature],
                        'outliers_removed': outliers_removed
                    }
        
        # 验证处理结果
        self._validate_outlier_processing(data, data_processed, total_outliers_removed)
        
        # 保存处理摘要
        self._save_processing_summary(processing_summary)
        
        return data_processed
    
    def _process_feature_outliers(self, data: pd.DataFrame, feature: str,
                                 category: str, skewness: float) -> int:
        """处理单个特征的异常值

        Args:
            data: 数据框
            feature: 特征名
            category: 偏度类别
            skewness: 偏度值

        Returns:
            int: 移除的异常值数量
        """
        original_data = data[feature].copy()

        # 对于重度偏斜，先进行对数变换
        if category == 'heavy_skew':
            return self._process_heavy_skew_feature(data, feature, skewness)

        # 根据偏度类别选择分位数范围
        if category == 'light_skew':
            lower_pct, upper_pct = self.config.light_percentile_range
        elif category == 'moderate_skew':
            lower_pct, upper_pct = self.config.moderate_percentile_range
        else:
            lower_pct, upper_pct = self.config.heavy_percentile_range

        # 计算分位数
        lower_bound = np.percentile(original_data.dropna(), lower_pct)
        upper_bound = np.percentile(original_data.dropna(), upper_pct)

        # 识别异常值
        outlier_mask = (original_data < lower_bound) | (original_data > upper_bound)
        outliers_count = outlier_mask.sum()

        if outliers_count > 0:
            # 截断异常值
            data[feature] = np.clip(original_data, lower_bound, upper_bound)

            print(f"     • {feature}: 偏度={skewness:.3f}, {outliers_count}个异常值({outliers_count/len(data)*100:.1f}%) → {lower_pct}%-{upper_pct}%分位数替换")
        else:
            print(f"     • {feature}: 偏度={skewness:.3f}, 无异常值")

        return outliers_count

    def _process_heavy_skew_feature(self, data: pd.DataFrame, feature: str, original_skewness: float) -> int:
        """处理重度偏斜特征 - 对数变换 + 分位数截断

        Args:
            data: 数据框
            feature: 特征名
            original_skewness: 原始偏度值

        Returns:
            int: 处理的异常值数量
        """
        original_data = data[feature].copy()

        # 确保所有值为正数（对数变换要求）
        min_val = original_data.min()
        if min_val <= 0:
            # 添加常数使所有值为正
            shift_value = abs(min_val) + 1e-6
            data[feature] = original_data + shift_value

        # 应用对数变换
        log_data = np.log1p(data[feature])
        log_skewness = log_data.skew()

        # 对变换后的数据进行分位数截断
        lower_pct, upper_pct = self.config.light_percentile_range  # 使用5%-95%
        lower_bound = np.percentile(log_data.dropna(), lower_pct)
        upper_bound = np.percentile(log_data.dropna(), upper_pct)

        # 识别异常值
        outlier_mask = (log_data < lower_bound) | (log_data > upper_bound)
        outliers_count = outlier_mask.sum()

        if outliers_count > 0:
            # 截断异常值
            log_data_clipped = np.clip(log_data, lower_bound, upper_bound)
            # 反变换回原始尺度
            data[feature] = np.expm1(log_data_clipped)
        else:
            # 即使没有异常值也保持对数变换后的结果
            data[feature] = np.expm1(log_data)

        # 计算最终偏度
        final_skewness = data[feature].skew()

        print(f"     {feature}: 原偏度={original_skewness:.3f} → 变换后={log_skewness:.3f} → 最终={final_skewness:.3f}")
        print(f"       对数变换 + {outliers_count}个异常值({outliers_count/len(data)*100:.1f}%)替换")

        return outliers_count
    
    def detect_outliers_iqr(self, data: pd.DataFrame, feature: str, 
                           multiplier: float = 1.5) -> Tuple[pd.Series, float, float]:
        """使用IQR方法检测异常值
        
        Args:
            data: 输入数据
            feature: 特征名
            multiplier: IQR倍数
            
        Returns:
            Tuple[pd.Series, float, float]: 异常值掩码、下界、上界
        """
        feature_data = data[feature].dropna()
        
        Q1 = feature_data.quantile(0.25)
        Q3 = feature_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)
        
        return outlier_mask, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, data: pd.DataFrame, feature: str, 
                              threshold: float = 3.0) -> pd.Series:
        """使用Z-score方法检测异常值
        
        Args:
            data: 输入数据
            feature: 特征名
            threshold: Z-score阈值
            
        Returns:
            pd.Series: 异常值掩码
        """
        feature_data = data[feature].dropna()
        
        z_scores = np.abs((feature_data - feature_data.mean()) / feature_data.std())
        outlier_indices = feature_data[z_scores > threshold].index
        
        outlier_mask = data.index.isin(outlier_indices)
        
        return outlier_mask
    
    def _validate_outlier_processing(self, original_data: pd.DataFrame, 
                                   processed_data: pd.DataFrame, 
                                   total_outliers: int) -> None:
        """验证异常值处理结果
        
        Args:
            original_data: 原始数据
            processed_data: 处理后数据
            total_outliers: 处理的异常值总数
        """
        # 检查数据形状
        if original_data.shape != processed_data.shape:
            raise OutlierHandlingError("异常值处理后数据形状发生变化")
        
        # 检查数据完整性
        validate_data_integrity(processed_data, "异常值处理后")
        
        print(f"\n   异常值处理总结:")
        print(f"     总共处理了 {total_outliers} 个异常值")
        
        # 计算处理前后的统计差异
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        
        print(f"     数据统计变化:")
        for col in numeric_cols:
            orig_mean = original_data[col].mean()
            proc_mean = processed_data[col].mean()
            mean_change = ((proc_mean - orig_mean) / orig_mean) * 100 if orig_mean != 0 else 0
            
            orig_std = original_data[col].std()
            proc_std = processed_data[col].std()
            std_change = ((proc_std - orig_std) / orig_std) * 100 if orig_std != 0 else 0
            
            if abs(mean_change) > 1 or abs(std_change) > 5:  # 只显示显著变化
                print(f"       {col}: 均值变化{mean_change:+.1f}%, 标准差变化{std_change:+.1f}%")
    
    def _save_processing_summary(self, processing_summary: Dict[str, Any]) -> None:
        """保存异常值处理摘要
        
        Args:
            processing_summary: 处理摘要字典
        """
        try:
            if not processing_summary:
                return
            
            # 创建摘要数据框
            summary_data = []
            for feature, info in processing_summary.items():
                summary_data.append([
                    feature,
                    info['category'],
                    f"{info['skewness']:.3f}",
                    info['outliers_removed']
                ])
            
            summary_df = pd.DataFrame(summary_data, 
                                    columns=['特征', '偏度类别', '偏度值', '处理异常值数量'])
            
            # 保存到文件
            output_path = os.path.join(self.config.results_dir, 'outlier_processing_summary.csv')
            summary_df.to_csv(output_path, index=False)
            
            print(f"     异常值处理摘要已保存: {output_path}")
            
        except Exception as e:
            print(f"     警告: 保存异常值处理摘要失败: {str(e)}")
    
    def get_outlier_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取异常值统计信息
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 异常值统计信息
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for col in numeric_cols:
            # IQR方法
            iqr_mask, iqr_lower, iqr_upper = self.detect_outliers_iqr(data, col)
            iqr_count = iqr_mask.sum()
            
            # Z-score方法
            zscore_mask = self.detect_outliers_zscore(data, col)
            zscore_count = zscore_mask.sum()
            
            outlier_stats[col] = {
                'iqr_outliers': int(iqr_count),
                'iqr_bounds': (float(iqr_lower), float(iqr_upper)),
                'zscore_outliers': int(zscore_count),
                'total_samples': len(data[col].dropna())
            }
        
        return outlier_stats
