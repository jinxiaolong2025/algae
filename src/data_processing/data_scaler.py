"""数据标准化模块

负责数据的标准化和归一化处理，包括偏度峰度分析。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
try:
    from .config import ProcessingConfig
    from .utils import (
        calculate_skewness_kurtosis,
        interpret_skewness,
        interpret_kurtosis,
        log_processing_step,
        validate_data_integrity
    )
except ImportError:
    from config import ProcessingConfig
    from utils import (
        calculate_skewness_kurtosis,
        interpret_skewness,
        interpret_kurtosis,
        log_processing_step,
        validate_data_integrity
    )


class DataScaler:
    """数据标准化器类"""
    
    def __init__(self, config: ProcessingConfig):
        """初始化数据标准化器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def analyze_and_scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """分析数据分布并进行标准化

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 标准化后的数据
        """
        log_processing_step("5. 数据分布分析和标准化")

        # 进行详细的偏度峰度分析
        skewness_analysis = self.analyze_skewness_kurtosis(data)

        # 保存分析结果
        self._save_skewness_analysis(skewness_analysis)

        # 目前不进行标准化，保持原始数据
        print("   保持原始数据尺度，不进行标准化处理")

        return data.copy()

    def apply_robust_scaling(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                           full_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """应用Robust标准化

        Args:
            train_data: 训练数据
            test_data: 测试数据
            full_data: 完整数据

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 标准化后的训练集、测试集、完整数据
        """
        log_processing_step("8. 进行Robust标准化")

        from sklearn.preprocessing import RobustScaler

        # 获取数值特征（排除目标变量）
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != self.config.target_column]

        # 创建scaler并在训练集上拟合
        scaler = RobustScaler()
        print("   - 在训练集上拟合scaler")

        # 拟合并变换训练集
        train_scaled = train_data.copy()
        train_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])

        # 变换测试集
        test_scaled = test_data.copy()
        test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])

        # 变换完整数据集
        full_scaled = full_data.copy()
        full_scaled[feature_cols] = scaler.transform(full_data[feature_cols])

        print("   - 将scaler应用到测试集")
        print(f"   标准化后数据形状: 训练集{train_scaled.shape}, 测试集{test_scaled.shape}")
        print(f"   完整数据集形状: {full_scaled.shape}")

        # 验证数据完整性
        if full_scaled.isnull().sum().sum() == 0:
            print("    数据完整，无缺失值")
        else:
            print(f"    警告: 标准化后仍有{full_scaled.isnull().sum().sum()}个缺失值")

        return train_scaled, test_scaled, full_scaled
    
    def analyze_skewness_kurtosis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """详细分析偏度和峰度
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 偏度峰度分析结果
        """
        print("\n   详细偏度和峰度分析:")
        
        # 计算偏度和峰度
        skewness_values, kurtosis_values, numeric_cols = calculate_skewness_kurtosis(data)
        
        # 创建分析结果
        analysis_results = []
        
        print(f"\n   {'特征名称':<30} {'偏度':<10} {'偏度解释':<15} {'峰度':<10} {'峰度解释':<15}")
        print("-" * 90)
        
        for col in numeric_cols:
            skew_val = skewness_values[col]
            kurt_val = kurtosis_values[col]
            
            skew_interp = interpret_skewness(skew_val)
            kurt_interp = interpret_kurtosis(kurt_val)
            
            print(f"   {col:<30} {skew_val:<10.3f} {skew_interp:<15} {kurt_val:<10.3f} {kurt_interp:<15}")
            
            analysis_results.append({
                'feature': col,
                'skewness': skew_val,
                'skewness_interpretation': skew_interp,
                'kurtosis': kurt_val,
                'kurtosis_interpretation': kurt_interp
            })
        
        # 统计分析
        self._print_distribution_summary(skewness_values, kurtosis_values)
        
        return {
            'analysis_results': analysis_results,
            'skewness_values': skewness_values.to_dict(),
            'kurtosis_values': kurtosis_values.to_dict(),
            'summary_stats': self._calculate_summary_stats(skewness_values, kurtosis_values)
        }
    
    def _print_distribution_summary(self, skewness_values: pd.Series, kurtosis_values: pd.Series) -> None:
        """打印分布统计摘要
        
        Args:
            skewness_values: 偏度值
            kurtosis_values: 峰度值
        """
        print(f"\n   分布特征统计:")
        
        # 偏度统计
        severe_right_skew = sum(skewness_values > 2)
        moderate_right_skew = sum((skewness_values > 1) & (skewness_values <= 2))
        light_right_skew = sum((skewness_values > 0.5) & (skewness_values <= 1))
        symmetric = sum((skewness_values >= -0.5) & (skewness_values <= 0.5))
        light_left_skew = sum((skewness_values >= -1) & (skewness_values < -0.5))
        moderate_left_skew = sum((skewness_values >= -2) & (skewness_values < -1))
        severe_left_skew = sum(skewness_values < -2)
        
        print(f"     偏度分布:")
        print(f"       严重右偏 (>2): {severe_right_skew}个")
        print(f"       中度右偏 (1-2): {moderate_right_skew}个")
        print(f"       轻度右偏 (0.5-1): {light_right_skew}个")
        print(f"       近似对称 (-0.5-0.5): {symmetric}个")
        print(f"       轻度左偏 (-1--0.5): {light_left_skew}个")
        print(f"       中度左偏 (-2--1): {moderate_left_skew}个")
        print(f"       严重左偏 (<-2): {severe_left_skew}个")
        
        # 峰度统计
        high_peak = sum(kurtosis_values > 3)
        normal_peak = sum((kurtosis_values >= 0) & (kurtosis_values <= 3))
        low_peak = sum(kurtosis_values < 0)
        
        print(f"     峰度分布:")
        print(f"       高峰态 (>3): {high_peak}个")
        print(f"       中峰态 (0-3): {normal_peak}个")
        print(f"       低峰态 (<0): {low_peak}个")
    
    def _calculate_summary_stats(self, skewness_values: pd.Series, kurtosis_values: pd.Series) -> Dict[str, Any]:
        """计算摘要统计
        
        Args:
            skewness_values: 偏度值
            kurtosis_values: 峰度值
            
        Returns:
            Dict[str, Any]: 摘要统计
        """
        return {
            'skewness_stats': {
                'mean': float(skewness_values.mean()),
                'std': float(skewness_values.std()),
                'min': float(skewness_values.min()),
                'max': float(skewness_values.max()),
                'median': float(skewness_values.median())
            },
            'kurtosis_stats': {
                'mean': float(kurtosis_values.mean()),
                'std': float(kurtosis_values.std()),
                'min': float(kurtosis_values.min()),
                'max': float(kurtosis_values.max()),
                'median': float(kurtosis_values.median())
            }
        }
    
    def _save_skewness_analysis(self, analysis: Dict[str, Any]) -> None:
        """保存偏度峰度分析结果
        
        Args:
            analysis: 分析结果
        """
        try:
            # 创建详细分析数据框
            analysis_df = pd.DataFrame(analysis['analysis_results'])
            
            # 保存到CSV
            output_path = self.config.get_output_paths()['skewness_analysis']
            analysis_df.to_csv(output_path, index=False)
            
            print(f"\n   偏度峰度分析结果已保存:")
            print(f"     - {output_path}")
            
        except Exception as e:
            print(f"   警告: 保存偏度峰度分析失败: {str(e)}")
    
    def standardize_features(self, data: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """标准化特征（可选功能）
        
        Args:
            data: 输入数据
            method: 标准化方法 ('zscore', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        data_scaled = data.copy()
        numeric_cols = data_scaled.select_dtypes(include=[np.number]).columns
        
        if method == 'zscore':
            # Z-score标准化
            for col in numeric_cols:
                if col != self.config.target_column:
                    mean_val = data_scaled[col].mean()
                    std_val = data_scaled[col].std()
                    if std_val != 0:
                        data_scaled[col] = (data_scaled[col] - mean_val) / std_val
        
        elif method == 'minmax':
            # Min-Max标准化
            for col in numeric_cols:
                if col != self.config.target_column:
                    min_val = data_scaled[col].min()
                    max_val = data_scaled[col].max()
                    if max_val != min_val:
                        data_scaled[col] = (data_scaled[col] - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            # 鲁棒标准化
            for col in numeric_cols:
                if col != self.config.target_column:
                    median_val = data_scaled[col].median()
                    mad_val = (data_scaled[col] - median_val).abs().median()
                    if mad_val != 0:
                        data_scaled[col] = (data_scaled[col] - median_val) / mad_val
        
        print(f"   使用{method}方法标准化了{len(numeric_cols)}个数值特征")
        
        return data_scaled
    
    def apply_log_transformation(self, data: pd.DataFrame, features: list = None) -> pd.DataFrame:
        """应用对数变换（用于处理偏斜数据）
        
        Args:
            data: 输入数据
            features: 要变换的特征列表，如果为None则自动选择高偏斜特征
            
        Returns:
            pd.DataFrame: 变换后的数据
        """
        data_transformed = data.copy()
        
        if features is None:
            # 自动选择高偏斜特征
            skewness_values, _, numeric_cols = calculate_skewness_kurtosis(data)
            features = [col for col in numeric_cols 
                       if abs(skewness_values[col]) > 1 and col != self.config.target_column]
        
        transformed_count = 0
        for feature in features:
            if feature in data_transformed.columns:
                # 确保所有值为正数（对数变换要求）
                min_val = data_transformed[feature].min()
                if min_val <= 0:
                    # 添加常数使所有值为正
                    data_transformed[feature] = data_transformed[feature] - min_val + 1
                
                # 应用对数变换
                data_transformed[feature] = np.log1p(data_transformed[feature])
                transformed_count += 1
        
        if transformed_count > 0:
            print(f"   对{transformed_count}个高偏斜特征应用了对数变换")
        
        return data_transformed
