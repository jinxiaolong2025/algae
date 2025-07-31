"""数据质量分析模块

负责分析数据质量，包括缺失值、异常值、偏度峰度等统计分析。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
try:
    from .config import ProcessingConfig
    from .utils import (
        calculate_skewness_kurtosis,
        interpret_skewness,
        interpret_kurtosis,
        print_section_header,
        format_percentage,
        log_processing_step
    )
except ImportError:
    from config import ProcessingConfig
    from utils import (
        calculate_skewness_kurtosis,
        interpret_skewness,
        interpret_kurtosis,
        print_section_header,
        format_percentage,
        log_processing_step
    )


class DataQualityAnalyzer:
    """数据质量分析器类"""
    
    def __init__(self, config: ProcessingConfig):
        """初始化数据质量分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """进行全面的数据质量分析
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 数据质量分析结果
        """
        log_processing_step("2. 进行数据质量分析")
        
        print_section_header("数据质量分析报告")
        
        # 基本信息分析
        basic_info = self._analyze_basic_info(data)
        
        # 数据类型分析
        dtype_info = self._analyze_data_types(data)
        
        # 缺失值分析
        missing_info = self._analyze_missing_values(data)
        
        # 零值分析
        zero_info = self._analyze_zero_values(data)
        
        # 偏度分析
        skewness_info = self._analyze_skewness(data)
        
        # 数值范围分析
        range_info = self._analyze_value_ranges(data)
        
        # 汇总分析结果
        quality_analysis = {
            'basic_info': basic_info,
            'dtype_info': dtype_info,
            'missing_info': missing_info,
            'zero_info': zero_info,
            'skewness_info': skewness_info,
            'range_info': range_info
        }
        
        # 保存分析结果
        self._save_quality_analysis(quality_analysis)
        
        # 打印总结
        self._print_quality_summary(quality_analysis)
        
        print_section_header("")
        
        return quality_analysis
    
    def _analyze_basic_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析基本信息"""
        info = {
            'shape': data.shape,
            'n_samples': len(data),
            'n_features': len(data.columns)
        }
        
        print(f"\n 数据基本信息:")
        print(f"   - 数据形状: {info['shape']}")
        print(f"   - 样本数量: {info['n_samples']}")
        print(f"   - 特征数量: {info['n_features']}")
        
        return info
    
    def _analyze_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析数据类型"""
        dtype_counts = data.dtypes.value_counts()
        
        print(f"\n 数据类型分布:")
        for dtype, count in dtype_counts.items():
            print(f"   - {dtype}: {count}个特征")
        
        return {'dtype_counts': dtype_counts.to_dict()}
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析缺失值"""
        missing_counts = data.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        total_missing = missing_counts.sum()
        
        print(f"\n 缺失值分析:")
        print(f"   - 总缺失值数量: {total_missing}")
        print(f"   - 有缺失值的特征数: {len(missing_features)}")
        
        if len(missing_features) > 0:
            print(f"   - 缺失值最多的特征:")
            # 按缺失值数量排序
            missing_sorted = missing_features.sort_values(ascending=False)
            for feature, count in missing_sorted.items():
                percentage = format_percentage(count, len(data))
                print(f"     * {feature}: {percentage}")
        
        return {
            'total_missing': int(total_missing),
            'missing_features': missing_features.to_dict(),
            'missing_percentage': (total_missing / (len(data) * len(data.columns))) * 100
        }
    
    def _analyze_zero_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析零值"""
        numeric_data = data.select_dtypes(include=[np.number])
        zero_counts = (numeric_data == 0).sum()
        zero_features = zero_counts[zero_counts > 0]
        total_zeros = zero_counts.sum()
        
        print(f"\n 零值分析:")
        print(f"   - 总零值数量: {total_zeros}")
        print(f"   - 有零值的特征数: {len(zero_features)}")
        
        if len(zero_features) > 0:
            print(f"   - 零值最多的特征:")
            zero_sorted = zero_features.sort_values(ascending=False)
            for feature, count in zero_sorted.items():
                percentage = format_percentage(count, len(data))
                print(f"     * {feature}: {percentage}")
        
        return {
            'total_zeros': int(total_zeros),
            'zero_features': zero_features.to_dict()
        }
    
    def _analyze_skewness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析偏度"""
        skewness_values, _, numeric_cols = calculate_skewness_kurtosis(data)
        
        # 统计不同偏度级别的特征数量
        severe_skew = sum(abs(skewness_values) > 2)
        moderate_skew = sum((abs(skewness_values) > 1) & (abs(skewness_values) <= 2))
        
        print(f"\n 偏度分析:")
        print(f"   - 高度偏斜特征 (|偏度| > 2): {severe_skew}个")
        print(f"   - 中度偏斜特征 (1 < |偏度| <= 2): {moderate_skew}个")
        print(f"   - 详细偏度和峰度分析将在缺失值填充后进行")
        
        return {
            'skewness_values': skewness_values.to_dict(),
            'severe_skew_count': severe_skew,
            'moderate_skew_count': moderate_skew
        }
    
    def _analyze_value_ranges(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析数值范围"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        print(f"\n 数值范围分析:")
        print(f"{'特征名称':<30} {'最小值':<10} {'最大值':<10} {'均值':<10} {'标准差':<10}")
        print("-" * 70)
        
        range_info = {}
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                print(f"{col:<30} {min_val:<10.2f} {max_val:<10.2f} {mean_val:<10.2f} {std_val:<10.2f}")
                
                range_info[col] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val,
                    'range': max_val - min_val
                }
        
        return range_info
    
    def _save_quality_analysis(self, quality_analysis: Dict[str, Any]) -> None:
        """保存数据质量分析结果"""
        try:
            # 创建汇总数据框
            summary_data = []
            
            # 基本信息
            basic = quality_analysis['basic_info']
            summary_data.append(['数据形状', f"{basic['shape']}"])
            summary_data.append(['样本数量', basic['n_samples']])
            summary_data.append(['特征数量', basic['n_features']])
            
            # 缺失值信息
            missing = quality_analysis['missing_info']
            summary_data.append(['总缺失值', missing['total_missing']])
            summary_data.append(['缺失值百分比', f"{missing['missing_percentage']:.2f}%"])
            
            # 零值信息
            zero = quality_analysis['zero_info']
            summary_data.append(['总零值', zero['total_zeros']])
            
            # 偏度信息
            skew = quality_analysis['skewness_info']
            summary_data.append(['高度偏斜特征数', skew['severe_skew_count']])
            summary_data.append(['中度偏斜特征数', skew['moderate_skew_count']])
            
            # 保存到CSV
            summary_df = pd.DataFrame(summary_data, columns=['指标', '值'])
            output_path = self.config.get_output_paths()['quality_analysis']
            summary_df.to_csv(output_path, index=False)
            
            print(f"\n 数据质量分析报告已保存:")
            print(f"   - {output_path}")
            
        except Exception as e:
            print(f"   警告: 保存数据质量分析失败: {str(e)}")
    
    def _print_quality_summary(self, quality_analysis: Dict[str, Any]) -> None:
        """打印数据质量总结"""
        missing_info = quality_analysis['missing_info']
        skew_info = quality_analysis['skewness_info']
        
        print(f"\n 数据质量总结:")
        
        if missing_info['total_missing'] > 0:
            print(f"     发现{missing_info['total_missing']}个缺失值，建议先对其进行填充处理")
        
        if skew_info['severe_skew_count'] > 0:
            print(f"     {skew_info['severe_skew_count']}个特征高度偏斜，建议考虑对数变换或其他标准化方法")
