"""
数据处理模块

该模块负责数据的加载、预处理和验证。
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split

try:
    from .config import FeatureSelectionConfig, DataLoadError
    from .utils import validate_input_data, print_section_header
except ImportError:
    from config import FeatureSelectionConfig, DataLoadError
    from utils import validate_input_data, print_section_header


class DataHandler:
    """数据处理器"""
    
    def __init__(self, config: FeatureSelectionConfig):
        """
        初始化数据处理器
        
        Args:
            config: 特征选择配置对象
        """
        self.config = config
        
        # 存储数据
        self.raw_data = None
        self.feature_data = None
        self.target_data = None
        self.feature_names = None
        
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载训练数据
        
        Returns:
            特征数据、目标数据、特征名称的元组
        """
        print_section_header("加载训练数据")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(self.config.train_data_path):
                raise DataLoadError(f"训练数据文件不存在: {self.config.train_data_path}")
            
            # 加载数据
            print(f"从文件加载数据: {self.config.train_data_path}")
            self.raw_data = pd.read_csv(self.config.train_data_path)
            
            print(f"数据形状: {self.raw_data.shape}")
            print(f"列名: {list(self.raw_data.columns)}")
            
            # 检查目标列是否存在
            if self.config.target_column not in self.raw_data.columns:
                raise DataLoadError(f"目标列 '{self.config.target_column}' 不存在于数据中")
            
            # 分离特征和目标变量
            feature_columns = [col for col in self.raw_data.columns 
                             if col != self.config.target_column]
            
            self.feature_data = self.raw_data[feature_columns].values
            self.target_data = self.raw_data[self.config.target_column].values
            self.feature_names = np.array(feature_columns)
            
            print(f"特征数量: {len(feature_columns)}")
            print(f"样本数量: {len(self.target_data)}")
            print(f"目标变量: {self.config.target_column}")
            
            # 验证数据
            self.validate_loaded_data()
            
            print("数据加载完成")
            
            return self.feature_data, self.target_data, self.feature_names
            
        except Exception as e:
            raise DataLoadError(f"加载训练数据失败: {str(e)}")
    
    def validate_loaded_data(self):
        """验证加载的数据"""
        if self.feature_data is None or self.target_data is None:
            raise DataLoadError("数据未正确加载")
        
        # 使用工具函数验证数据
        validate_input_data(self.feature_data, self.target_data, self.feature_names)
        
        # 检查数据范围
        if np.any(np.isinf(self.feature_data)) or np.any(np.isinf(self.target_data)):
            raise DataLoadError("数据包含无穷大值")
        
        # 检查特征方差
        feature_vars = np.var(self.feature_data, axis=0)
        zero_var_features = np.where(feature_vars == 0)[0]
        if len(zero_var_features) > 0:
            zero_var_names = self.feature_names[zero_var_features]
            print(f"警告: 发现{len(zero_var_features)}个零方差特征: {zero_var_names}")
        
        print("数据验证通过")
    
    def get_data_summary(self) -> dict:
        """
        获取数据摘要信息
        
        Returns:
            数据摘要字典
        """
        if self.feature_data is None:
            raise DataLoadError("请先加载数据")
        
        summary = {
            'n_samples': self.feature_data.shape[0],
            'n_features': self.feature_data.shape[1],
            'target_column': self.config.target_column,
            'feature_names': self.feature_names.tolist(),
            'target_stats': {
                'mean': float(np.mean(self.target_data)),
                'std': float(np.std(self.target_data)),
                'min': float(np.min(self.target_data)),
                'max': float(np.max(self.target_data)),
                'median': float(np.median(self.target_data))
            },
            'feature_stats': {
                'means': np.mean(self.feature_data, axis=0).tolist(),
                'stds': np.std(self.feature_data, axis=0).tolist(),
                'mins': np.min(self.feature_data, axis=0).tolist(),
                'maxs': np.max(self.feature_data, axis=0).tolist()
            }
        }
        
        return summary
    
    def split_data(self, test_size: float = 0.2, 
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        分割数据为训练集和测试集
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.feature_data is None:
            raise DataLoadError("请先加载数据")
        
        if random_state is None:
            random_state = self.config.random_state
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.feature_data, 
            self.target_data,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # 回归问题不需要分层
        )
        
        print(f"数据分割完成:")
        print(f"  训练集: {X_train.shape}")
        print(f"  测试集: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_subset(self, feature_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取特征子集
        
        Args:
            feature_indices: 特征索引数组
            
        Returns:
            选择的特征数据和特征名称
        """
        if self.feature_data is None:
            raise DataLoadError("请先加载数据")
        
        if len(feature_indices) == 0:
            raise ValueError("特征索引不能为空")
        
        if np.any(feature_indices < 0) or np.any(feature_indices >= self.feature_data.shape[1]):
            raise ValueError("特征索引超出范围")
        
        selected_features = self.feature_data[:, feature_indices]
        selected_names = self.feature_names[feature_indices]
        
        return selected_features, selected_names
    
    def create_feature_dataframe(self, feature_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        创建特征DataFrame
        
        Args:
            feature_indices: 可选的特征索引，如果为None则使用所有特征
            
        Returns:
            特征DataFrame
        """
        if self.feature_data is None:
            raise DataLoadError("请先加载数据")
        
        if feature_indices is None:
            # 使用所有特征
            data = self.feature_data
            names = self.feature_names
        else:
            # 使用选择的特征
            data, names = self.get_feature_subset(feature_indices)
        
        df = pd.DataFrame(data, columns=names)
        
        # 添加目标变量
        df[self.config.target_column] = self.target_data
        
        return df
    
    def save_selected_data(self, feature_indices: np.ndarray, 
                          output_path: str):
        """
        保存选择的特征数据
        
        Args:
            feature_indices: 选择的特征索引
            output_path: 输出文件路径
        """
        if self.feature_data is None:
            raise DataLoadError("请先加载数据")
        
        # 创建包含选择特征的DataFrame
        df = self.create_feature_dataframe(feature_indices)
        
        # 保存到文件
        df.to_csv(output_path, index=False)
        print(f"选择的特征数据已保存到: {output_path}")
    
    def get_feature_statistics(self, feature_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        获取特征统计信息
        
        Args:
            feature_indices: 可选的特征索引
            
        Returns:
            特征统计DataFrame
        """
        if self.feature_data is None:
            raise DataLoadError("请先加载数据")
        
        if feature_indices is None:
            data = self.feature_data
            names = self.feature_names
        else:
            data, names = self.get_feature_subset(feature_indices)
        
        stats_df = pd.DataFrame({
            'feature_name': names,
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q25': np.percentile(data, 25, axis=0),
            'q75': np.percentile(data, 75, axis=0),
            'skewness': pd.DataFrame(data, columns=names).skew(),
            'kurtosis': pd.DataFrame(data, columns=names).kurtosis()
        })
        
        return stats_df
    
    def check_data_quality(self) -> dict:
        """
        检查数据质量
        
        Returns:
            数据质量报告字典
        """
        if self.feature_data is None:
            raise DataLoadError("请先加载数据")
        
        quality_report = {
            'missing_values': {
                'total': int(np.sum(np.isnan(self.feature_data))),
                'by_feature': {}
            },
            'infinite_values': {
                'total': int(np.sum(np.isinf(self.feature_data))),
                'by_feature': {}
            },
            'zero_variance_features': [],
            'highly_correlated_features': [],
            'outliers': {
                'total': 0,
                'by_feature': {}
            }
        }
        
        # 检查每个特征的缺失值和无穷值
        for i, feature_name in enumerate(self.feature_names):
            feature_data = self.feature_data[:, i]
            
            missing_count = int(np.sum(np.isnan(feature_data)))
            infinite_count = int(np.sum(np.isinf(feature_data)))
            
            if missing_count > 0:
                quality_report['missing_values']['by_feature'][feature_name] = missing_count
            
            if infinite_count > 0:
                quality_report['infinite_values']['by_feature'][feature_name] = infinite_count
            
            # 检查零方差
            if np.var(feature_data) == 0:
                quality_report['zero_variance_features'].append(feature_name)
            
            # 检查异常值（使用IQR方法）
            q1 = np.percentile(feature_data, 25)
            q3 = np.percentile(feature_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.sum((feature_data < lower_bound) | (feature_data > upper_bound))
            if outliers > 0:
                quality_report['outliers']['by_feature'][feature_name] = int(outliers)
                quality_report['outliers']['total'] += outliers
        
        return quality_report
    
    def print_data_info(self):
        """打印数据信息"""
        if self.feature_data is None:
            print("数据未加载")
            return
        
        print("数据信息:")
        print("=" * 50)
        print(f"样本数量: {self.feature_data.shape[0]}")
        print(f"特征数量: {self.feature_data.shape[1]}")
        print(f"目标变量: {self.config.target_column}")
        
        print(f"\n目标变量统计:")
        print(f"  均值: {np.mean(self.target_data):.4f}")
        print(f"  标准差: {np.std(self.target_data):.4f}")
        print(f"  范围: [{np.min(self.target_data):.4f}, {np.max(self.target_data):.4f}]")
        
        print(f"\n特征列表:")
        for i, name in enumerate(self.feature_names):
            print(f"  {i+1:2d}. {name}")
        
        print("=" * 50)
