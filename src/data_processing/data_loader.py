"""数据加载模块

负责从原始数据文件加载数据，并进行基本的数据清理。
"""

import pandas as pd
import os
from typing import Tuple
try:
    from .config import ProcessingConfig, DataLoadError
    from .utils import log_processing_step, validate_data_integrity
except ImportError:
    # 处理直接运行时的导入
    from config import ProcessingConfig, DataLoadError
    from utils import log_processing_step, validate_data_integrity


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, config: ProcessingConfig):
        """初始化数据加载器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def load_raw_data(self) -> pd.DataFrame:
        """加载原始数据
        
        Returns:
            pd.DataFrame: 加载的原始数据
            
        Raises:
            DataLoadError: 数据加载失败时抛出
        """
        log_processing_step("1. 加载原始数据")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(self.config.raw_data_path):
                raise DataLoadError(f"原始数据文件不存在: {self.config.raw_data_path}")
            
            # 加载数据
            data = pd.read_excel(self.config.raw_data_path)
            
            # 删除配置中指定的排除列
            if self.config.exclude_columns:
                existing_exclude_cols = [col for col in self.config.exclude_columns if col in data.columns]
                if existing_exclude_cols:
                    data = data.drop(existing_exclude_cols, axis=1)
                    print(f"   已删除列: {existing_exclude_cols}")
            
            # 验证数据完整性
            validate_data_integrity(data, "原始数据加载")
            
            print(f"   原始数据形状: {data.shape}")
            
            return data
            
        except Exception as e:
            raise DataLoadError(f"加载原始数据失败: {str(e)}")
    
    def save_processed_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                           full_data: pd.DataFrame) -> None:
        """保存处理后的数据
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            full_data: 完整数据
        """
        log_processing_step("9. 保存处理后的数据")
        
        try:
            # 获取输出路径
            output_paths = self.config.get_output_paths()
            
            # 保存数据
            train_data.to_csv(output_paths['train_data'], index=False)
            test_data.to_csv(output_paths['test_data'], index=False)
            full_data.to_csv(output_paths['processed_data'], index=False)
            
            print("   数据已保存到:")
            print(f"     - {output_paths['train_data']}")
            print(f"     - {output_paths['test_data']}")
            print(f"     - {output_paths['processed_data']}")
            
        except Exception as e:
            raise DataLoadError(f"保存处理后的数据失败: {str(e)}")
    
    def split_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分割数据集为训练集和测试集
        
        Args:
            data: 输入数据
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 训练集、测试集
        """
        log_processing_step("7. 分割数据集")
        
        try:
            from sklearn.model_selection import train_test_split
            
            train_data, test_data = train_test_split(
                data, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
            
            print(f"   训练集大小: {train_data.shape}")
            print(f"   测试集大小: {test_data.shape}")
            
            return train_data, test_data
            
        except Exception as e:
            raise DataLoadError(f"数据集分割失败: {str(e)}")


class DataValidator:
    """数据验证器类"""
    
    def __init__(self, config: ProcessingConfig):
        """初始化数据验证器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def validate_target_column(self, data: pd.DataFrame) -> None:
        """验证目标列是否存在
        
        Args:
            data: 输入数据
            
        Raises:
            DataLoadError: 目标列不存在时抛出
        """
        if self.config.target_column not in data.columns:
            raise DataLoadError(f"目标列 '{self.config.target_column}' 不存在于数据中")
    
    def validate_data_types(self, data: pd.DataFrame) -> None:
        """验证数据类型
        
        Args:
            data: 输入数据
        """
        # 检查数值列
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise DataLoadError("数据中没有数值列")
        
        print(f"   发现 {len(numeric_cols)} 个数值特征")
    
    def validate_data_range(self, data: pd.DataFrame) -> None:
        """验证数据范围的合理性
        
        Args:
            data: 输入数据
        """
        # 检查是否有无穷大值
        if data.select_dtypes(include=['number']).isin([float('inf'), float('-inf')]).any().any():
            raise DataLoadError("数据中包含无穷大值")
        
        # 检查目标变量是否为负值
        if self.config.target_column in data.columns:
            target_data = data[self.config.target_column].dropna()
            if (target_data < 0).any():
                print(f"   警告: 目标变量 '{self.config.target_column}' 包含负值")
    
    def get_basic_info(self, data: pd.DataFrame) -> dict:
        """获取数据基本信息
        
        Args:
            data: 输入数据
            
        Returns:
            dict: 基本信息字典
        """
        info = {
            'shape': data.shape,
            'n_samples': len(data),
            'n_features': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'dtypes': data.dtypes.value_counts().to_dict()
        }
        
        return info
