"""数据加载和预处理模块

包含数据加载器和数据预处理器类
"""
import pandas as pd
import os
from typing import List, Tuple

try:
    from .config import TrainingConfig, DataLoadError
except ImportError:
    from config import TrainingConfig, DataLoadError


class DataLoader:
    """数据加载器类，负责加载训练数据和特征选择结果"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def load_training_data(self) -> Tuple[pd.DataFrame, List[str], str]:
        """
        加载训练数据和特征选择结果
        
        Returns:
            Tuple[pd.DataFrame, List[str], str]: (训练数据, 选择的特征列表, 数据来源描述)
        """
        try:
            # 加载训练数据
            train_data = self._load_train_data()
            
            # 加载特征选择结果
            selected_features, data_source = self._load_selected_features(train_data)
            
            return train_data, selected_features, data_source
            
        except Exception as e:
            raise DataLoadError(f"数据加载失败: {str(e)}")
    
    def _load_train_data(self) -> pd.DataFrame:
        """加载训练数据"""
        if not os.path.exists(self.config.train_data_path):
            raise DataLoadError(f"训练数据文件不存在: {self.config.train_data_path}")
        
        train_data = pd.read_csv(self.config.train_data_path)
        print(f"   训练数据加载成功: {train_data.shape}")
        return train_data
    
    def _load_selected_features(self, train_data: pd.DataFrame) -> Tuple[List[str], str]:
        """加载特征选择结果"""
        try:
            if not os.path.exists(self.config.ga_features_path):
                print(f"   警告: 特征选择文件不存在: {self.config.ga_features_path}")
                print("   将使用所有特征进行训练")
                all_features = [col for col in train_data.columns if col != 'lipid(%)']
                return all_features, "all_features"
            
            ga_features_df = pd.read_csv(self.config.ga_features_path)
            if 'feature_name' in ga_features_df.columns:
                selected_features = ga_features_df['feature_name'].tolist()
            elif 'feature' in ga_features_df.columns:
                selected_features = ga_features_df['feature'].tolist()
            else:
                print("   警告: 特征选择文件格式不正确，缺少'feature'或'feature_name'列")
                print("   将使用所有特征进行训练")
                all_features = [col for col in train_data.columns if col != 'lipid(%)']
                return all_features, "all_features"
            print(f"   GA特征选择结果加载成功: {len(selected_features)} 个特征")
            return selected_features, "ga_selected"
            
        except Exception as e:
            print(f"   警告: 加载特征选择结果时发生错误: {str(e)}")
            print("   将使用所有特征进行训练")
            all_features = [col for col in train_data.columns if col != 'lipid(%)']
            return all_features, "all_features"


class DataPreprocessor:
    """数据预处理器类，负责准备训练数据"""
    
    @staticmethod
    def prepare_training_data(train_data: pd.DataFrame,
                            selected_features: List[str],
                            data_source: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        准备训练数据
        
        Args:
            train_data: 原始训练数据
            selected_features: 选择的特征列表
            data_source: 数据来源描述
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, List[str]]: (特征数据, 目标变量, 最终特征列表)
        """
        # 检查特征是否存在
        available_features = [col for col in train_data.columns if col != 'lipid(%)']
        missing_features = [f for f in selected_features if f not in available_features]
        
        if missing_features:
            print(f"   警告: 以下特征在训练数据中不存在: {missing_features}")
            selected_features = [f for f in selected_features if f in available_features]
            print(f"   使用可用特征: {len(selected_features)} 个")
        
        if not selected_features:
            raise DataLoadError("没有可用的特征进行训练")
        
        # 准备特征和目标变量
        X_train = train_data[selected_features].copy()
        y_train = train_data['lipid(%)'].copy()
        
        # 检查数据完整性
        if X_train.isnull().any().any():
            print("   警告: 特征数据中存在缺失值")
        if y_train.isnull().any():
            print("   警告: 目标变量中存在缺失值")
        
        # 显示数据信息
        print(f"   最终特征数量: {len(selected_features)}")
        print(f"   训练样本数量: {len(X_train)}")
        print(f"   目标变量范围: {y_train.min():.3f} - {y_train.max():.3f}")
        print(f"   数据来源: {data_source}")
        
        return X_train, y_train, selected_features
