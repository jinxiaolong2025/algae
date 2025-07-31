"""缺失值处理模块

负责处理数据中的缺失值，使用多种策略包括均值填充、回归填充、KNN填充等。
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from typing import Dict, Any
try:
    from .config import ProcessingConfig, MissingValueError
    from .utils import log_processing_step, validate_data_integrity
except ImportError:
    from config import ProcessingConfig, MissingValueError
    from utils import log_processing_step, validate_data_integrity


class MissingValueHandler:
    """缺失值处理器类"""
    
    def __init__(self, config: ProcessingConfig):
        """初始化缺失值处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值 - 针对性处理策略

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 处理后的数据
        """
        log_processing_step("3. 缺失值处理")

        print("\n 缺失值处理详情:")

        data_filled = data.copy()

        # 1. 特殊处理phosphate和TP的关系
        data_filled = self._handle_phosphate_tp_relationship(data_filled)

        # 2. 处理剩余的缺失值（N(%)和C(%)）
        data_filled = self._handle_remaining_missing_values(data_filled)

        # 验证处理结果
        self._validate_imputation_results(data, data_filled)

        return data_filled

    def _handle_phosphate_tp_relationship(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理phosphate和TP的关系 - 使用线性回归填充

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 处理后的数据
        """
        data_filled = data.copy()

        # 1. 分析phosphate和TP的相关性
        if 'phosphate' in data.columns and 'TP' in data.columns:
            # 获取两列都不为空的数据
            valid_mask = data['phosphate'].notna() & data['TP'].notna()
            if valid_mask.sum() > 0:
                correlation = data.loc[valid_mask, 'phosphate'].corr(data.loc[valid_mask, 'TP'])
                print(f"   phosphate与TP的相关系数: {correlation:.4f}")

                # 2. 使用TP对phosphate进行线性回归填充
                if data['phosphate'].isna().sum() > 0:
                    from sklearn.linear_model import LinearRegression

                    # 准备训练数据（phosphate和TP都不为空的样本）
                    train_mask = data['phosphate'].notna() & data['TP'].notna()
                    X_train = data.loc[train_mask, 'TP'].values.reshape(-1, 1)
                    y_train = data.loc[train_mask, 'phosphate'].values

                    # 训练线性回归模型
                    lr_model = LinearRegression()
                    lr_model.fit(X_train, y_train)

                    # 预测缺失值
                    missing_mask = data['phosphate'].isna() & data['TP'].notna()
                    if missing_mask.sum() > 0:
                        X_pred = data.loc[missing_mask, 'TP'].values.reshape(-1, 1)
                        y_pred = lr_model.predict(X_pred)

                        # 填充缺失值
                        data_filled.loc[missing_mask, 'phosphate'] = y_pred

                        print(f"    使用TP线性回归填充phosphate缺失值: {missing_mask.sum()}个")
                        print(f"   回归方程: phosphate = {lr_model.coef_[0]:.4f} * TP + {lr_model.intercept_:.4f}")
                        print(f"   R²得分: {lr_model.score(X_train, y_train):.4f}")

        return data_filled

    def _handle_remaining_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理剩余的缺失值 - 使用KNN填充

        Args:
            data: 输入数据

        Returns:
            pd.DataFrame: 处理后的数据
        """
        data_filled = data.copy()

        # 3. 处理N(%)和C(%)的缺失值 - 使用KNN填充
        remaining_missing = data_filled.isnull().sum()
        if remaining_missing.sum() > 0:
            print(f"\n   处理N(%)和C(%)缺失值:")

            # 只对N(%)和C(%)使用KNN填充
            missing_cols = remaining_missing[remaining_missing > 0].index.tolist()
            print(f"   需要处理的特征: {missing_cols}")

            if len(missing_cols) > 0:
                from sklearn.impute import KNNImputer

                # 使用KNN填充器，选择相关特征作为参考
                knn_imputer = KNNImputer(n_neighbors=5)

                # 选择相关特征进行KNN填充
                reference_cols = ['H(%)', 'O(%)', 'P(%)', 'protein(%)']
                available_refs = [col for col in reference_cols if col in data_filled.columns]

                # 构建用于KNN的特征集
                knn_features = missing_cols + available_refs
                knn_features = [col for col in knn_features if col in data_filled.columns]

                if len(knn_features) >= 2:  # 至少需要2个特征进行KNN
                    # 应用KNN填充
                    knn_data = data_filled[knn_features].copy()
                    knn_filled = knn_imputer.fit_transform(knn_data)

                    # 更新原数据
                    data_filled[knn_features] = knn_filled

                    print(f"   使用KNN(k=5)填充，参考特征: {available_refs}")
                    print(f"   填充完成: {len(missing_cols)}个特征的缺失值")
                else:
                    # 如果特征不足，使用均值填充
                    for col in missing_cols:
                        if data_filled[col].isnull().sum() > 0:
                            mean_val = data_filled[col].mean()
                            data_filled[col].fillna(mean_val, inplace=True)
                            print(f"   {col}: 使用均值填充 ({mean_val:.3f})")

        return data_filled
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析缺失值模式
        
        Args:
            data: 输入数据
            
        Returns:
            Dict[str, Any]: 缺失值分析结果
        """
        missing_counts = data.isnull().sum()
        missing_by_feature = missing_counts[missing_counts > 0].to_dict()
        total_missing = missing_counts.sum()
        
        return {
            'total_missing': int(total_missing),
            'missing_by_feature': missing_by_feature,
            'missing_percentage': (total_missing / (len(data) * len(data.columns))) * 100
        }
    
    def _select_imputation_strategy(self, data: pd.DataFrame, feature: str, missing_count: int) -> str:
        """为特征选择合适的填充策略
        
        Args:
            data: 输入数据
            feature: 特征名
            missing_count: 缺失值数量
            
        Returns:
            str: 选择的策略名称
        """
        total_samples = len(data)
        missing_ratio = missing_count / total_samples
        
        # 如果缺失比例过高，使用简单策略
        if missing_ratio > 0.5:
            return 'mean'
        
        # 检查是否为数值特征
        if not pd.api.types.is_numeric_dtype(data[feature]):
            return 'mode'
        
        # 对于数值特征，尝试找到高相关性特征进行回归填充
        numeric_features = data.select_dtypes(include=[np.number]).columns
        other_features = [f for f in numeric_features if f != feature]
        
        if len(other_features) > 0:
            # 计算与其他特征的相关性
            correlations = data[other_features].corrwith(data[feature]).abs()
            max_corr = correlations.max()
            
            if max_corr > self.config.correlation_threshold:
                return 'regression'
        
        # 如果样本数量足够且特征数量适中，使用KNN
        if total_samples > 20 and len(other_features) <= 10:
            return 'knn'
        
        # 默认使用均值填充
        return 'mean'
    
    def _apply_imputation_strategy(self, data: pd.DataFrame, feature: str, strategy: str) -> pd.DataFrame:
        """应用填充策略
        
        Args:
            data: 输入数据
            feature: 特征名
            strategy: 填充策略
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        data_copy = data.copy()
        missing_mask = data_copy[feature].isnull()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            return data_copy
        
        print(f"     {feature}: {missing_count}个缺失值 -> 使用{strategy}策略")
        
        try:
            if strategy == 'mean':
                fill_value = data_copy[feature].mean()
                data_copy[feature] = data_copy[feature].fillna(fill_value)
                print(f"       均值填充: {fill_value:.3f}")
                
            elif strategy == 'median':
                fill_value = data_copy[feature].median()
                data_copy[feature] = data_copy[feature].fillna(fill_value)
                print(f"       中位数填充: {fill_value:.3f}")
                
            elif strategy == 'mode':
                fill_value = data_copy[feature].mode().iloc[0] if not data_copy[feature].mode().empty else 0
                data_copy[feature] = data_copy[feature].fillna(fill_value)
                print(f"       众数填充: {fill_value}")
                
            elif strategy == 'regression':
                data_copy = self._regression_imputation(data_copy, feature)
                
            elif strategy == 'knn':
                data_copy = self._knn_imputation(data_copy, feature)
                
            else:
                raise MissingValueError(f"未知的填充策略: {strategy}")
                
        except Exception as e:
            print(f"       警告: {strategy}策略失败，使用均值填充: {str(e)}")
            fill_value = data_copy[feature].mean()
            data_copy[feature] = data_copy[feature].fillna(fill_value)
        
        return data_copy
    
    def _regression_imputation(self, data: pd.DataFrame, target_feature: str) -> pd.DataFrame:
        """回归填充
        
        Args:
            data: 输入数据
            target_feature: 目标特征
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        data_copy = data.copy()
        
        # 选择数值特征作为预测变量
        numeric_features = data_copy.select_dtypes(include=[np.number]).columns
        predictor_features = [f for f in numeric_features if f != target_feature]
        
        if len(predictor_features) == 0:
            raise ValueError("没有可用的预测特征")
        
        # 找到与目标特征相关性最高的特征
        correlations = data_copy[predictor_features].corrwith(data_copy[target_feature]).abs()
        best_predictors = correlations.nlargest(min(3, len(predictor_features))).index.tolist()
        
        # 准备训练数据（无缺失值的样本）
        complete_mask = data_copy[best_predictors + [target_feature]].notnull().all(axis=1)
        train_data = data_copy[complete_mask]
        
        if len(train_data) < 5:
            raise ValueError("训练数据不足")
        
        # 训练回归模型
        X_train = train_data[best_predictors]
        y_train = train_data[target_feature]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 预测缺失值
        missing_mask = data_copy[target_feature].isnull()
        X_missing = data_copy.loc[missing_mask, best_predictors]
        
        # 检查预测数据是否有缺失值
        if X_missing.isnull().any().any():
            raise ValueError("预测特征中存在缺失值")
        
        predicted_values = model.predict(X_missing)
        data_copy.loc[missing_mask, target_feature] = predicted_values
        
        print(f"       回归填充: 使用特征{best_predictors}, R²={model.score(X_train, y_train):.3f}")
        
        return data_copy
    
    def _knn_imputation(self, data: pd.DataFrame, target_feature: str) -> pd.DataFrame:
        """KNN填充
        
        Args:
            data: 输入数据
            target_feature: 目标特征
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        data_copy = data.copy()
        
        # 选择数值特征
        numeric_features = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) < 2:
            raise ValueError("数值特征不足，无法进行KNN填充")
        
        # 使用KNN填充器
        imputer = KNNImputer(n_neighbors=min(self.config.knn_neighbors, len(data_copy) - 1))
        
        # 只对数值特征进行KNN填充
        data_numeric = data_copy[numeric_features]
        data_imputed = imputer.fit_transform(data_numeric)
        
        # 更新数据
        data_copy[numeric_features] = data_imputed
        
        print(f"       KNN填充: K={min(self.config.knn_neighbors, len(data_copy) - 1)}")
        
        return data_copy
    
    def _validate_imputation_results(self, original_data: pd.DataFrame, filled_data: pd.DataFrame) -> None:
        """验证填充结果
        
        Args:
            original_data: 原始数据
            filled_data: 填充后数据
        """
        # 检查数据形状是否一致
        if original_data.shape != filled_data.shape:
            raise MissingValueError("填充后数据形状发生变化")
        
        # 检查是否还有缺失值
        remaining_missing = filled_data.isnull().sum().sum()
        original_missing = original_data.isnull().sum().sum()
        
        print(f"\n   缺失值处理结果:")
        print(f"     处理前: {original_missing} 个缺失值")
        print(f"     处理后: {remaining_missing} 个缺失值")
        
        if remaining_missing == 0:
            print("     ✓ 所有缺失值已成功处理")
        else:
            print(f"     仍有{remaining_missing}个缺失值未处理")
        
        # 验证数据完整性
        validate_data_integrity(filled_data, "缺失值处理后")
