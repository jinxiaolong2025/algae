"""模型训练模块

包含模型训练器类，负责训练和评估多种机器学习模型
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from .config import TrainingConfig
    from .model_factory import ModelFactory
    from .data_augmentor import DataAugmentor
except ImportError:
    from config import TrainingConfig
    from model_factory import ModelFactory
    from data_augmentor import DataAugmentor


class ModelTrainer:
    """模型训练器类，负责训练和评估多种机器学习模型"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_factory = ModelFactory(config)
        self.data_augmentor = DataAugmentor(config) if config.use_augmentation else None
    
    def _calculate_metrics(self, y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict[str, float]: 包含r2, mae, rmse的指标字典
        """
        return {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def train_and_compare_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                               feature_names: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        训练并对比多种模型，使用正确的交叉验证和数据增强

        训练流程说明：
        1. 交叉验证：每个fold内独立进行数据增强，避免数据泄露
        2. 最终模型：根据配置在增强数据或原始数据上训练
        3. 训练指标：始终在原始训练数据上计算，以反映对真实数据的拟合程度

        Args:
            X_train: 训练特征数据
            y_train: 训练目标变量
            feature_names: 特征名称列表

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: (训练结果, 训练好的模型)
            
        Note:
            训练结果包含交叉验证指标(r2, mae, rmse及其标准差)和训练集指标(train_r2, train_mae, train_rmse)
        """
        print("   开始多模型训练和对比...")
        
        # 创建模型套件
        models = self.model_factory.create_model_suite()
        results = {}
        trained_models = {}
        
        # 设置交叉验证
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True,
                     random_state=self.config.cv_random_state)
        
        # 训练每个模型
        for model_name, model_template in models.items():
            print(f"     训练 {model_name}...")
            try:
                # 交叉验证
                if self.config.use_augmentation:
                    # 使用数据增强的交叉验证
                    cv_results = self._cross_validate_with_augmentation(
                        model_template, X_train, y_train, feature_names, kfold, model_name
                    )
                else:
                    # 标准交叉验证
                    cv_results = self._cross_validate_standard(
                        model_template, X_train, y_train, kfold, model_name
                    )
                
                # 训练最终模型
                final_model = self._train_final_model(
                    model_template, X_train, y_train, feature_names, model_name
                )

                # 添加最终模型的交叉验证评估
                final_cv_metrics = self._evaluate_final_model(
                    final_model, X_train, y_train, feature_names, model_name
                )

                # 存储结果时包含三种评估
                results[model_name] = {
                    **cv_results,          # 原始交叉验证结果
                    **final_cv_metrics,    # 最终模型的交叉验证指标
                    'model': final_model
                }

                trained_models[model_name] = final_model
                
                # 只对XGBoost显示详细的评估结果对比
                if model_name.lower() == "xgboost":
                    print(f"       原始交叉验证 R²: {cv_results['r2']:.4f} (±{cv_results['r2_std']:.4f})")
                    print(f"       原始交叉验证 MAE: {cv_results['mae']:.3f} (±{cv_results['mae_std']:.3f})")
                    print(f"       原始交叉验证 RMSE: {cv_results['rmse']:.3f} (±{cv_results['rmse_std']:.3f})")
                    print(f"       最终模型评估 R²: {final_cv_metrics['final_cv_r2']:.4f} (±{final_cv_metrics['final_cv_r2_std']:.4f})")
                    print(f"       最终模型评估 MAE: {final_cv_metrics['final_cv_mae']:.3f} (±{final_cv_metrics['final_cv_mae_std']:.3f})")
                    print(f"       最终模型评估 RMSE: {final_cv_metrics['final_cv_rmse']:.3f} (±{final_cv_metrics['final_cv_rmse_std']:.3f})")
                else:
                    # 其他模型只显示简单的交叉验证结果
                    print(f"       交叉验证 R²: {cv_results['r2']:.4f} (±{cv_results['r2_std']:.4f})")
                
            except Exception as e:
                print(f"       {model_name} 训练失败: {str(e)}")
                results[model_name] = self._create_failed_result(str(e))
        
        return results, trained_models

    def _create_failed_result(self, error_message: str) -> Dict[str, Any]:
        """创建失败结果的字典"""
        return {
            'r2': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'r2_std': np.nan,
            'mae_std': np.nan,
            'rmse_std': np.nan,
            'model': None,
            'error': error_message
        }

    def _cross_validate_base(self, model_template: Any, X_train: pd.DataFrame,
                           y_train: pd.Series, feature_names: List[str],
                           kfold: KFold, model_name: str = "",
                           use_augmentation: bool = False) -> Dict[str, float]:
        """
        通用交叉验证基础方法

        Args:
            model_template: 模型模板（未训练）
            X_train: 训练特征数据
            y_train: 训练目标变量
            feature_names: 特征名称列表
            kfold: 交叉验证分割器
            model_name: 模型名称（用于控制详细输出）
            use_augmentation: 是否使用数据增强

        Returns:
            Dict[str, float]: 包含r2, mae, rmse及其标准差的交叉验证结果
        """
        cv_r2_scores = []
        cv_mae_scores = []
        cv_rmse_scores = []
        show_details = model_name.lower() == "xgboost"  # 只对XGBoost显示详细信息

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            if show_details:
                print(f"       Fold {fold_idx + 1}/{self.config.cv_folds}...")

            # 分离当前fold的数据
            X_fold_train = X_train.iloc[train_idx].copy() if use_augmentation else X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx].copy() if use_augmentation else y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx].copy() if use_augmentation else X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx].copy() if use_augmentation else y_train.iloc[val_idx]

            if show_details and use_augmentation:
                print(f"         训练样本: {len(X_fold_train)}, 验证样本: {len(X_fold_val)}")

            # 根据参数决定是否进行数据增强
            if use_augmentation:
                X_fold_train_aug, y_fold_train_aug = self.data_augmentor.augment_data(
                    X_fold_train, y_fold_train, feature_names,
                    random_state=self.config.cv_random_state + fold_idx,
                    verbose=show_details
                )
                if show_details:
                    print(f"         增强后训练样本: {len(X_fold_train_aug)}")
                # 训练模型
                model = clone(model_template)
                model.fit(X_fold_train_aug, y_fold_train_aug)
            else:
                # 训练模型
                model = clone(model_template)
                model.fit(X_fold_train, y_fold_train)

            # 在原始验证集上预测
            y_fold_pred = model.predict(X_fold_val)

            # 计算当前fold的性能
            fold_metrics = self._calculate_metrics(y_fold_val, y_fold_pred)
            fold_r2 = fold_metrics['r2']
            fold_mae = fold_metrics['mae']
            fold_rmse = fold_metrics['rmse']

            if show_details:
                print(f"         Fold {fold_idx + 1} R²: {fold_r2:.4f}")

            cv_r2_scores.append(fold_r2)
            cv_mae_scores.append(fold_mae)
            cv_rmse_scores.append(fold_rmse)

        # 汇总交叉验证结果
        return {
            'r2': np.mean(cv_r2_scores),
            'mae': np.mean(cv_mae_scores),
            'rmse': np.mean(cv_rmse_scores),
            'r2_std': np.std(cv_r2_scores),
            'mae_std': np.std(cv_mae_scores),
            'rmse_std': np.std(cv_rmse_scores)
        }

    def _cross_validate_with_augmentation(self, model_template: Any, X_train: pd.DataFrame,
                                         y_train: pd.Series, feature_names: List[str],
                                         kfold: KFold, model_name: str = "") -> Dict[str, float]:
        """使用数据增强的交叉验证（防止数据泄露）

        关键特性：
        - 每个fold内独立进行数据增强，确保验证集不受训练集增强影响
        - 只对训练fold进行数据增强，验证fold保持原始状态
        - 在原始验证集上计算性能，确保评估结果的真实性

        Args:
            model_template: 模型模板（未训练）
            X_train: 训练特征数据
            y_train: 训练目标变量
            feature_names: 特征名称列表
            kfold: 交叉验证分割器
            model_name: 模型名称（用于控制详细输出）

        Returns:
            Dict[str, float]: 包含r2, mae, rmse及其标准差的交叉验证结果
        """
        return self._cross_validate_base(
            model_template, X_train, y_train, feature_names, kfold, model_name, use_augmentation=True
        )

    def _cross_validate_standard(self, model_template: Any, X_train: pd.DataFrame,
                               y_train: pd.Series, kfold: KFold, model_name: str = "") -> Dict[str, float]:
        """
        标准交叉验证（无数据增强）

        与增强版本的区别：
        - 不进行任何数据增强，直接在原始数据上训练和验证
        - 适用于数据量充足或不需要数据增强的场景
        - 计算速度更快，但可能在小数据集上表现不如增强版本

        Args:
            model_template: 模型模板（未训练）
            X_train: 训练特征数据
            y_train: 训练目标变量
            kfold: 交叉验证分割器

        Returns:
            Dict[str, float]: 包含r2, mae, rmse及其标准差的交叉验证结果
        """
        return self._cross_validate_base(
            model_template, X_train, y_train, [], kfold, model_name, use_augmentation=False
        )

    def _train_final_model(self, model_template: Any, X_train: pd.DataFrame,
                         y_train: pd.Series, feature_names: List[str], model_name: str = "") -> Any:
        """
        训练最终模型

        根据配置决定是否使用数据增强：
        - 如果启用数据增强(use_augmentation=True)：在增强后的数据上训练模型
        - 如果未启用数据增强：在原始数据上训练模型

        Args:
            model_template: 模型模板（未训练）
            X_train: 原始训练特征数据
            y_train: 原始训练目标变量
            feature_names: 特征名称列表

        Returns:
            Any: 训练好的模型实例
        """
        show_details = model_name.lower() == "xgboost"  # 只对XGBoost显示详细信息

        if self.config.use_augmentation:
            if show_details:
                print(f"       训练最终模型（使用数据增强）...")
            final_random_state = self.config.cv_random_state + 999
            X_train_aug, y_train_aug = self.data_augmentor.augment_data(
                X_train, y_train, feature_names, random_state=final_random_state,
                verbose=show_details
            )
            final_model = clone(model_template)
            final_model.fit(X_train_aug, y_train_aug)
        else:
            if show_details:
                print(f"       训练最终模型...")
            final_model = clone(model_template)
            final_model.fit(X_train, y_train)

        return final_model

    def _evaluate_final_model(self, final_model: Any, X_train: pd.DataFrame,
                              y_train: pd.Series, feature_names: List[str], model_name: str = "") -> Dict[str, float]:
        """
        评估最终模型的性能（使用与训练相同的数据增强策略）

        这个评估更准确地反映最终模型的真实性能，因为它使用了与最终模型训练
        相同的数据准备方式。
        """
        if not self.config.use_augmentation:
            # 如果没有数据增强，直接返回训练指标
            return {}

        # 使用与最终模型训练相同的数据增强策略进行评估
        show_details = model_name.lower() == "xgboost"  # 只对XGBoost显示详细信息
        if show_details:
            print("       评估最终模型性能（使用一致的数据增强策略）...")

        # 创建用于评估的交叉验证
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True,
                     random_state=self.config.cv_random_state + 1000)  # 不同的种子避免重复

        cv_r2_scores = []
        cv_mae_scores = []
        cv_rmse_scores = []

        final_random_state = self.config.cv_random_state + 999

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx].copy()
            y_fold_train = y_train.iloc[train_idx].copy()
            X_fold_val = X_train.iloc[val_idx].copy()
            y_fold_val = y_train.iloc[val_idx].copy()

            # 使用与最终模型相同的增强策略
            X_fold_train_aug, y_fold_train_aug = self.data_augmentor.augment_data(
                X_fold_train, y_fold_train, feature_names,
                random_state=final_random_state + fold_idx,
                verbose=show_details
            )

            # 训练临时模型
            temp_model = clone(final_model.__class__(**final_model.get_params()))
            temp_model.fit(X_fold_train_aug, y_fold_train_aug)

            # 在原始验证集上预测
            y_fold_pred = temp_model.predict(X_fold_val)

            # 计算性能
            fold_metrics = self._calculate_metrics(y_fold_val, y_fold_pred)
            cv_r2_scores.append(fold_metrics['r2'])
            cv_mae_scores.append(fold_metrics['mae'])
            cv_rmse_scores.append(fold_metrics['rmse'])

        return {
            'final_cv_r2': np.mean(cv_r2_scores),
            'final_cv_mae': np.mean(cv_mae_scores),
            'final_cv_rmse': np.mean(cv_rmse_scores),
            'final_cv_r2_std': np.std(cv_r2_scores),
            'final_cv_mae_std': np.std(cv_mae_scores),
            'final_cv_rmse_std': np.std(cv_rmse_scores)
        }
